from openai import OpenAI
import dtlpy as dl
import os
import logging
import json

logger = logging.getLogger("DBRX Adapter")


class ModelAdapter(dl.BaseModelAdapter):
    client = None

    def __init__(self, model_entity: dl.Model, databricks_api_key_name=None):
        self.api_key = os.environ.get(databricks_api_key_name, None)
        if self.api_key is None:
            raise ValueError(f"Missing API key: {databricks_api_key_name}")
        super().__init__(model_entity)

    def load(self, local_path, **kwargs):
        self.client = OpenAI(
          api_key=self.api_key,
          base_url="https://adb-824894719035605.5.azuredatabricks.net/serving-endpoints"
          )

    def prepare_item_func(self, item: dl.Item):
        if ('json' not in item.mimetype or
                item.metadata.get('system', dict()).get('shebang', dict()).get('dltype') != 'prompt'):
            raise ValueError('Only prompt items are supported')
        buffer = json.load(item.download(save_locally=False))
        return buffer

    def predict(self, batch, **kwargs):
        system_prompt = self.model_entity.configuration.get('system_prompt', "")

        annotations = []
        for prompt_item in batch:
            ann_collection = dl.AnnotationCollection()
            for prompt_name, prompt_content in prompt_item.get('prompts').items():
                # get latest question
                question = [p['value'] for p in prompt_content if 'text' in p['mimetype']][0]
                messages = [{"role": "system",
                             "content": system_prompt},
                            {"role": "user",
                             "content": question}]
                nearest_items = [p['nearestItems'] for p in prompt_content if 'metadata' in p['mimetype'] and
                                 'nearestItems' in p]
                if len(nearest_items) > 0:
                    nearest_items = nearest_items[0]
                    # build context
                    context = ""
                    for item_id in nearest_items:
                        context_item = dl.items.get(item_id=item_id)
                        with open(context_item.download(), 'r', encoding='utf-8') as f:
                            text = f.read()
                        context += f"\n{text}"
                    messages.append({"role": "assistant", "content": context})
                completion = self.client.chat.completions.create(
                    model="dbrx-instruct-dataloop",
                    messages=messages,
                    temperature=self.model_entity.configuration.get('temperature', 0.5),
                    top_p=self.model_entity.configuration.get('top_p', 1),
                    max_tokens=self.model_entity.configuration.get('max_tokens', 256),
                    stream=self.model_entity.configuration.get('stream', True)
                )
                full_answer = ""
                for chunk in completion:
                    if chunk.choices[0].delta.content is not None:
                        full_answer += chunk.choices[0].delta.content
                ann_collection.add(
                    annotation_definition=dl.FreeText(text=full_answer),
                    prompt_id=prompt_name,
                    model_info={
                        'name': self.model_entity.name,
                        'model_id': self.model_entity.id,
                        'confidence': 1.0
                    }
                )
            annotations.append(ann_collection)
        return annotations


if __name__ == '__main__':
    dl.setenv('prod')
    model = dl.models.get(model_id='665f009d6997dab62d56ae08')
    item = dl.items.get(item_id='665f1bb154032d32b2a81ba1')
    adapter = ModelAdapter(model, "DATABRICKS_TOKEN")
    adapter.predict_items(items=[item])
