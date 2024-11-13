from openai import OpenAI
import openai
import dtlpy as dl
import os
import logging

logger = logging.getLogger("BaseChatCompletionModelAdapter")


class ModelAdapter(dl.BaseModelAdapter):

    def load(self, local_path, **kwargs):
        api_key = os.environ.get("DATABRICKS_API_KEY", None)
        if api_key is None:
            raise ValueError("Missing API key: DATABRICKS_API_KEY")

        self.client = OpenAI(
            api_key=api_key,
            base_url=self.model_entity.configuration.get("base_url")
        )

    def prepare_item_func(self, item: dl.Item):
        prompt_item = dl.PromptItem.from_item(item)
        return prompt_item

    def stream_response(self, messages):
        stream = self.configuration.get("stream", True)
        response = self.client.chat.completions.create(
            messages=messages,
            max_tokens=self.configuration.get("max_tokens", openai.NOT_GIVEN),
            temperature=self.configuration.get("temperature", openai.NOT_GIVEN),
            top_p=self.configuration.get("top_p", openai.NOT_GIVEN),
            stream=stream,
            model=self.model_entity.configuration.get("databricks_model_name"),
        )
        if stream:
            for chunk in response:
                yield chunk.choices[0].delta.content or ""
        else:
            yield response.choices[0].message.content or ""

    def predict(self, batch, **kwargs):
        system_prompt = self.model_entity.configuration.get('system_prompt', "")

        for prompt_item in batch:
            # Get all messages including model annotations
            _messages = prompt_item.to_messages(model_name=self.model_entity.name)
            messages = self.reformat_messages(_messages)

            messages.insert(0, {"role": "system",
                                "content": system_prompt})

            nearest_items = prompt_item.prompts[-1].metadata.get('nearestItems', [])
            if len(nearest_items) > 0:
                context = prompt_item.build_context(nearest_items=nearest_items,
                                                    add_metadata=self.configuration.get("add_metadata"))
                logger.info(f"Nearest items Context: {context}")
                messages.append({"role": "assistant", "content": context})

            stream_response = self.stream_response(messages=messages)
            response = ""
            for chunk in stream_response:
                #  Build text that includes previous stream
                response += chunk
                prompt_item.add(message={"role": "assistant",
                                         "content": [{"mimetype": dl.PromptType.TEXT,
                                                      "value": response}]},
                                stream=True,
                                model_info={'name': self.model_entity.name,
                                            'confidence': 1.0,
                                            'model_id': self.model_entity.id})

        return []

    @staticmethod
    def reformat_messages(messages):
        """
        Convert SDK message format to the required format.

        :param messages: A list of messages in the OpenAI format (default by SDK).
        :return: A list of messages reformatted.
        """
        reformat_messages = list()
        for message in messages:
            content = message["content"]
            question = content[0][content[0].get("type")]
            role = message["role"]

            reformat_message = {"role": role, "content": question}
            reformat_messages.append(reformat_message)

        return reformat_messages
