import dtlpy as dl
import logging
import openai
import os

logger = logging.getLogger("BaseEmbeddingsModelAdapter")


class ModelAdapter(dl.BaseModelAdapter):
    def load(self, local_path, **kwargs):
        """Load configuration for Databricks adapter"""
        api_key = os.environ.get("DATABRICKS_API_KEY")
        base_url = self.configuration.get("base_url")

        if not api_key:
            raise ValueError("Missing API key.")
        if not base_url:
            raise ValueError("Configuration error: 'base_url' is required.")
        if base_url == "<insert-dbrx-endpoint-url>":
            raise ValueError(
                "Configuration error: 'base_url' must be replaced with your actual Databricks endpoint URL."
            )

        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)

    def call_model(self, text):
        model_name = self.configuration.get("model_name")
        encoding_format = self.configuration.get("encoding_format")
        if model_name is None:
            raise ValueError("Configuration error: 'model_name' is required.")
        if encoding_format is None:
            logger.warning("encoding_format not set. Using default model value")

        response = self.client.embeddings.create(
            input=text,
            model=model_name,
            encoding_format=encoding_format,
        )
        return response.data[0].embedding

    def embed(self, batch, **kwargs):
        hyde_model_name = self.configuration.get('hyde_model_name')
        embeddings = []
        for item in batch:
            if isinstance(item, str):
                self.adapter_defaults.upload_features = True
                text = item
            else:
                self.adapter_defaults.upload_features = False
                try:
                    prompt_item = dl.PromptItem.from_item(item)
                    is_hyde = item.metadata.get("prompt", dict()).get("is_hyde", False)
                    if is_hyde is True:
                        messages = prompt_item.to_messages(
                            model_name=hyde_model_name
                        )[-1]
                        if messages["role"] == "assistant":
                            text = messages["content"][-1]["text"]
                        else:
                            raise ValueError(
                                "Only assistant messages are supported for hyde model"
                            )
                    else:
                        messages = prompt_item.to_messages(include_assistant=False)[-1]
                        text = messages["content"][-1]["text"]

                except ValueError as e:
                    raise ValueError(
                        f"Only mimetype text or prompt items are supported {e}"
                    )

            embedding = self.call_model(text)
            logger.info(f"Extracted embeddings for text {item}: {embedding}")
            embeddings.append(embedding)

        return embeddings
