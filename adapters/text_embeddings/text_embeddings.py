import openai
import dtlpy as dl
import os
import logging

logger = logging.getLogger("databricks-text-embeddings")


class ModelAdapter(dl.BaseModelAdapter):
    def load(self, local_path, **kwargs):
        """Load configuration for Databricks adapter"""
        api_key = os.environ.get("DATABRICKS_API_KEY")
        if not api_key:
            raise ValueError(
                "Environment variable 'DATABRICKS_API_KEY' is not set. "
                "Please set it to proceed."
            )
        self.base_url = self.configuration.get("base_url")
        self.model_name = self.configuration.get("model_name")

        if not self.base_url:
            raise ValueError("Configuration error: 'base_url' is required.")
        if self.base_url == "<insert-dbrx-endpoint-url>":
            raise ValueError(
                "Configuration error: 'base_url' must be replaced with your actual Databricks endpoint URL."
            )
        if not self.model_name:
            raise ValueError("Configuration error: 'model_name' is required.")

        self.client = openai.OpenAI(api_key=api_key, base_url=self.base_url)

    def embed(self, batch, **kwargs):
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
                            model_name=self.configuration.get("hyde_model_name")
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

            response = self.client.embeddings.create(
                input=text,
                model=self.model_name,
                encoding_format=self.model_entity.configuration.get(
                    "encoding_format", "base64"
                ),  # base64 is default
            )
            embedding = response.data[0].embedding
            logger.info(f"Extracted embeddings for text {item}: {embedding}")
            embeddings.append(embedding)

        return embeddings
