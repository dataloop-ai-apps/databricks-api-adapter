import dtlpy as dl
import logging
import openai
import os
from typing import List, Dict


logger = logging.getLogger("databricks-chat-completion-model-adapter")


@dl.Package.decorators.module(
    name="databricks-chat-completion-model-adapter",
    description="Chat Completion Model Adapter for Databricks models",
    init_inputs={"model_entity": dl.Model},
)
class ModelAdapter(dl.BaseModelAdapter):
    def load(self, local_path, **kwargs):
        """Load configuration for Databricks adapter"""
        api_key = os.environ.get("DATABRICKS_API_KEY")
        self.base_url = self.configuration.get("base_url")
        self.model_name = self.configuration.get("model_name")
        self.adapter_defaults.upload_annotations = False
        self.stream = self.configuration.get("stream", True)
        self.max_tokens = self.configuration.get("max_tokens", openai.NOT_GIVEN)
        self.temperature = self.configuration.get("temperature", openai.NOT_GIVEN)
        self.top_p = self.configuration.get("top_p", openai.NOT_GIVEN)
        self.system_prompt = self.configuration.get("system_prompt", "")

        if not api_key:
            raise ValueError(
                "Environment variable 'DATABRICKS_API_KEY' is not set. "
                "Please set it to proceed."
            )
        if not self.base_url:
            raise ValueError("Configuration error: 'base_url' is required.")
        if self.base_url == "<insert-dbrx-endpoint-url>":
            raise ValueError(
                "Configuration error: 'base_url' must be replaced with your actual Databricks endpoint URL."
            )
        if not self.model_name:
            raise ValueError("Configuration error: 'model_name' is required.")

        if self.max_tokens is openai.NOT_GIVEN:
            logger.warning("max_tokens not set. Using default model value")
        if self.temperature is openai.NOT_GIVEN:
            logger.warning("temperature not set. Using default model value")
        if self.top_p is openai.NOT_GIVEN:
            logger.warning("top_p not set. Using default model value")
        if not self.system_prompt:
            logger.warning("system_prompt not set. Using empty string")

        self.client = openai.OpenAI(api_key=api_key, base_url=self.base_url)

    def prepare_item_func(self, item: dl.Item) -> dl.PromptItem:
        prompt_item = dl.PromptItem.from_item(item)
        return prompt_item

    def stream_response(self, messages: List[Dict]):
        response = self.client.chat.completions.create(
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            stream=self.stream,
            model=self.model_name,
        )
        if self.stream:
            for chunk in response:
                yield chunk.choices[0].delta.content or ""
        else:
            yield response.choices[0].message.content or ""

    def predict(self, batch: List[dl.PromptItem], **kwargs):
        system_prompt = self.system_prompt
        for prompt_item in batch:
            # Databricks are not able to take mimetypes
            # So we need to convert the messages to the required format
            _messages = prompt_item.to_messages(model_name=self.model_entity.name)
            messages = self.reformat_messages(_messages)
            messages.insert(0, {"role": "system", "content": system_prompt})
            nearest_items = prompt_item.prompts[-1].metadata.get("nearestItems", [])
            if len(nearest_items) > 0:
                context = prompt_item.build_context(
                    nearest_items=nearest_items,
                    add_metadata=self.configuration.get("add_metadata"),
                )
                logger.info(f"Nearest items Context: {context}")
                messages.append({"role": "assistant", "content": context})
            stream_response = self.stream_response(messages=messages)
            response = ""
            for chunk in stream_response:
                #  Build text that includes previous stream
                response += chunk
                prompt_item.add(
                    message={
                        "role": "assistant",
                        "content": [
                            {"mimetype": dl.PromptType.TEXT, "value": response}
                        ],
                    },
                    stream=True,
                    model_info={
                        "name": self.model_entity.name,
                        "confidence": 1.0,
                        "model_id": self.model_entity.id,
                    },
                )

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
