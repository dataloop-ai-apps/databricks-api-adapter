import dtlpy as dl
import logging
import openai
from openai import NOT_GIVEN
import os
from typing import List, Dict


logger = logging.getLogger("databricks-chat-completion-model-adapter")


class ModelAdapter(dl.BaseModelAdapter):
    def load(self, local_path, **kwargs):
        """Load configuration for Databricks adapter"""
        api_key = os.environ.get("DATABRICKS_API_KEY")
        self.base_url = self.configuration.get("base_url")
        self.adapter_defaults.upload_annotations = False

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

        self.client = openai.OpenAI(api_key=api_key, base_url=self.base_url)

    def call_model(self, messages: List[Dict]):
        model_name = self.configuration.get("model_name", None)
        stream = self.configuration.get("stream", True)
        max_tokens = self.configuration.get("max_tokens", NOT_GIVEN)
        temperature = self.configuration.get("temperature", NOT_GIVEN)
        top_p = self.configuration.get("top_p", NOT_GIVEN)

        if not model_name:
            raise ValueError("Configuration error: 'model_name' is required.")
        if max_tokens is NOT_GIVEN:
            logger.warning("max_tokens not set. Using default model value")
        if temperature is NOT_GIVEN:
            logger.warning("temperature not set. Using default model value")
        if top_p is NOT_GIVEN:
            logger.warning("top_p not set. Using default model value")

        response = self.client.chat.completions.create(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=stream,
            model=model_name,
        )
        if stream:
            for chunk in response:
                yield chunk.choices[0].delta.content or ""
        else:
            yield response.choices[0].message.content or ""

    def prepare_item_func(self, item: dl.Item) -> dl.PromptItem:
        prompt_item = dl.PromptItem.from_item(item)
        return prompt_item

    def predict(self, batch: List[dl.PromptItem], **kwargs):
        system_prompt = self.configuration.get("system_prompt", "")
        model_entity_name = self.model_entity.name
        for prompt_item in batch:
            # Databricks are not able to take mimetypes
            # So we need to convert the messages to the required format
            _messages = prompt_item.to_messages(model_name=model_entity_name)
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
            stream_response = self.call_model(messages=messages)
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
