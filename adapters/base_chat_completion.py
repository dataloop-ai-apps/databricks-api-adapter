from openai import OpenAI, NOT_GIVEN
import dtlpy as dl
import os
import logging

logger = logging.getLogger("BaseChatCompletionModelAdapter")


class ModelAdapter(dl.BaseModelAdapter):

    def load(self, local_path, **kwargs):
        api_key = os.environ.get("DATABRICKS_API_KEY")
        base_url = self.model_entity.configuration.get("base_url")

        if api_key is None:
            raise ValueError("Missing API key.")
        if base_url is None:
            raise ValueError("Configuration error: 'base_url' is required.")
        if base_url == "<insert-dbrx-endpoint-url>":
            raise ValueError(
                "Configuration error: 'base_url' must be replaced with your actual Databricks endpoint URL."
            )
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

    def prepare_item_func(self, item: dl.Item):
        prompt_item = dl.PromptItem.from_item(item)
        return prompt_item

    def stream_response(self, messages):
        stream = self.configuration.get("stream", True)
        extra_headers = {
            "User-Agent": "integration/Dataloop",
            "Dtlpy-Model": f"{self.model_entity.name}/0.0.1"
        }

        response = self.client.chat.completions.create(
            messages=messages,
            max_tokens=self.configuration.get("max_tokens", NOT_GIVEN),
            temperature=self.configuration.get("temperature", NOT_GIVEN),
            top_p=self.configuration.get("top_p", NOT_GIVEN),
            stream=stream,
            model=self.model_entity.configuration.get("databricks_model_name"),
            extra_headers=extra_headers
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
            role = message.get("role")
            message_content = ""
            for content in message.get("content"):
                type = content.get("type")
                if 'text' in type:
                    question = content.get(type)
                    message_content += question
                else:
                    logger.warning("Multimodal options is not supported.")

            reformat_message = {"role": role, "content": message_content}
            reformat_messages.append(reformat_message)

        return reformat_messages
