from typing import Dict, Any, List, Optional, Union, Tuple
import time
import logging

class LLMAPIWrapper:
    """
    Generic Language Model API wrapper.

    Implement other LLMs here, as well as dummy responses
    """

    def __init__(self, api_key: Optional[str] = None, timeout: int = 10, model_name: Optional[str] = None):
        """
        Initializes the LLMAPIWrapper instance.

        :param api_key: The API key for the language model.
        :param timeout: The timeout duration in seconds for API requests.
        :param model_name: The specific model to be used
        """
        self.api_key = api_key
        self.timeout = timeout
        self.model_name = model_name or self.default_model_name()

    def default_model_name(self) -> str:
        """
        Returns the default model name for the wrapper.
        """
        raise NotImplementedError("Subclasses must implement default_model_name method")

    def get_embedding(self, text: str) -> Union[Dict, Any]:
        """
        Retrieves the embedding for the given text.

        :param text: The text for which embedding is required.
        :return: The embedding for the given text.
        """
        raise NotImplementedError("Subclasses must implement get_embedding method")

    def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Generates a chat completion using the specified model.

        :param messages: A list of message dictionaries with 'role' and 'content' keys.
        :param kwargs: Additional keyword arguments for the chat completion API call.
        :return: The result of the chat completion API call.
        """
        raise NotImplementedError("Subclasses must implement chat_completion method")