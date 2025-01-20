from typing import Dict, Any, List, Optional, Union, Tuple, Type, Sequence, Callable
import time
import logging
from abc import ABC, abstractmethod

from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
    AIMessage,
    FunctionMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.language_models import LanguageModelInput
from langchain_core.runnables import Runnable
from typing import TypeAlias


ModelInput: TypeAlias = LanguageModelInput
ModelOutput: TypeAlias = Union[Dict, BaseModel]
StructuredRunnable: TypeAlias = Runnable[ModelInput, ModelOutput]


class LLMAPIWrapper(ABC):
    """
    Abstract base class for Language Model API wrappers.

    This class provides a standardized interface for interacting with various LLMs,
    ensuring consistent behavior across different models and providers.

    Key Features:
        - Abstract methods for core LLM functionalities:
            - `default_model_name`: Specifies the default model for the wrapper.
            - `get_embedding`: Retrieves text embeddings.
            - `chat_completion`: Generates chat completions.
        - **Tool/Function Calling Support:**
            - `bind_tools`: Method to bind tools to the LLM for structured interactions.
            - `with_structured_output`: Enables structured output based on a given schema.
        - Enforces implementation of these methods in derived classes.
        - Handles common parameters like API key, timeout, and model name.
        - Provides logging for API interactions.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: int = 10,
        model_name: Optional[str] = None,
    ):
        """
        Initializes the LLMAPIWrapper.

        Args:
            api_key: The API key for the language model.
            timeout: The timeout duration in seconds for API requests.
            model_name: The specific model to be used. If None, defaults to `default_model_name`.
        """
        self.api_key = api_key
        self.timeout = timeout
        self.model_name = model_name or self.default_model_name()
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def default_model_name(self) -> str:
        """
        Returns the default model name for the wrapper.

        This method must be implemented by subclasses to specify the default model
        used when no model is explicitly provided during initialization.
        """
        ...

    @abstractmethod
    def get_embedding(self, text: str, **kwargs: Any) -> Union[List[float], Any]:
        """
        Retrieves the embedding for the given text.

        Args:
            text: The text for which to retrieve the embedding.
            kwargs: Additional keyword arguments specific to the embedding API.

        Returns:
            The embedding for the given text, typically as a list of floats.
            The exact return type may vary depending on the specific LLM API.
        """
        ...

    @abstractmethod
    def chat_completion(self, messages: List[BaseMessage], **kwargs: Any) -> Union[AIMessage, Any]:
        """
        Generates a chat completion for a sequence of messages.

        Args:
            messages: A list of `BaseMessage` objects representing the conversation history.
                      Supported message types include `SystemMessage`, `HumanMessage`, `AIMessage`,
                      `FunctionMessage`, and `ToolMessage`.
            kwargs: Additional keyword arguments specific to the chat completion API,
                    potentially including parameters for tool/function calling.

        Returns:
            The generated chat completion, typically as an `AIMessage` object.
            The exact return type may vary depending on the specific LLM API.
        """
        ...

    @abstractmethod
    def execute_with_tools_or_schema(self, tools: Optional[Sequence[Union[Dict[str, Any], Type, Callable, BaseTool]]] = None, schema: Optional[Union[Dict[str, Any], Type[BaseModel]]] = None, **kwargs: Any) -> Union[Runnable[ModelInput, BaseMessage], StructuredRunnable]:
        """Single abstract method for both tool binding and schema handling"""
        ...

    def bind_tools(self, tools: Sequence[Union[Dict[str, Any], Type, Callable, BaseTool]], **kwargs: Any) -> Union[Runnable[ModelInput, BaseMessage], StructuredRunnable]:
        """
        Binds tools to the chat model for structured interactions.

        Args:
            tools: A list of tool definitions to bind.
                   Supported types: dict, BaseModel subclass, callable, or BaseTool.
            kwargs: Additional keyword arguments to customize tool binding
                    (e.g., tool_choice, strict).

        Returns:
            A `Runnable` object that can be used to invoke the model with the bound tools.
        """

        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        return self.execute_with_tools_or_schema(tools=formatted_tools, **kwargs)

    def with_structured_output(self, schema: Union[Dict[str, Any], Type[BaseModel]], **kwargs: Any) -> Union[Runnable[ModelInput, BaseMessage], StructuredRunnable]:
        return self.execute_with_tools_or_schema(schema=schema, **kwargs)
