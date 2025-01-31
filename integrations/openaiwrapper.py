from typing import Union, List, Dict, Any, Optional, Sequence, Callable, Type
import time
import logging

from langchain_core.messages import BaseMessage, AIMessage

from langchain_core.tools import BaseTool
from langchain_core.language_models import LanguageModelInput
from langchain_core.runnables import Runnable, RunnableMap, RunnablePassthrough
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.output_parsers.openai_tools import (
    JsonOutputKeyToolsParser,
    PydanticToolsParser,
)

from langchain_core.utils.function_calling import convert_to_openai_tool
from operator import itemgetter

# Replace the openai import with langchain_openai
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import SecretStr, BaseModel

from llm_wrapper import LLMAPIWrapper

RETRY_SLEEP_DURATION = 1  # seconds
MAX_RETRIES = 5


class OpenAIAPIWrapper(LLMAPIWrapper):
    """
    OpenAI Language Model API wrapper using LangChain's OpenAI classes.

    Implements the generic LLM wrapper and supports tool/function calling and structured output.
    """

    def __init__(
        self,
        api_key: str,
        engine: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 10,
        use_dummy_response: bool = False,
        **kwargs: Any,
    ):
        """Initialize OpenAI wrapper and store tools if any are passed."""
        super().__init__(api_key, timeout, model)
        self.engine = engine
        self.use_dummy_response = use_dummy_response
        self.model_kwargs = kwargs

        # Use ChatOpenAI for chat completions
        self.chat_client = ChatOpenAI(
            api_key=SecretStr(api_key),
            model=model or self.default_model_name(),
            timeout=timeout,
            **kwargs,
        )
        # Use OpenAIEmbeddings for embeddings
        self.embedding_client = OpenAIEmbeddings(
            api_key=SecretStr(api_key),
            model=engine or "text-embedding-ada-002",  # Default embedding model
            timeout=timeout,
        )

    def default_model_name(self) -> str:
        """Returns the default model name for the OpenAI wrapper."""
        return "gpt-4-1106-preview"

    def get_embedding(self, text: str, **kwargs: Any) -> Union[List[float], Any]:
        """
        Retrieves the embedding for the given text using OpenAI's API.

        :param text: The text for which embedding is required.
        :return: The embedding for the given text.
        """
        if self.use_dummy_response:
            return {"data": [{"embedding": [0.1, 0.2, 0.3]}]}  # Dummy response

        start_time = time.time()
        retries = 0

        while time.time() - start_time < self.timeout:
            try:
                # Use LangChain's OpenAIEmbeddings for embeddings
                return self.embedding_client.embed_query(text)
            except Exception as e:
                logging.error(f"OpenAI API error: {e}")
                retries += 1
                if retries >= MAX_RETRIES:
                    raise
                time.sleep(RETRY_SLEEP_DURATION)

                if "Rate limit" in str(e):
                    print("Rate limit reached... sleeping for 20 seconds")
                    start_time += 20
                    time.sleep(20)
        raise TimeoutError("API call timed out")

    def chat_completion(self, messages: List[BaseMessage], **kwargs: Any) -> Union[AIMessage, Any]:
        """
        Generates a chat completion using OpenAI's API.

        :param messages: A list of BaseMessage objects representing the conversation.
        :param kwargs: Additional keyword arguments for the chat completion API call.
        :return: The result of the chat completion API call.
        """
        if self.use_dummy_response:
            return AIMessage(content="This is a dummy response.")  # Dummy response

        if "model" not in kwargs:
            kwargs["model"] = self.model_name

        start_time = time.time()
        retries = 0

        while time.time() - start_time < self.timeout:
            try:
                return self.chat_client.invoke(
                    input=messages,
                    **kwargs,
                )
            except Exception as e:
                logging.error(f"OpenAI API error: {e}")
                retries += 1
                if retries >= MAX_RETRIES:
                    raise
                time.sleep(RETRY_SLEEP_DURATION)

                if "Rate limit reached" in str(e):
                    print("Rate limit reached... sleeping for 20 seconds")
                    start_time += 20
                    time.sleep(20)
        raise TimeoutError("API call timed out")

    def _bind(
        self,
        tools: Optional[Sequence[Union[Dict[str, Any], Type, Callable, BaseTool]]] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tools to the LLM for tool calling."""
        # If no tools or functions are provided, pass through the call
        if tools is None and "functions" not in kwargs and "tools" not in kwargs:
            return self.chat_client

        # Ensure the required parameters are not disabled
        if "tools" in kwargs or tools is not None:
            kwargs = self._validate_params(
                tools=tools,
                parallel_tool_calls=kwargs.get("parallel_tool_calls", True),
                tool_choice=kwargs.get("tool_choice"),
                **kwargs,
            )
        if "functions" in kwargs:
            kwargs = self._validate_params(
                functions=kwargs["functions"],
                function_call=kwargs.get("function_call"),
                **kwargs,
            )

        bound_llm: Runnable = self.chat_client.bind(**kwargs)

        return bound_llm

    def _with_structured_output(
        self,
        schema: Union[Dict[str, Any], Type[BaseModel]],
        method: str = "function_calling",
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, Union[Dict, BaseModel]]:
        """Enable structured output generation based on a given schema."""
        if method == "function_calling":
            if isinstance(schema, type) and issubclass(schema, BaseModel):
                schema_name = schema.__name__
            elif isinstance(schema, dict) and "name" in schema:
                schema_name = schema["name"]
            else:
                raise ValueError("Schema must be a pydantic.BaseModel or a dictionary with a 'name' key when method is 'function_calling'")

            llm = self._bind(tools=[schema])

            output_parser = PydanticToolsParser(tools=[schema], first_tool_only=True) if isinstance(schema, type) and issubclass(schema, BaseModel) else JsonOutputKeyToolsParser(key_name=schema_name, first_tool_only=True)
        elif method == "json_mode":
            llm = self._bind(response_format={"type": "json_object"})
            output_parser = PydanticOutputParser(pydantic_object=schema) if isinstance(schema, type) and issubclass(schema, BaseModel) else JsonOutputParser()
        else:
            raise ValueError("Invalid method specified. Choose 'function_calling' or 'json_mode'.")

        if include_raw:
            return RunnableMap(
                raw=llm,
                parsed=lambda x: itemgetter(x) | output_parser,
                parsing_error=lambda _: None,
            )
        else:
            return llm | output_parser

    def _validate_params(self, **kwargs: Any) -> Dict[str, Any]:
        """Ensure the parameters for OpenAI calls are not disabled."""
        # Placeholder for any validation logic needed
        return kwargs
