import functools
import os
import time
import logging
from typing import Any, Callable, List, Optional, TypeVar, Dict


from langchain_core.messages import BaseMessage, ChatMessage, SystemMessage, HumanMessage, AIMessage, FunctionMessage, ToolMessage

T = TypeVar("T", bound=Callable[..., Any])

DEFAULT_EXCEPTION_MESSAGE = "An error occurred"
ENV_VAR_NOT_SET_MESSAGE = "environment variable is not set"


def get_env_variable(var_name: str, default: Optional[str] = None, raise_error: bool = True) -> Optional[str]:
    """
    Retrieves an environment variable.

    Args:
        var_name (str): The name of the environment variable to retrieve.
        default (Optional[str]): The default value to return if the variable is not found. Defaults to None.
        raise_error (bool): Flag to indicate whether to raise an error if the variable is not found. Defaults to True.

    Returns:
        Optional[str]: The value of the environment variable or the default value.

    Raises:
        EnvironmentError: If raise_error is True and the environment variable is not set.
    """
    value = os.getenv(var_name)
    if value is None and raise_error:
        raise EnvironmentError(f"Error: {var_name} {ENV_VAR_NOT_SET_MESSAGE}.")
    return value or default


def time_function(func: T) -> Any | T:
    """
    Decorator to measure the execution time of a function.

    Args:
        func (Callable): The function to measure.

    Returns:
        Callable: A wrapper function that adds execution time measurement to the input function.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter() - start_time
        logging.info(f"Time taken by {func.__name__}: {end_time:.6f} seconds")
        return result

    return wrapper


def log_exception(exception: Exception, message: str = DEFAULT_EXCEPTION_MESSAGE) -> None:
    """
    Logs exceptions with a custom message.

    Args:
        exception (Exception): The exception to log.
        message (str): Custom message to prepend to the exception message. Defaults to a standard error message.
    """
    logging.error(f"{message}: {exception}")


def convert_message_to_dict(message: BaseMessage) -> dict:
    """Convert a LangChain message to a dictionary."""
    message_dict: Dict[str, Any] = {"content": message.content}
    if (name := message.additional_kwargs.get("name")) is not None:
        message_dict["name"] = name

    # populate role and additional message data
    if isinstance(message, ChatMessage):
        message_dict["role"] = message.role
    elif isinstance(message, HumanMessage):
        message_dict["role"] = "user"
    elif isinstance(message, AIMessage):
        message_dict["role"] = "assistant"
        if "function_call" in message.additional_kwargs:
            message_dict["function_call"] = message.additional_kwargs["function_call"]
        if "tool_calls" in message.additional_kwargs:
            message_dict["tool_calls"] = message.additional_kwargs["tool_calls"]
            tool_call_supported_props = {"id", "type", "function"}
            message_dict["tool_calls"] = [{k: v for k, v in tool_call.items() if k in tool_call_supported_props} for tool_call in message_dict["tool_calls"]]
        if "function_call" in message_dict or "tool_calls" in message_dict:
            message_dict["content"] = message_dict["content"] or None
    elif isinstance(message, SystemMessage):
        message_dict["role"] = "system"
    elif isinstance(message, FunctionMessage):
        message_dict["role"] = "function"
    elif isinstance(message, ToolMessage):
        message_dict["role"] = "tool"
        message_dict["tool_call_id"] = message.tool_call_id

        supported_props = {"content", "role", "tool_call_id"}
        message_dict = {k: v for k, v in message_dict.items() if k in supported_props}
    else:
        raise TypeError(f"Got unknown type {message}")
    return message_dict


def convert_messages_to_dicts(messages: List[BaseMessage]) -> List[Dict]:
    """Converts a list of `BaseMessage` objects to a list of dictionaries."""
    return [convert_message_to_dict(m) for m in messages]
