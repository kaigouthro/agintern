import threading
import time
import os
from typing import List

from dotenv import load_dotenv
from colorama import Fore, Style

from agents.microagent_manager import MicroAgentManager
from agents.agent_registry import AgentRegistry
from agents.task_decomposition import SimpleDecomposition
from utils.utility import get_env_variable, time_function
from utils.ui import clear_console, display_agent_info, print_final_output, format_text
from integrations.openaiwrapper import OpenAIAPIWrapper

# Constants
QUESTION_SET = ["What is 5+9?", "What is the population of Thailand?", "What is the population of Sweden?", "What is the population of Sweden and Thailand combined?"]


def initialize_manager(api_key: str, use_dummy_response: bool = False) -> MicroAgentManager:
    """
    Initialize and return the MicroAgentManager with the given API key.
    """
    llm_wrapper = OpenAIAPIWrapper(api_key, get_env_variable("OPENAI_EMBEDDING", "text-embedding-ada-002", False), get_env_variable("OPENAI_MODEL", "gpt-4-1106-preview", False), use_dummy_response=use_dummy_response)
    agent_registry = AgentRegistry()
    manager = MicroAgentManager(llm_wrapper, agent_registry=agent_registry)
    manager.create_agents()
    return manager


@time_function
def process_user_input(manager: MicroAgentManager, user_input: str) -> str:
    """
    Processes a single user input and generates a response.
    """
    agent = manager.get_or_create_agent("Bootstrap Agent", depth=1, sample_input=user_input)
    return agent.respond(user_input)


def process_questions(manager: MicroAgentManager, outputs: List[str]):
    """
    Process each question in the QUESTION_SET and append outputs.
    """
    for question_number, user_input in enumerate(QUESTION_SET, start=1):
        response = process_user_input(manager, user_input)
        output_text = format_text(question_number, user_input, response)
        outputs.append(output_text)


def main():
    load_dotenv()
    api_key = get_env_variable("OPENAI_KEY")

    if not api_key:
        print(f"{Fore.RED}🚫 Error: OPENAI_KEY environment variable is not set.{Style.RESET_ALL}")
        return

    use_dummy_response = True
    manager = initialize_manager(api_key, use_dummy_response)

    outputs = []
    stop_event = threading.Event()
    display_thread = threading.Thread(target=display_agent_info, args=(manager, stop_event, outputs))
    display_thread.start()

    try:
        process_questions(manager, outputs)
    finally:
        time.sleep(5)
        stop_event.set()
        print_final_output(outputs, manager)
        display_thread.join()


if __name__ == "__main__":
    main()
