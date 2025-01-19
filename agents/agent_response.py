import logging
from typing import Dict, Any, List, Optional
from integrations.llm_wrapper import LLMAPIWrapper
from prompt_management.prompts import (
    REACT_STEP_POST, REACT_STEP_PROMPT, REACT_SYSTEM_PROMPT, REACT_PLAN_PROMPT, STATIC_PRE_PROMPT, STATIC_PRE_PROMPT_PRIME, REACT_STEP_PROMPT_PRIME, REACT_STEP_POST_PRIME
)
from agents.capabilities import Ability, CodeExecutionAbility, FileSystemAbility, DiffAbility, GitAbility, PythonFunctionAbility

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

class AgentResponse:
    def __init__(self, llm_wrapper: LLMAPIWrapper, manager: Any, agent: Any, creator: Any, depth: int, abilities: Dict[str, Ability]):
        """
        Initializes the AgentResponse class.

        :param llm_wrapper: The LLM API wrapper.
        :param manager: The MicroAgentManager instance.
        :param agent: The agent instance.
        :param creator: The agent creator instance.
        :param depth: The current depth of the agent.
        :param abilities: The dictionary of abilities available to the agent.
        """
        self.llm_wrapper = llm_wrapper
        self.manager = manager
        self.agent = agent
        self.creator = creator
        self.depth = depth
        self.abilities = abilities

    def generate_response(self, input_text: str, dynamic_prompt: str, max_depth: int) -> Tuple[str, str, bool, int]:
        """
        Generates a response for the given input text.

        :param input_text: The input text to respond to.
        :param dynamic_prompt: The dynamic prompt for the agent.
        :param max_depth: The maximum depth for recursion.
        :return: A tuple containing the response, conversation history, solution flag, and number of iterations.
        """
        runtime_context = self._generate_runtime_context(dynamic_prompt)
        system_prompt = self._compose_system_prompt(runtime_context, dynamic_prompt)
        conversation_accumulator = ""
        thought_number = 0
        action_number = 0
        found_new_solution = False

        for _ in range(max_depth):
            react_prompt = self._build_react_prompt(input_text, conversation_accumulator, thought_number, action_number)
            self.agent.update_status(f'Thinking .. (Iteration #{thought_number})')
            response = self._generate_chat_response(system_prompt, react_prompt)
            conversation_accumulator, thought_number, action_number = self._process_response(
                response, conversation_accumulator, thought_number, action_number, input_text
            )

            if "Query Solved" in response:
                found_new_solution = True
                break

        return self._conclude_output(conversation_accumulator), conversation_accumulator, found_new_solution, thought_number

    def _compose_system_prompt(self, runtime_context: str, dynamic_prompt: str) -> str:
        """
        Composes the system prompt based on the agent's type and runtime context.

        :param runtime_context: The runtime context for the agent.
        :param dynamic_prompt: The dynamic prompt for the agent.
        :return: The composed system prompt.
        """
        pre_prompt = STATIC_PRE_PROMPT_PRIME if self.agent.is_prime else STATIC_PRE_PROMPT
        return f"{pre_prompt} {runtime_context} {dynamic_prompt}
DELIVER THE NEXT PACKAGE."

    def _generate_runtime_context(self, dynamic_prompt: str) -> str:
        """
        Generates the runtime context for the agent.

        :param dynamic_prompt: The dynamic prompt for the agent.
        :return: The generated runtime context.
        """
        available_agents = [agent for agent in self.manager.agents if agent.purpose != "Bootstrap Agent"]
        available_agents_info = ', '.join([f"{agent.name} (depth={agent.depth})" for agent in available_agents])
        return f"Your Purpose: {dynamic_prompt}. Available agents (Feel free to invent new ones if required!): {available_agents_info}."

    def _build_react_prompt(self, input_text: str, conversation_accumulator: str, thought_number: int, action_number: int) -> str:
        """
        Builds the ReAct prompt for the agent.

        :param input_text: The input text to respond to.
        :param conversation_accumulator: The accumulated conversation history.
        :param thought_number: The current thought number.
        :param action_number: The current action number.
        :return: The constructed ReAct prompt.
        """
        thought_prompt = REACT_STEP_PROMPT_PRIME if self.agent.is_prime else REACT_STEP_PROMPT
        action_prompt = REACT_STEP_POST_PRIME if self.agent.is_prime else REACT_STEP_POST
        return (
            f"Question: {input_text}
"
            f"{conversation_accumulator}
"
            f"Thought {thought_number}: {thought_prompt}
"
            f"Action {action_number}: {action_prompt}"
        )

    def _generate_chat_response(self, system_prompt: str, react_prompt: str) -> str:
        """
        Generates a chat response using the LLM API.

        :param system_prompt: The system prompt for the agent.
        :param react_prompt: The ReAct prompt for the agent.
        :return: The generated chat response.
        """
        return self.llm_wrapper.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": react_prompt}
            ]
        )

    def _process_response(self, response: str, conversation_accumulator: str, thought_number: int, action_number: int, input_text: str) -> Tuple[str, int, int]:
        """
        Processes the response generated by the LLM API.

        :param response: The response generated by the LLM API.
        :param conversation_accumulator: The accumulated conversation history.
        :param thought_number: The current thought number.
        :param action_number: The current action number.
        :param input_text: The original input text.
        :return: A tuple containing the updated conversation accumulator, thought number, and action number.
        """
        conversation_accumulator += f"
{response}"
        thought_number += 1
        action_number += 1

        if "```python" in response:
            self.agent.update_status('Executing Python code')
            self.agent.number_of_code_executions += 1
            code_execution_ability = self.agent.get_ability('CodeExecutionAbility')
            if code_execution_ability:
                exec_response = code_execution_ability.execute(self.agent, text_with_code=response)
                conversation_accumulator += f"
Observation: Executed Python code
Output: {exec_response}"

        if "Use Agent[" in response:
            agent_name, input_text = self._parse_agent_info(response)
            self.agent.update_active_agents(self.agent.name, agent_name)
            self.agent.update_status('Waiting for Agent')
            delegated_agent = self.creator.get_or_create_agent(agent_name, depth=self.depth + 1, sample_input=input_text)
            delegated_response = delegated_agent.respond(input_text)
            conversation_accumulator += f"
Output {thought_number}: Delegated task to Agent {agent_name}
Output of Agent: {action_number}: {delegated_response}"

        return conversation_accumulator, thought_number, action_number

    def _parse_agent_info(self, response: str) -> Tuple[str, str]:
        """
        Parses the agent information from the response.

        :param response: The response generated by the LLM API.
        :return: A tuple containing the agent name and input text.
        """
        agent_info = response.split('Use Agent[')[1].split(']')[0]
        agent_name, input_text = (agent_info.split(":") + [""])[:2]
        return agent_name.strip(), input_text.strip()

    def _conclude_output(self, conversation: str) -> str:
        """
        Concludes the output based on the conversation history.

        :param conversation: The accumulated conversation history.
        :return: The final concluded output.
        """
        self.agent.update_status('Reviewing output')
        return self.llm_wrapper.chat_completion(
            messages=[
                {"role": "system", "content": REACT_SYSTEM_PROMPT},
                {"role": "user", "content": conversation}
            ]
        )