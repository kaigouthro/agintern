import logging
from typing import List, Optional, Any
from agents.agent_creation import AgentCreation
from agents.agent_similarity import AgentSimilarity
from integrations.llm_wrapper import LLMAPIWrapper
from agents.agent_registry import AgentRegistry
from agents.task_decomposition import TaskDecompositionStrategy, SimpleDecomposition, AdvancedDecomposition
from agents.capabilities import CodeExecutionAbility, FileSystemAbility, DiffAbility, GitAbility, PythonFunctionAbility
from integrations.llm_wrapper import LLMAPIWrapper
from prompt_management.prompts import (
    PRIME_PROMPT, PRIME_NAME,
    PROMPT_ENGINEERING_SYSTEM_PROMPT,
    PROMPT_ENGINEERING_TEMPLATE, EXAMPLES
)
import json

class MicroAgentManager:
    """
    Manages the creation, retrieval, and interaction of micro agents.
    """

    def __init__(self, llm_wrapper: LLMAPIWrapper, max_agents: int = 20, agent_registry: Optional[AgentRegistry] = None, logger: Optional[logging.Logger] = None):
        """
        Initializes the MicroAgentManager.

        :param llm_wrapper: The LLM API wrapper.
        :param max_agents: The maximum number of agents allowed.
        :param agent_registry: The agent registry for managing agent types.
        :param logger: Logger instance for logging messages.
        """
        self.llm_wrapper = llm_wrapper
        self.max_agents = max_agents
        self.agent_creator = AgentCreation(self.llm_wrapper, max_agents, agent_registry)
        self.logger = logger or self._setup_logger()
        self.agent_registry = agent_registry or AgentRegistry()
        self.initialize_agent_registry()
        self.shared_knowledge = {}  # Initialize shared knowledge

    def _setup_logger(self) -> logging.Logger:
        """Sets up a logger for the class."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.ERROR)
        logger.addHandler(logging.StreamHandler())
        return logger

    def initialize_agent_registry(self):
        """
        Initializes the agent registry with predefined agent types.
        """
        # Define task decomposition strategies
        simple_decomposition = SimpleDecomposition()
        advanced_decomposition = AdvancedDecomposition()
        code_execution_ability = CodeExecutionAbility()
        file_system_ability = FileSystemAbility()
        diff_ability = DiffAbility()
        git_ability = GitAbility()
        python_function_ability = PythonFunctionAbility()

        # Register the prime agent type
        self.agent_registry.register_agent_type(
            PRIME_NAME,
            lambda initial_prompt, purpose, depth, agent_creator, llm_wrapper, max_depth, bootstrap_agent, is_prime, name: MicroAgent(
                initial_prompt, purpose, depth, agent_creator, llm_wrapper,
                [code_execution_ability, file_system_ability, diff_ability, git_ability],
                advanced_decomposition, max_depth, bootstrap_agent, is_prime, name
            )
        )

        # Register the general agent type
        self.agent_registry.register_agent_type(
            "General",
            lambda initial_prompt, purpose, depth, agent_creator, llm_wrapper, max_depth, bootstrap_agent, is_prime, name: MicroAgent(
                initial_prompt, purpose, depth, agent_creator, llm_wrapper,
                [code_execution_ability, python_function_ability],
                simple_decomposition, max_depth, bootstrap_agent, is_prime, name
            )
        )

    def get_agents(self) -> List[Any]:
        """Returns the list of agents."""
        return self.agent_creator.agents

    def create_agents(self) -> None:
        """Creates prime agents and logs the process."""
        self.logger.info("Creating agents...")
        try:
            self.agent_creator.create_prime_agent()
            self.logger.info("Agents created successfully.")
        except Exception as e:
            self.logger.error(f"Error in creating agents: {e}")
            raise

    def get_or_create_agent(self, purpose: str, depth: int, sample_input: str) -> Any:
        """
        Retrieves an existing agent or creates a new one based on the given purpose.
        """
        self.logger.info(f"Getting or creating agent for purpose: {purpose}")
        try:
            agent = self.agent_creator.get_or_create_agent(purpose, depth, sample_input)
            self.logger.info(f"Agent for purpose '{purpose}' retrieved or created.")
            return agent
        except Exception as e:
            self.logger.error(f"Error in getting or creating agent: {e}")
            raise

    def find_closest_agent(self, purpose: str) -> Any:
        """
        Finds the closest agent matching the given purpose.
        """
        self.logger.info(f"Finding closest agent for purpose: {purpose}")
        try:
            agent_similarity = AgentSimilarity(self.llm_wrapper, self.agent_creator.agents)
            purpose_embedding = agent_similarity.get_embedding(purpose)
            closest_agent = agent_similarity.find_closest_agent(purpose_embedding)
            self.logger.info(f"Closest agent for purpose '{purpose}' found.")
            return closest_agent
        except Exception as e:
            self.logger.error(f"Error in finding closest agent: {e}")
            raise

    def display_agent_status(self):
        """Displays the current status of all agents."""
        for agent in self.get_agents():
            self.logger.info(f"Agent {agent.name}: Status = {agent.current_status}, Evolve Count = {agent.evolve_count}")

    def display_active_agent_tree(self):
        """Displays a tree view of active agent relationships."""
        for agent in self.get_agents():
            if agent.active_agents:
                self.logger.info(f"Agent {agent.name} is calling: {agent.active_agents}")
            else:
                self.logger.info(f"Agent {agent.name} is currently idle.")

    def save_agent_state(self, agent: MicroAgent, filepath: str):
        """
        Saves the state of an agent to a file.

        :param agent: The agent whose state needs to be saved.
        :param filepath: The path to the file where the state will be saved.
        """
        agent_state = {
            'dynamic_prompt': agent.dynamic_prompt,
            'purpose': agent.purpose,
            'depth': agent.depth,
            'max_depth': agent.max_depth,
            'usage_count': agent.usage_count,
            'working_agent': agent.working_agent,
            'evolve_count': agent.evolve_count,
            'number_of_code_executions': agent.number_of_code_executions,
            'current_status': agent.current_status,
            'active_agents': agent.active_agents,
            'last_input': agent.last_input,
            'is_prime': agent.is_prime,
            'name': agent.name
        }

        try:
            with open(filepath, 'w') as f:
                json.dump(agent_state, f)
            self.logger.info(f"Agent '{agent.name}' state saved to '{filepath}'.")
        except Exception as e:
            self.logger.error(f"Error saving agent state for '{agent.name}': {e}")

    def load_agent_state(self, filepath: str) -> MicroAgent:
        """
        Loads the state of an agent from a file.

        :param filepath: The path to the file from which the state will be loaded.
        :return: A MicroAgent instance with the loaded state.
        """
        try:
            with open(filepath, 'r') as f:
                agent_state = json.load(f)
            self.logger.info(f"Agent state loaded from '{filepath}'.")

            # Create a new agent using the loaded state
            loaded_agent = self.agent_registry.create_agent(
                "General" if not agent_state['is_prime'] else PRIME_NAME,
                agent_state['dynamic_prompt'],
                agent_state['purpose'],
                agent_state['depth'],
                self,
                self.llm_wrapper,
                agent_state['max_depth'],
                agent_state['working_agent'],
                agent_state['is_prime'],
                agent_state['name']
            )

            # Update the agent's attributes
            loaded_agent.usage_count = agent_state['usage_count']
            loaded_agent.evolve_count = agent_state['evolve_count']
            loaded_agent.number_of_code_executions = agent_state['number_of_code_executions']
            loaded_agent.current_status = agent_state['current_status']
            loaded_agent.active_agents = agent_state['active_agents']
            loaded_agent.last_input = agent_state['last_input']

            return loaded_agent
        except Exception as e:
            self.logger.error(f"Error loading agent state from '{filepath}': {e}")
            raise

    def update_shared_knowledge(self, key: str, value: Any):
        """
        Updates the shared knowledge base.

        :param key: The key for the knowledge entry.
        :param value: The value of the knowledge entry.
        """
        self.shared_knowledge[key] = value

    def get_shared_knowledge(self, key: str) -> Any:
        """
        Retrieves a value from the shared knowledge base.

        :param key: The key for the knowledge entry.
        :return: The value associated with the key, or None if the key is not found.
        """
        return self.shared_knowledge.get(key)