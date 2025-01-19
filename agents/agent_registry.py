from typing import Dict, Callable, Any
from agents.microagent import MicroAgent

class AgentRegistry:
    """
    Manages the registration and creation of different types of agents.
    """

    def __init__(self):
        """
        Initializes the agent registry.
        """
        self.agent_types: Dict[str, Callable[..., MicroAgent]] = {}

    def register_agent_type(self, agent_type: str, creator_func: Callable[..., MicroAgent]):
        """
        Registers a new agent type.

        :param agent_type: The type of the agent.
        :param creator_func: A function that creates an instance of the agent.
        """
        self.agent_types[agent_type] = creator_func

    def create_agent(self, agent_type: str, *args: Any, **kwargs: Any) -> MicroAgent:
        """
        Creates an agent of the specified type.

        :param agent_type: The type of the agent to create.
        :param args: Positional arguments for the agent's constructor.
        :param kwargs: Keyword arguments for the agent's constructor.
        :return: An instance of the specified agent type.
        :raises ValueError: If the agent type is not registered.
        """
        if agent_type not in self.agent_types:
            raise ValueError(f"Unknown agent type: {agent_type}")
        return self.agent_types[agent_type](*args, **kwargs)