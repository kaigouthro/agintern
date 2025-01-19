from typing import List, Optional

from agents.microagent import MicroAgent
from agents.agent_similarity import AgentSimilarity
from integrations.llm_wrapper import LLMAPIWrapper
from prompt_management.prompts import (
    PRIME_PROMPT, PRIME_NAME,
    PROMPT_ENGINEERING_SYSTEM_PROMPT,
    PROMPT_ENGINEERING_TEMPLATE, EXAMPLES
)
from agents.agent_registry import AgentRegistry

DEFAULT_MAX_AGENTS = 20
PRIME_AGENT_WEIGHT = 25

class AgentCreation:
    def __init__(self, llm_wrapper: LLMAPIWrapper, max_agents: int = DEFAULT_MAX_AGENTS, agent_registry: Optional[AgentRegistry] = None):
        self.agents: List[MicroAgent] = []
        self.llm_wrapper = llm_wrapper
        self.max_agents = max_agents
        self.agent_registry = agent_registry or AgentRegistry()

    def create_prime_agent(self) -> None:
        """
        Creates the prime agent and adds it to the agent list.
        """
        prime_agent = self.agent_registry.create_agent(
            PRIME_NAME,
            PRIME_PROMPT,
            PRIME_NAME,
            0,
            self,
            self.llm_wrapper,
            PRIME_AGENT_WEIGHT,
            True,
            True,
            PRIME_NAME
        )
        self.agents.append(prime_agent)

    def get_or_create_agent(self, purpose: str, depth: int, sample_input: str) -> MicroAgent:
        """
        Retrieves or creates an agent based on the given purpose.
        """
        agent_similarity = AgentSimilarity(self.llm_wrapper, self.agents)
        purpose_embedding = agent_similarity.get_embedding(purpose)
        closest_agent, highest_similarity = agent_similarity.find_closest_agent(purpose_embedding)
        similarity_threshold = agent_similarity.calculate_similarity_threshold()

        if highest_similarity >= similarity_threshold:
            closest_agent.usage_count += 1
            return closest_agent

        self.remove_least_used_agent_if_needed()
        new_agent = self.create_new_agent(purpose, depth, sample_input)
        return new_agent

    def remove_least_used_agent_if_needed(self) -> None:
        """
        Removes the least used agent if the maximum number of agents is exceeded.
        """
        if len(self.agents) >= self.max_agents:
            self.agents.sort(key=lambda agent: agent.usage_count)
            self.agents.pop(0)

    def create_new_agent(self, purpose: str, depth: int, sample_input: str) -> MicroAgent:
        """
        Creates a new agent.
        """
        prompt = self.generate_llm_prompt(purpose, sample_input)
        new_agent = self.agent_registry.create_agent(
            "General",  # Use the "General" agent type for new agents
            prompt,
            purpose,
            depth,
            self,
            self.llm_wrapper,
            3,  # Default max_depth
            False,  # Not a bootstrap agent
            False,  # Not a prime agent
            purpose   # Use purpose as the name
        )
        new_agent.usage_count = 1
        self.agents.append(new_agent)
        return new_agent

    def generate_llm_prompt(self, goal: str, sample_input: str) -> str:
        """
        Generates a prompt for the LLM based on the given goal and sample input.
        """
        messages = [
            {"role": "system", "content": PROMPT_ENGINEERING_SYSTEM_PROMPT},
            {"role": "user", "content": PROMPT_ENGINEERING_TEMPLATE.format(goal=goal, sample_input=sample_input, examples=EXAMPLES)}
        ]

        try:
            return self.llm_wrapper.chat_completion(messages=messages)
        except Exception as e:
            print(f"Error generating LLM prompt: {e}")
            return ""