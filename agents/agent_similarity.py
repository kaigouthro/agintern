import numpy as np
from typing import Any, List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
from integrations.llm_wrapper import LLMAPIWrapper
from microagent import MicroAgent


class Agent:
    def __init__(self, purpose: str):
        self.purpose = purpose


class AgentSimilarity:
    def __init__(self, llm_wrapper: LLMAPIWrapper, agents: List[Agent | MicroAgent]):
        """
        Initializes the AgentSimilarity object.

        :param llm_wrapper: Instance of LLMAPIWrapper to interact with OpenAI API.
        :param agents: List of Agent objects.
        """
        self.llm_wrapper = llm_wrapper
        self.agents = agents

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Retrieves the embedding for a given text.

        :param text: Text to get embedding for.
        :return: Embedding as a numpy array.
        """
        try:
            response: List[float] | Any = self.llm_wrapper.get_embedding(text)
            # if not a dict..
            if not isinstance(response, dict):
                return np.array(response)
            if "data" in response and len(response["data"]) > 0 and "embedding" in response["data"][0]:
                return np.array(response["data"][0]["embedding"])
            else:
                raise ValueError("Invalid response format")
        except Exception as e:
            raise ValueError(f"Error retrieving embedding: {e}") from e

    def calculate_similarity_threshold(self) -> float:
        """
        Calculates the 98th percentile of the similarity threshold across all agents.

        :return: 98th percentile of similarity threshold.
        """
        try:
            embeddings = [self.get_embedding(agent.purpose) for agent in self.agents]
            if len(embeddings) < 250:
                return 0.9

            similarities = [cosine_similarity([e1], [e2])[0][0] for i, e1 in enumerate(embeddings) for e2 in embeddings[i + 1 :]]
            return float(np.percentile(similarities, 98) if similarities else 0.9)
        except Exception as e:
            raise ValueError(f"Error calculating similarity threshold: {e}") from e

    def find_closest_agent(self, purpose_embedding: np.ndarray) -> Tuple[Optional[Agent | MicroAgent], float]:
        """
        Finds the closest agent based on the given purpose embedding.

        :param purpose_embedding: The embedding of the purpose to find the closest agent for.
        :return: Tuple of the closest agent and the highest similarity score.
        """
        closest_agent: Optional[Agent | MicroAgent] = None
        highest_similarity: float = -np.inf

        try:
            for agent in self.agents:
                agent_embedding = self.get_embedding(agent.purpose)
                similarity = cosine_similarity([agent_embedding], [purpose_embedding])[0][0]

                if similarity > highest_similarity:
                    highest_similarity = similarity
                    closest_agent = agent

            return closest_agent, highest_similarity
        except Exception as e:
            raise ValueError(f"Error finding closest agent: {e}") from e
