import logging
from integrations.llm_wrapper import LLMAPIWrapper

# Basic logging setup
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

class AgentEvaluator:
    """
    Evaluates AI agent's responses using OpenAI's GPT model.
    """

    def __init__(self, llm_wrapper: LLMAPIWrapper):
        self.llm_api = llm_wrapper

    def evaluate(self, input_text: str, prompt: str, output: str) -> str:
        """
        Returns evaluation of agent's output as 'Poor', 'Good', or 'Perfect'.
        """
        try:
            query = ("Evaluate LLM Output: '{input}' with prompt '{prompt}' "
                     "for quality/relevance. Possible Answers: Poor, Good, Perfect. "
                     "LLM output: '{output}'").format(input=input_text, prompt=prompt, output=output)

            return self.llm_api.chat_completion(messages=[{"role": "system", "content": query}])
        except Exception as error:
            logging.info(f"Agent evaluation error: {error}")
            raise