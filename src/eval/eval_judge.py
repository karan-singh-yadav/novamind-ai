import os
import re
import time
from deepeval.models.base_model import DeepEvalBaseLLM
from groq import Groq, RateLimitError
from dotenv import load_dotenv

load_dotenv()

class GroqJudge(DeepEvalBaseLLM):
    def __init__(self, model_name: str = "llama-3.1-8b-instant"):
        self.model_name = model_name
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
        self.client = Groq(api_key=api_key)

    def load_model(self):
        return self.client

    def _clean_json_output(self, text: str) -> str:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        match_obj = re.search(r"(\{.*\})", text, re.DOTALL)
        if match_obj:
            return match_obj.group(1).strip()
        return text.strip()

    def generate(self, prompt: str) -> str:
        max_retries = 4
        backoff = 3

        for attempt in range(max_retries):
            try:
                time.sleep(1.5)
                completion = self.client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an evaluation engine. You must output strictly valid JSON without explanations."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    model=self.model_name,
                    temperature=0.0,
                    max_tokens=600,
                    response_format={"type": "json_object"}
                )
                raw_response = completion.choices[0].message.content
                return self._clean_json_output(raw_response)
            except RateLimitError:
                if attempt < max_retries - 1:
                    time.sleep(backoff)
                    backoff += 4
                else:
                    raise
            except Exception as e:
                if "response_format" in str(e).lower() and attempt < max_retries - 1:
                    completion = self.client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model=self.model_name,
                        temperature=0.0,
                        max_tokens=600
                    )
                    return self._clean_json_output(completion.choices[0].message.content)
                raise e

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return self.model_name