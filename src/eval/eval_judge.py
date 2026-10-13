import os
import re
import time
from deepeval.models.base_model import DeepEvalBaseLLM
from groq import Groq, RateLimitError
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")


class GroqJudge(DeepEvalBaseLLM):
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
        self.client = Groq(api_key=api_key)

    def load_model(self):
        return self.client

    def _clean_json_output(self, text: str) -> str:
        text = text or ""
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
                time.sleep(2)
                completion = self.client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an evaluation engine. You must output strictly valid JSON without explanations.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    model=self.model_name,
                    temperature=0.0,
                    max_completion_tokens=4096,
                    reasoning_effort="low",
                    response_format={"type": "json_object"},
                )
                return self._clean_json_output(completion.choices[0].message.content)

            except RateLimitError:
                if attempt < max_retries - 1:
                    time.sleep(backoff)
                    backoff += 4
                else:
                    raise

            except Exception as e:
                msg = str(e).lower()
                if ("response_format" in msg or "reasoning_effort" in msg) and attempt < max_retries - 1:
                    completion = self.client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model=self.model_name,
                        temperature=0.0,
                        max_completion_tokens=4096,
                    )
                    return self._clean_json_output(completion.choices[0].message.content)
                raise

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return self.model_name