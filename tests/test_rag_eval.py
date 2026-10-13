import os
import json
import time
import pytest
from dotenv import load_dotenv
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric

load_dotenv()

from src.eval.eval_judge import GroqJudge
from src.retrieval.retriever import retrieve_docs
from src.llm.grok_model import get_llm_response

judge = GroqJudge()  # uses GROQ_MODEL, default openai/gpt-oss-120b
faithfulness_metric = FaithfulnessMetric(threshold=0.6, model=judge)
relevancy_metric = AnswerRelevancyMetric(threshold=0.6, model=judge)


def load_test_data():
    dataset_path = os.path.join(os.path.dirname(__file__), "eval_dataset.json")
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.mark.parametrize("item", load_test_data())
def test_rag_pipeline_quality(item):
    time.sleep(2)
    query = item["input"]
    expected_output = item["expected_output"]

    retrieved_chunks = retrieve_docs(query)
    actual_output = get_llm_response(query=query, context=retrieved_chunks)

    # DeepEval needs retrieval_context as a list of strings
    context_strings = [getattr(c, "page_content", str(c)) for c in retrieved_chunks]

    test_case = LLMTestCase(
        input=query,
        actual_output=str(actual_output),
        expected_output=expected_output,
        retrieval_context=context_strings,
    )

    assert_test(test_case, [faithfulness_metric, relevancy_metric])