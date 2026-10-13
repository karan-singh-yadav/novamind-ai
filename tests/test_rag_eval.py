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

judge = GroqJudge(model_name="llama-3.3-70b-versatile")
faithfulness_metric = FaithfulnessMetric(threshold=0.6, model=judge)
relevancy_metric = AnswerRelevancyMetric(threshold=0.6, model=judge)

def load_test_data():
    dataset_path = os.path.join(os.path.dirname(__file__), "eval_dataset.json")
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)

@pytest.mark.parametrize("item", load_test_data())
def test_rag_pipeline_quality(item):
    # Space out API requests to prevent Groq rate limits
    time.sleep(3)
    query = item["input"]
    expected_output = item["expected_output"]

    retrieved_chunks = retrieve_docs(query)
    actual_output = get_llm_response(query=query, context=retrieved_chunks)

    test_case = LLMTestCase(
        input=query,
        actual_output=str(actual_output),
        expected_output=expected_output,
        retrieval_context=retrieved_chunks
    )

    assert_test(test_case, [faithfulness_metric, relevancy_metric])