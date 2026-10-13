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
from src.ingestion.pdf_loader import load_pdf
from src.ingestion.text_splitter import split_text
from src.retrieval.vector_store import store_embeddings
from src.retrieval.retriever import retrieve_docs
from src.llm.grok_model import get_llm_response

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_PATH = os.path.join(ROOT, "machine_learning_tutorial.pdf")

judge = GroqJudge()
faithfulness_metric = FaithfulnessMetric(threshold=0.6, model=judge)
relevancy_metric = AnswerRelevancyMetric(threshold=0.6, model=judge)


@pytest.fixture(scope="session", autouse=True)
def ingest_pdf():
    """CI has no chroma_db (it is gitignored), so build it from the PDF first."""
    chunks = split_text(load_pdf(PDF_PATH))
    assert chunks, f"No text extracted from {PDF_PATH}"
    try:
        store_embeddings(chunks)
    except Exception:
        # store_embeddings deletes the collection first, which errors
        # when it does not exist yet (fresh CI machine). Retry creates it.
        import chromadb
        from src.ingestion.embeddings import embedding_function
        import uuid

        client = chromadb.PersistentClient(path="./chroma_db")
        collection = client.get_or_create_collection(name="rag_collection")
        collection.add(
            documents=chunks,
            embeddings=embedding_function(chunks),
            ids=[str(uuid.uuid4()) for _ in chunks],
        )


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
    assert len(retrieved_chunks) > 0, "Retriever returned nothing"

    actual_output = get_llm_response(query=query, context=retrieved_chunks)

    context_strings = [getattr(c, "page_content", str(c)) for c in retrieved_chunks]

    test_case = LLMTestCase(
        input=query,
        actual_output=str(actual_output),
        expected_output=expected_output,
        retrieval_context=context_strings,
    )

    assert_test(test_case, [faithfulness_metric, relevancy_metric])