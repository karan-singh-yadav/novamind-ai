import os
import time
from dotenv import load_dotenv
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric

load_dotenv()

from src.eval.eval_judge import GroqJudge
from src.retrieval.retriever import retrieve_docs
from src.llm.grok_model import get_llm_response

judge = GroqJudge(model_name="openai/gpt-oss-120b")
faithfulness = FaithfulnessMetric(threshold=0.6, model=judge)
relevancy = AnswerRelevancyMetric(threshold=0.6, model=judge)

queries = [
    "What is Supervised Learning?",
    "What is the primary difference between classification and regression?",
    "What is overfitting?"
]

def run_benchmark():
    print("\n" + "="*65)
    print("  NovaMind AI: Automated Retrieval Quality Benchmark")
    print("="*65 + "\n")

    results = []
    for query in queries:
        time.sleep(2)
        print(f"--> Evaluating: '{query}'")
        chunks = retrieve_docs(query)
        output = get_llm_response(query, chunks)

        tc = LLMTestCase(
            input=query,
            actual_output=str(output),
            retrieval_context=chunks
        )

        faithfulness.measure(tc)
        relevancy.measure(tc)

        results.append({
            "Query": query[:32] + "...",
            "Faithfulness": round(faithfulness.score, 2),
            "Relevancy": round(relevancy.score, 2)
        })

    print("\n" + "-"*65)
    print(f"{'Query':<38} | {'Faithfulness':<12} | {'Relevancy':<10}")
    print("-"*65)
    for r in results:
        print(f"{r['Query']:<38} | {r['Faithfulness']:<12} | {r['Relevancy']:<10}")
    print("-"*65 + "\n")

if __name__ == "__main__":
    run_benchmark()