import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise EnvironmentError("The GROQ_API_KEY environment variable is required.")

MODEL_NAME = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name=MODEL_NAME,
    temperature=0.2,
    max_tokens=2048,  # gpt-oss spends tokens on reasoning, so 600 is too low
    model_kwargs={"reasoning_effort": "low"},
)


def _to_text(chunk) -> str:
    """Accepts plain strings or LangChain Document objects."""
    return getattr(chunk, "page_content", str(chunk))


def get_llm_response(query: str, context: list) -> str:
    context_text = "\n\n".join(_to_text(c) for c in context)

    prompt = f"""You are a helpful AI assistant. Answer the question based strictly on the provided context.
If the context does not contain the answer, say you don't have enough information.

Context:
{context_text}

Question:
{query}
"""
    response = llm.invoke(prompt)
    return response.content