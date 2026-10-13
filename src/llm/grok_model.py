import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise EnvironmentError("The GROQ_API_KEY environment variable is required.")

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile",
    temperature=0.2,
    max_tokens=600
)

def get_llm_response(query: str, context: list) -> str:
    prompt = f"""
    You are a helpful AI assistant. Answer the question based strictly on the provided context.

    Context:
    {context}

    Question:
    {query}
    """
    response = llm.invoke(prompt)
    return response.content