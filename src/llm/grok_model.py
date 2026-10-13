from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise EnvironmentError(
        "The GROQ_API_KEY environment variable is required. Add it to your .env file or environment."
    )

llm = ChatGroq(
    groq_api_key=groq_api_key, 
    model_name="qwen/qwen3.8-27b", 
    temperature=0.3
)

def get_llm_response(query, context):
    prompt = f"""
    You are a helpful AI assistant.

    Answer the user's question based ONLY on the provided context.

    Context:
    {context}

    Question:
    {query}
    """
    response = llm.invoke(prompt)
    return response.content