from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

# Load environment variables

load_dotenv()

# Initialize Groq LLM

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),

    # Updated working model
    model_name="llama-3.3-70b-versatile",

    temperature=0.3
)

# Function to get AI response

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
