import chromadb
from src.ingestion.embeddings import embedding_function

client = chromadb.PersistentClient(path="./chroma_db")


def retrieve_docs(query):

    collection = client.get_or_create_collection(name="rag_collection")
    query_embedding = embedding_function([query])[0]

    results = collection.query(query_embeddings=[query_embedding], n_results=3)

    documents = results.get("documents", [[]])[0] if results else []
    return documents or []
