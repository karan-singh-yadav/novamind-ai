import chromadb
import uuid
from src.ingestion.embeddings import embedding_function

client = chromadb.PersistentClient(path="./chroma_db")


def store_embeddings(chunks):

    client.delete_collection(name="rag_collection")
    collection = client.get_or_create_collection(name="rag_collection")

    embeddings = embedding_function(chunks)
    ids = [str(uuid.uuid4()) for _ in chunks]

    collection.add(documents=chunks, embeddings=embeddings, ids=ids)
