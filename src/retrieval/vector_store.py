import chromadb
from src.ingestion.embeddings import embedding_function

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_or_create_collection(
    name="rag_collection"
)

def store_embeddings(chunks):

    embeddings = embedding_function(chunks)

    ids = [str(i) for i in range(len(chunks))]

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids
    )