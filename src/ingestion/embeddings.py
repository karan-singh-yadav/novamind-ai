from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

def embedding_function(texts):

    embeddings = model.encode(texts)

    return embeddings.tolist()