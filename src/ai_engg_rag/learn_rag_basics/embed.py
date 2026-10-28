import os
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics.pairwise import (cosine_similarity,euclidean_distances,linear_kernel)

load_dotenv()

client = OpenAI(api_key = os.getenv("OPENROUTER_API_KEY"),
                base_url = "https://openrouter.ai/api/v1")

# EMBEDDING_MODEL = "nvidia/nemotron-3-embed-1b:free"
EMBEDDING_MODEL = "openai/text-embedding-3-large"
def embed_documents(texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [d.embedding for d in resp.data]

def retrieve_cosine_similarity(query_emb, doc_embs, docs, top_k=10):
    scores = cosine_similarity(query_emb, doc_embs)[0]

    # pair each document with its score, sort highest first, take top_k
    scored = list(zip(docs, scores))
    # print(scored)
    scored.sort(key=lambda x: x[1], reverse=True)

    return [{"document": doc, "score": float(score)} for doc, score in scored[:top_k]]

def retrieve_euclidean_distance(query_emb, doc_embs, docs, top_k=10):
    scores = euclidean_distances(query_emb, doc_embs)[0]

    # pair each document with its score, sort highest first, take top_k
    scored = list(zip(docs, scores))
    # print(scored)
    scored.sort(key=lambda x: x[1])

    return [{"document": doc, "score": float(score)} for doc, score in scored[:top_k]]

def retrieve_dot_product(query_emb, doc_embs, docs, top_k=10):
    scores = linear_kernel(query_emb, doc_embs)[0]

    # pair each document with its score, sort highest first, take top_k
    scored = list(zip(docs, scores))
    # print(scored)
    scored.sort(key=lambda x: x[1], reverse=True)

    return [{"document": doc, "score": float(score)} for doc, score in scored[:top_k]]