from embed import embed_documents
from documents import docs
from qdrant import create_collection, query_results, upsert_documents
import time

COLLECTION_EXACT = "exact" 
COLLECTION_HNSW_M8 = "hnsw_m8" 
COLLECTION_HNSW_M32 = "hnsw32"

documents = docs

doc_embeddings = embed_documents([document["text"] for document in documents])
emb_size = len(doc_embeddings[0])

create_collection(COLLECTION_EXACT,emb_size=emb_size )
create_collection(COLLECTION_HNSW_M8,emb_size=emb_size, is_exact=False, m = 8, ef_construct = 100)
create_collection(COLLECTION_HNSW_M32,emb_size=emb_size, is_exact=False, m = 32, ef_construct = 100)

upsert_documents(COLLECTION_EXACT, documents , doc_embeddings)
upsert_documents(COLLECTION_HNSW_M8, documents , doc_embeddings)
upsert_documents(COLLECTION_HNSW_M32, documents , doc_embeddings)

query = "how do I scale an AI application with caching and microservices?"
query_embedding = embed_documents([query])

starttime = time.perf_counter()

exact_results = query_results(COLLECTION_EXACT , query_embedding[0], exact=True )

endtime = time.perf_counter()

print("*"*100)
print("Exact Query Results latency " , endtime - starttime)

for result in exact_results:
    print(result.payload["text"])
starttime = time.perf_counter()
hnsw8_ef8_results= query_results(COLLECTION_HNSW_M8,query_embedding[0],ef = 8 )

print("*"*100)
print("hnsw8_ef8_results latency : ", time.perf_counter() - starttime)

for result in hnsw8_ef8_results:
    print(result.payload["text"])

starttime = time.perf_counter()
hnsw8_ef32_results= query_results(COLLECTION_HNSW_M8,query_embedding[0],ef = 32 )

print("*"*100)
print("hnsw8_ef32_results latency : ", time.perf_counter() - starttime)

for result in hnsw8_ef32_results:
    print(result.payload["text"])
starttime = time.perf_counter()
hnsw8_ef128_results= query_results(COLLECTION_HNSW_M8,query_embedding[0],ef = 128 )

print("*"*100)
print("hnsw8_ef128_results latency : ", time.perf_counter() - starttime)

for result in hnsw8_ef128_results:
    print(result.payload["text"])
starttime = time.perf_counter()
hnsw32_ef8_results= query_results(COLLECTION_HNSW_M32,query_embedding[0],ef = 8 )

print("*"*100)
print("hnsw32_ef8_results latency : ", time.perf_counter() - starttime)

for result in hnsw32_ef8_results:
    print(result.payload["text"])
starttime = time.perf_counter()
hnsw32_ef32_results= query_results(COLLECTION_HNSW_M32,query_embedding[0],ef = 32 )

print("*"*100)
print("hnsw32_ef32_results latency : ", time.perf_counter() - starttime)

for result in hnsw32_ef32_results:
    print(result.payload["text"])
starttime = time.perf_counter()
hnsw32_ef128_results= query_results(COLLECTION_HNSW_M32,query_embedding[0],ef = 128 )
print("*"*100)
print("hnsw32_ef128_results latency : ", time.perf_counter() - starttime)
for result in hnsw32_ef128_results:
    print(result.payload["text"])

from qdrant import qdrant
for n in ["hnsw_m8", "hnsw32"]:
    print(n, qdrant.get_collection(n).indexed_vectors_count)


import statistics

queries = [
    "how do I scale an AI application with caching and microservices?",
    "What does a vector database do?",
    "How does retrieval augmented generation work?",
    "Explain approximate nearest neighbor search using graphs",
    "How can I reduce the memory used by embeddings?",
    "Filtering vector search results by metadata",
    "Container orchestration with self-healing",
    "Event streaming for asynchronous systems",
    "What is cosine similarity used for?",
    "Monitoring distributed applications with logs and traces",
]
qvecs = embed_documents(queries)

# ground truth: true top 50 per query (top 5 is its first 5)
truth = [[r.id for r in query_results(COLLECTION_EXACT, q, exact=True, limit=50)] for q in qvecs]

def timed(coll, q, ef, limit, repeats=5):
    query_results(coll, q, ef=ef, limit=limit)            # warm-up, not timed
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        res = query_results(coll, q, ef=ef, limit=limit)
        times.append((time.perf_counter() - t0) * 1000)
    return [r.id for r in res], statistics.mean(times)

print(f"{'setup':<12}{'recall@5':>10}{'recall@50':>11}{'latency ms':>12}")
for coll, label in [(COLLECTION_HNSW_M8, "m8"), (COLLECTION_HNSW_M32, "m32")]:
    for ef in [8, 16, 32, 64, 128]:
        r5, r50, lats = [], [], []
        for q, t in zip(qvecs, truth):
            ids5, lat = timed(coll, q, ef, 5)
            ids50, _ = timed(coll, q, ef, 50, repeats=1)
            r5.append(len(set(ids5) & set(t[:5])) / 5)
            r50.append(len(set(ids50) & set(t)) / 50)
            lats.append(lat)
        print(f"{label}_ef{ef:<7}{statistics.mean(r5):>10.2f}{statistics.mean(r50):>11.2f}{statistics.mean(lats):>12.1f}")