"""Exact vs HNSW: Recall@K and latency over 10 queries, m in {8, 32}, hnsw_ef in {8..128}."""
import os
import time
import statistics

import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, HnswConfigDiff,
    OptimizersConfigDiff, SearchParams, CollectionStatus,
)

from embed import embed_documents      # your existing helper
from documents import docs             # your 200 points: {"id", "text", "topic", "level", ...}

# ─── Settings ───────────────────────────────────────────────────────────
COLLECTION_EXACT = "exact"
COLLECTION_M8 = "hnsw_m8"
COLLECTION_M32 = "hnsw32"
M_VALUES = {8: COLLECTION_M8, 32: COLLECTION_M32}
EF_VALUES = [8, 16, 32, 64, 128]
K_LIST = [5, 50]       # Recall@5 is the homework metric; Recall@50 makes the test harder
REPEATS = 5            # timed runs per query (after one warm-up)

# Ten varied queries so the true top results are not one cluster of near-duplicates
QUERIES = [
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

client = QdrantClient(
    url="http://localhost:6333",
    check_compatibility=False,         # silences the client/server version warning
)


# ─── Collections ────────────────────────────────────────────────────────
def build_collection(name, size, vectors, m=None, ef_construct=100):
    if client.collection_exists(name):
        client.delete_collection(name)

    if m is None:                       # plain collection, used only for exact search
        client.create_collection(name, vectors_config=VectorParams(size=size, distance=Distance.COSINE))
    else:
        # Tiny thresholds on purpose: with 200 points Qdrant would otherwise skip HNSW
        # and do a plain scan, which makes recall 1.0 for the wrong reason.
        client.create_collection(
            name,
            vectors_config=VectorParams(
                size=size, distance=Distance.COSINE,
                hnsw_config=HnswConfigDiff(m=m, ef_construct=ef_construct, full_scan_threshold=1),
            ),
            optimizers_config=OptimizersConfigDiff(indexing_threshold=1),
        )

    points = [
        PointStruct(id=d["id"], vector=v,
                    payload={k: val for k, val in d.items() if k != "id"})
        for d, v in zip(docs, vectors)
    ]
    for i in range(0, len(points), 50):
        client.upsert(collection_name=name, points=points[i:i + 50])


def wait_until_indexed(name, expected, timeout=90):
    """Indexing is asynchronous: searching too early would not measure HNSW."""
    start = time.time()
    while time.time() - start < timeout:
        info = client.get_collection(name)
        indexed = info.indexed_vectors_count or 0
        if info.status == CollectionStatus.GREEN and indexed >= expected:
            print(f"  {name}: indexed {indexed}/{expected} vectors")
            return True
        time.sleep(1)
    print(f"  WARNING {name}: indexing not confirmed "
          f"(indexed_vectors_count={client.get_collection(name).indexed_vectors_count}). "
          "Results may be plain scans, not HNSW.")
    return False


# ─── Search ─────────────────────────────────────────────────────────────
def search_ids(collection, qvec, limit, ef=None, exact=False):
    # exact=True skips the index, so hnsw_ef only applies to the HNSW path
    params = SearchParams(exact=True) if exact else SearchParams(hnsw_ef=ef)
    res = client.query_points(collection_name=collection, query=qvec, limit=limit,
                              with_payload=False, search_params=params)
    return [p.id for p in res.points]


def timed_search(collection, qvec, limit, ef=None, exact=False):
    """One untimed warm-up, then REPEATS timed runs. Returns (ids, mean latency in ms)."""
    search_ids(collection, qvec, limit, ef=ef, exact=exact)
    times, ids = [], None
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        ids = search_ids(collection, qvec, limit, ef=ef, exact=exact)
        times.append((time.perf_counter() - t0) * 1000)
    return ids, statistics.mean(times)


# ─── Main ───────────────────────────────────────────────────────────────
def main():
    print(f"{len(docs)} points, {len({d['id'] for d in docs})} unique ids")
    vectors = embed_documents([d["text"] for d in docs])
    size = len(vectors[0])
    qvecs = embed_documents(QUERIES)

    build_collection(COLLECTION_EXACT, size, vectors)
    for m, name in M_VALUES.items():
        print(f"Building {name} (m={m}, ef_construct=100) ...")
        build_collection(name, size, vectors, m=m)
        wait_until_indexed(name, expected=len(docs))

    # Ground truth from true exact search, for the largest K, sliced for smaller K
    max_k = max(K_LIST)
    truth = [search_ids(COLLECTION_EXACT, q, max_k, exact=True) for q in qvecs]

    rows = []

    # Exact row (its latency is the baseline cost of scanning everything)
    exact_lats = [timed_search(COLLECTION_EXACT, q, 5, exact=True)[1] for q in qvecs]
    rows.append({"strategy": "exact", "m": "-", "hnsw_ef": "-",
                 **{f"Recall@{k}": 1.0 for k in K_LIST},
                 "avg_latency_ms": round(statistics.mean(exact_lats), 2)})

    for m, name in M_VALUES.items():
        for ef in EF_VALUES:
            recalls = {k: [] for k in K_LIST}
            lats = []
            for q, true_ids in zip(qvecs, truth):
                for k in K_LIST:
                    ids, lat = timed_search(name, q, k, ef=ef)
                    recalls[k].append(len(set(ids) & set(true_ids[:k])) / k)
                    if k == 5:
                        lats.append(lat)            # latency reported for the top-5 query
            rows.append({"strategy": "hnsw", "m": m, "hnsw_ef": ef,
                         **{f"Recall@{k}": round(statistics.mean(recalls[k]), 3) for k in K_LIST},
                         "avg_latency_ms": round(statistics.mean(lats), 2)})

    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False))
    df.to_csv("hw4_results.csv", index=False)

    try:
        import matplotlib.pyplot as plt
        hnsw = df[df["strategy"] == "hnsw"]
        for m, g in hnsw.groupby("m"):
            plt.plot(g["avg_latency_ms"], g["Recall@5"], marker="o", label=f"m={m}")
            for _, r in g.iterrows():
                plt.annotate(int(r["hnsw_ef"]), (r["avg_latency_ms"], r["Recall@5"]))
        plt.xlabel("avg latency (ms)"); plt.ylabel("Recall@5"); plt.legend()
        plt.savefig("hw4_recall_vs_latency.png", dpi=150)
    except ImportError:
        pass


if __name__ == "__main__":
    main()