from qdrant_client import QdrantClient, models
from qdrant_client.models import HnswConfigDiff, SearchParams, OptimizersConfigDiff
# qdrant = QdrantClient(":memory:")

qdrant = QdrantClient(url="http://localhost:6333")

def create_collection(collection_name : str, emb_size:int , is_exact : bool = True, m : int = None, ef_construct : int = None):

    if qdrant.collection_exists(collection_name):
        qdrant.delete_collection(collection_name)

    if is_exact:
        qdrant.create_collection(
            collection_name=collection_name,

            vectors_config=models.VectorParams(
                size= emb_size ,   # always read from the model, never hard-code
                distance=models.Distance.COSINE,
            ),
            optimizers_config=OptimizersConfigDiff(indexing_threshold=1),
            quantization_config=models.ScalarQuantization(
                    scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8,    # 1 byte per number instead of 4
                    quantile=0.99,                  # ignore the most extreme 1% of values when picking the range
                    always_ram=True,                # keep the compressed copy in RAM
                    )
        ),
        )
    else:
        qdrant.create_collection(
            collection_name=collection_name,

            vectors_config=models.VectorParams(
                size= emb_size ,   # always read from the model, never hard-code
                distance=models.Distance.COSINE,
                hnsw_config=HnswConfigDiff(m= m, ef_construct= ef_construct,full_scan_threshold=10),
            ),
            optimizers_config=OptimizersConfigDiff(indexing_threshold=1),
        )

def upsert_documents(collection_name, documents, doc_embeddings):
    points = [
        models.PointStruct(
            id=doc["id"],
            vector=vec,
            payload={"text": doc["text"], "topic": doc["topic"], "level": doc["level"]},
        )
        for doc, vec in zip(documents, doc_embeddings)
    ]

    qdrant.upsert(collection_name=collection_name, points=points, wait=True)
    print(qdrant.count(collection_name))

def query_results(collection_name, query_embedding, ef=None, exact=False, limit=5):

    params = SearchParams(exact=True) if exact else SearchParams(hnsw_ef=ef,
                                                                 quantization = models.QuantizationSearchParams(
                                                                 ignore=False,      # use the quantized vectors (True would skip them)
                                                                 rescore=True,      # re-check the best candidates with the original float32 vectors
            ))
    
    res = qdrant.query_points(
        collection_name=collection_name,
        query= query_embedding,
        limit=limit,
        with_payload=True,
        query_filter=models.Filter(
        must=[
            models.FieldCondition(key="topic", match=models.MatchValue(value="backend")),
        ],
        ),
        search_params= params,    # re-check the best candidates with the original float32 vectors
        )
    return res.points


def retrieve_all(collection_name, ids):
    return qdrant.retrieve(collection_name,ids)