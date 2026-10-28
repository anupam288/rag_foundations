from qdrant import create_collection, upsert_documents, query_results, retrieve_all
from embed import embed_documents,retrieve_cosine_similarity,retrieve_dot_product,retrieve_euclidean_distance

COLLECTION = "rag_fundamentals"

documents = [
    {"id": 1,  "text": "Spring Boot simplifies building production-ready Java backend applications.",                                               "topic": "backend",           "level": "beginner"},
    {"id": 2,  "text": "Kubernetes orchestrates containers and provides scheduling, service discovery, and self-healing.",                          "topic": "devops",            "level": "intermediate"},
    {"id": 3,  "text": "Redis is an in-memory data store commonly used for caching, sessions, and fast lookups.",                                  "topic": "databases",         "level": "beginner"},
    {"id": 4,  "text": "Kafka is a distributed event streaming platform used to build high-throughput asynchronous systems.",                      "topic": "distributed-systems","level": "intermediate"},
    {"id": 5,  "text": "PostgreSQL is a relational database with strong SQL support, transactions, indexing, and extensibility.",                  "topic": "databases",         "level": "beginner"},
    {"id": 6,  "text": "Vector databases store embeddings and make similarity search over high-dimensional vectors efficient.",                    "topic": "ai",                "level": "beginner"},
    {"id": 7,  "text": "RAG retrieves relevant external context and gives that context to a language model before generating an answer.",          "topic": "ai",                "level": "beginner"},
    {"id": 8,  "text": "An embedding model converts text into a dense numerical vector that captures semantic relationships.",                    "topic": "ai",                "level": "beginner"},
    {"id": 9,  "text": "Cosine similarity compares the angle between vectors and is commonly used for semantic text retrieval.",                  "topic": "ai",                "level": "intermediate"},
    {"id": 10, "text": "HNSW builds a navigable graph so approximate nearest-neighbor search can avoid scanning every vector.",                   "topic": "vector-search",     "level": "advanced"},
    {"id": 11, "text": "Exact nearest-neighbor search compares the query against every candidate vector to obtain the true top results.",         "topic": "vector-search",     "level": "intermediate"},
    {"id": 12, "text": "Quantization compresses vector representations so a vector database can reduce memory usage and improve search efficiency.","topic": "vector-search",     "level": "advanced"},
    {"id": 13, "text": "Payload indexes allow a vector database to combine semantic similarity with structured filters.",                         "topic": "vector-search",     "level": "advanced"},
    {"id": 14, "text": "Python is widely used for machine learning, data processing, and building AI applications.",                              "topic": "programming",       "level": "beginner"},
    {"id": 15, "text": "Async programming can improve throughput for applications that spend significant time waiting on network I/O.",           "topic": "programming",       "level": "intermediate"},
    {"id": 16, "text": "API gateways can provide routing, authentication, rate limiting, and observability for microservices.",                   "topic": "distributed-systems","level": "intermediate"},
    {"id": 17, "text": "Microservices split an application into independently deployable services that communicate over APIs or events.",         "topic": "distributed-systems","level": "intermediate"},
    {"id": 18, "text": "Observability combines logs, metrics, and traces to understand the behavior of distributed applications.",               "topic": "devops",            "level": "intermediate"},
    {"id": 19, "text": "A cache can reduce database load and latency when frequently requested data can be reused safely.",                       "topic": "databases",         "level": "beginner"},
    {"id": 20, "text": "Reranking can improve retrieval quality by applying a more expensive relevance model to an initial candidate set.",       "topic": "ai",                "level": "advanced"},
]

doc_embeddings = embed_documents([document["text"] for document in documents])

print("Number of embeddings:", len(doc_embeddings))
print("Dimensions per vector:", len(doc_embeddings[0]))
print("Total storage for 5 million docs :" , (len(doc_embeddings) * len(doc_embeddings[0]) * 5000000) / (1024* 1024*1024) , "Gigabytes")

query1 = "Which programming language is commonly used for AI applications?"
query_embedding1 = embed_documents([query1])

query2 = "What does Quantization do?"
query_embedding2 = embed_documents([query2])

query3 = "How does caching help ?"
query_embedding3 = embed_documents([query3])

create_collection(COLLECTION, len(doc_embeddings[0]))
upsert_documents(COLLECTION, documents, doc_embeddings)

print("************************")
print("Cosine Similarity")

for document in retrieve_cosine_similarity(query_embedding1,doc_embeddings,documents):
    print(document["document"]["text"])

results1 = query_results(COLLECTION, query_embedding1[0])

print("************************")
print("Qdrant Collection Results")

for result in results1:
    print(result.payload["text"])

print("************************")
print("Euclidean Distance")

for document in retrieve_euclidean_distance(query_embedding2,doc_embeddings,documents):
    print(document["document"]["text"])

results2 = query_results(COLLECTION, query_embedding2[0])

print("************************")
print("Qdrant Collection Results")

for result in results2:
    print(result.payload["text"])

print("************************")
print("Dot product")

for document in retrieve_dot_product(query_embedding3,doc_embeddings,documents):
    print(document["document"]["text"])

results3 = query_results(COLLECTION, query_embedding3[0])

print("************************")
print("Qdrant Collection Results")

for result in results3:
    print(result.payload["text"])

updated_documents = [
    {"id": 20, 
     "text": "a more expensive relevance model to an initial candidate set to improve retrieval quality is called re-ranking", 
     "topic": "ai", 
     "level": "advanced"},
]
updated_doc_embeddings = embed_documents([document["text"] for document in updated_documents])
upsert_documents(COLLECTION, updated_documents, updated_doc_embeddings )

ids=[1,2,3,4,20]

print(retrieve_all(COLLECTION,ids))