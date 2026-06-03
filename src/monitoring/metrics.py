from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

ingest_counter = Counter("queryforge_ingestions_total", "Total documents ingested")
query_counter = Counter("queryforge_queries_total", "Total queries processed")
query_latency = Histogram("queryforge_query_seconds", "Query latency in seconds")
faithfulness_histogram = Histogram(
    "queryforge_faithfulness_score",
    "Faithfulness scores",
    buckets=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
)
