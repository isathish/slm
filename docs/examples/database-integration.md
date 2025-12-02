---
layout: default
title: Database Integration
parent: Examples
nav_order: 2
---

# Database Integration Examples
{: .no_toc }

Load training data from various databases including PostgreSQL, MySQL, SQLite, MongoDB, Redis, and Elasticsearch.
{: .fs-6 .fw-300 }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## PostgreSQL

Load data from a PostgreSQL database.

### Prerequisites

```bash
pip install slm-builder[db]
# or
pip install sqlalchemy psycopg2-binary
```

### Basic Example

```python
from slm_builder import SLMBuilder

builder = SLMBuilder(
    project_name="postgres-qa",
    base_model="google/flan-t5-small"
)

# Load from PostgreSQL
builder.load_from_database(
    connection_params={
        "dialect": "postgresql",
        "host": "localhost",
        "port": 5432,
        "database": "qa_database",
        "user": "your_user",
        "password": "your_password"
    },
    query="SELECT question, answer FROM qa_pairs WHERE active=true"
)

# Train and evaluate
builder.train(epochs=3, method="lora")
metrics = builder.evaluate()
```

### Advanced Query

```python
# Complex query with joins
query = """
    SELECT 
        q.question_text as question,
        a.answer_text as answer,
        q.category,
        q.difficulty
    FROM questions q
    JOIN answers a ON q.id = a.question_id
    WHERE q.verified = true 
    AND a.upvotes > 5
    ORDER BY q.created_at DESC
    LIMIT 10000
"""

builder.load_from_database(
    connection_params={...},
    query=query
)
```

---

## MySQL

Load data from MySQL database.

### Prerequisites

```bash
pip install sqlalchemy pymysql
```

### Example

```python
from slm_builder import SLMBuilder

builder = SLMBuilder(
    project_name="mysql-qa",
    base_model="google/flan-t5-small"
)

# Load from MySQL
builder.load_from_database(
    connection_params={
        "dialect": "mysql",
        "host": "localhost",
        "port": 3306,
        "database": "customer_support",
        "user": "root",
        "password": "password"
    },
    query="SELECT ticket_question, ticket_answer FROM support_tickets"
)

builder.train(epochs=3, method="lora")
```

---

## SQLite

Load data from a local SQLite database.

### Example

```python
from slm_builder import SLMBuilder

builder = SLMBuilder(
    project_name="sqlite-qa",
    base_model="google/flan-t5-small"
)

# Load from SQLite (no host/port needed)
builder.load_from_database(
    connection_params={
        "dialect": "sqlite",
        "database": "./my_data.db"
    },
    query="SELECT question, answer FROM qa_table"
)

builder.train(epochs=3, method="lora")
```

---

## MongoDB

Load data from MongoDB collections.

### Prerequisites

```bash
pip install pymongo
```

### Basic Example

```python
from slm_builder import SLMBuilder

builder = SLMBuilder(
    project_name="mongo-qa",
    base_model="google/flan-t5-small"
)

# Load from MongoDB
builder.load_from_mongodb(
    connection_params={
        "host": "localhost",
        "port": 27017,
        "database": "qa_database",
        "collection": "qa_pairs",
        "user": "admin",
        "password": "password"
    },
    query={"status": "approved"},
    projection={"question": 1, "answer": 1, "_id": 0}
)

builder.train(epochs=3, method="lora")
```

### Advanced Query

```python
# Complex MongoDB query
query = {
    "$and": [
        {"verified": True},
        {"upvotes": {"$gte": 10}},
        {"category": {"$in": ["technology", "programming"]}},
        {"created_at": {"$gte": "2024-01-01"}}
    ]
}

projection = {
    "question_text": 1,
    "answer_text": 1,
    "metadata.category": 1,
    "_id": 0
}

builder.load_from_mongodb(
    connection_params={...},
    query=query,
    projection=projection,
    limit=5000
)
```

### Aggregation Pipeline

```python
# Use aggregation for complex transformations
pipeline = [
    {"$match": {"verified": True}},
    {"$lookup": {
        "from": "answers",
        "localField": "answer_id",
        "foreignField": "_id",
        "as": "answer_data"
    }},
    {"$unwind": "$answer_data"},
    {"$project": {
        "question": "$question_text",
        "answer": "$answer_data.text",
        "_id": 0
    }},
    {"$limit": 10000}
]

builder.load_from_mongodb_aggregation(
    connection_params={...},
    pipeline=pipeline
)
```

---

## Redis

Load data from Redis key-value store.

### Prerequisites

```bash
pip install redis
```

### Example

```python
from slm_builder import SLMBuilder

builder = SLMBuilder(
    project_name="redis-qa",
    base_model="google/flan-t5-small"
)

# Load from Redis
builder.load_from_redis(
    connection_params={
        "host": "localhost",
        "port": 6379,
        "db": 0,
        "password": "your_password"
    },
    key_pattern="qa:*",  # Match keys
    value_type="json"     # or "string", "hash"
)

builder.train(epochs=3, method="lora")
```

### Redis Hash Example

```python
# Load from Redis hashes
builder.load_from_redis(
    connection_params={...},
    key_pattern="qa:question:*",
    value_type="hash",
    hash_fields=["question", "answer", "category"]
)
```

---

## Elasticsearch

Load data from Elasticsearch indices.

### Prerequisites

```bash
pip install elasticsearch
```

### Basic Example

```python
from slm_builder import SLMBuilder

builder = SLMBuilder(
    project_name="elastic-qa",
    base_model="google/flan-t5-small"
)

# Load from Elasticsearch
builder.load_from_elasticsearch(
    connection_params={
        "host": "localhost",
        "port": 9200,
        "user": "elastic",
        "password": "password"
    },
    index="qa_documents",
    query={
        "match_all": {}
    }
)

builder.train(epochs=3, method="lora")
```

### Advanced Query

```python
# Complex Elasticsearch query
query = {
    "bool": {
        "must": [
            {"match": {"status": "approved"}},
            {"range": {"upvotes": {"gte": 5}}}
        ],
        "filter": [
            {"terms": {"category": ["tech", "programming"]}},
            {"exists": {"field": "answer"}}
        ]
    }
}

builder.load_from_elasticsearch(
    connection_params={...},
    index="qa_documents",
    query=query,
    size=10000,  # Batch size
    source=["question", "answer", "metadata"]
)
```

---

## Multi-Database Integration

Load data from multiple databases and combine them.

### Example

```python
from slm_builder import SLMBuilder

builder = SLMBuilder(
    project_name="multi-db",
    base_model="google/flan-t5-base"
)

# Load from PostgreSQL
builder.load_from_database(
    connection_params={"dialect": "postgresql", ...},
    query="SELECT question, answer FROM qa_table"
)

# Add MongoDB data
builder.load_from_mongodb(
    connection_params={...},
    query={"verified": True}
)

# Add Elasticsearch data
builder.load_from_elasticsearch(
    connection_params={...},
    index="qa_index",
    query={"match_all": {}}
)

print(f"Total samples from all sources: {builder.dataset_size}")

# Train on combined data
builder.train(epochs=5, method="lora")
```

---

## Connection Pooling

Efficient database connections for large datasets.

### Example

```python
from slm_builder import SLMBuilder
from slm_builder.data import DatabaseLoader
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

# Create engine with connection pooling
engine = create_engine(
    "postgresql://user:pass@localhost/db",
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20
)

# Use with builder
loader = DatabaseLoader(engine=engine)
data = loader.load_from_query(
    "SELECT question, answer FROM qa_pairs"
)

builder = SLMBuilder(project_name="pooled-connection")
builder.load_data(data)
builder.train(epochs=3, method="lora")
```

---

## Incremental Loading

Load large datasets incrementally to manage memory.

### Example

```python
from slm_builder import SLMBuilder

builder = SLMBuilder(
    project_name="incremental",
    base_model="google/flan-t5-small"
)

# Load in batches
batch_size = 10000
offset = 0

while True:
    query = f"""
        SELECT question, answer 
        FROM qa_pairs 
        LIMIT {batch_size} OFFSET {offset}
    """
    
    batch = builder.load_from_database(
        connection_params={...},
        query=query,
        append=True  # Append to existing data
    )
    
    if len(batch) == 0:
        break
    
    offset += batch_size
    print(f"Loaded {offset} samples...")

print(f"Total samples: {builder.dataset_size}")
builder.train(epochs=3, method="lora")
```

---

## Error Handling

Handle database connection errors gracefully.

### Example

```python
from slm_builder import SLMBuilder
from slm_builder.exceptions import DatabaseConnectionError
import time

def load_with_retry(builder, connection_params, query, max_retries=3):
    """Load data with automatic retry on failure."""
    for attempt in range(max_retries):
        try:
            builder.load_from_database(
                connection_params=connection_params,
                query=query
            )
            print(f"✅ Data loaded successfully")
            return True
        except DatabaseConnectionError as e:
            print(f"❌ Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                print("❌ All retries failed")
                raise
    return False

# Usage
builder = SLMBuilder(project_name="retry-example")
load_with_retry(
    builder,
    connection_params={...},
    query="SELECT question, answer FROM qa_table"
)
```

---

## Next Steps

- [API Integration](api-integration) - Load from REST APIs
- [Advanced Examples](advanced-examples) - Production use cases
- [Data Sources Reference](../features/data-sources) - All data sources

## Need Help?

- [GitHub Issues](https://github.com/isathish/slm/issues)
- [Discussions](https://github.com/isathish/slm/discussions)
