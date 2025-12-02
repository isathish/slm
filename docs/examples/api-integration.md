---
layout: default
title: API Integration
parent: Examples
nav_order: 3
---

# API Integration Examples
{: .no_toc }

Fetch training data from REST APIs, authenticated endpoints, and webhook responses.
{: .fs-6 .fw-300 }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Basic REST API

Load data from a simple REST API endpoint.

### Example

```python
from slm_builder import SLMBuilder

builder = SLMBuilder(
    project_name="api-qa",
    base_model="google/flan-t5-small"
)

# Load from public API
builder.load_from_api(
    endpoint="https://api.example.com/qa-data",
    method="GET"
)

builder.train(epochs=3, method="lora")
```

---

## Authenticated API

Load data from an API that requires authentication.

### Bearer Token

```python
from slm_builder import SLMBuilder

builder = SLMBuilder(
    project_name="auth-api",
    base_model="google/flan-t5-small"
)

# API with Bearer token
builder.load_from_api(
    endpoint="https://api.example.com/training-data",
    headers={
        "Authorization": "Bearer YOUR_ACCESS_TOKEN",
        "Content-Type": "application/json"
    }
)

builder.train(epochs=3, method="lora")
```

### API Key

```python
# API with API key
builder.load_from_api(
    endpoint="https://api.example.com/data",
    headers={
        "X-API-Key": "your-api-key-here"
    }
)
```

### Basic Auth

```python
import requests
from requests.auth import HTTPBasicAuth

# API with basic authentication
builder.load_from_api(
    endpoint="https://api.example.com/data",
    auth=HTTPBasicAuth("username", "password")
)
```

---

## Paginated API

Handle APIs with pagination.

### Offset-based Pagination

```python
from slm_builder import SLMBuilder

builder = SLMBuilder(project_name="paginated-api")

page = 0
page_size = 100
total_loaded = 0

while True:
    response = builder.load_from_api(
        endpoint=f"https://api.example.com/data?page={page}&size={page_size}",
        headers={"Authorization": "Bearer TOKEN"},
        append=True  # Append to existing data
    )
    
    if len(response) == 0:
        break
    
    total_loaded += len(response)
    page += 1
    print(f"Loaded {total_loaded} samples...")

print(f"✅ Total samples loaded: {total_loaded}")
builder.train(epochs=3, method="lora")
```

### Cursor-based Pagination

```python
cursor = None
total_loaded = 0

while True:
    url = f"https://api.example.com/data"
    if cursor:
        url += f"?cursor={cursor}"
    
    response = builder.load_from_api(
        endpoint=url,
        headers={"Authorization": "Bearer TOKEN"},
        append=True
    )
    
    # Extract next cursor from response
    cursor = response.get('metadata', {}).get('next_cursor')
    
    if not cursor:
        break
    
    total_loaded += len(response.get('data', []))
    print(f"Loaded {total_loaded} samples...")
```

---

## POST Requests

Send POST requests to APIs with custom payloads.

### Example

```python
from slm_builder import SLMBuilder
import json

builder = SLMBuilder(project_name="post-api")

# POST request with JSON payload
payload = {
    "query": {
        "category": "technology",
        "verified": True
    },
    "limit": 5000
}

builder.load_from_api(
    endpoint="https://api.example.com/query",
    method="POST",
    headers={
        "Authorization": "Bearer TOKEN",
        "Content-Type": "application/json"
    },
    data=json.dumps(payload)
)

builder.train(epochs=3, method="lora")
```

---

## GraphQL API

Query GraphQL endpoints.

### Example

```python
from slm_builder import SLMBuilder
import json

builder = SLMBuilder(project_name="graphql-api")

# GraphQL query
graphql_query = """
query {
  qaData(first: 1000, verified: true) {
    nodes {
      question
      answer
      metadata {
        category
        difficulty
      }
    }
  }
}
"""

payload = {
    "query": graphql_query
}

builder.load_from_api(
    endpoint="https://api.example.com/graphql",
    method="POST",
    headers={
        "Authorization": "Bearer TOKEN",
        "Content-Type": "application/json"
    },
    data=json.dumps(payload),
    response_parser=lambda r: r.json()['data']['qaData']['nodes']
)

builder.train(epochs=3, method="lora")
```

---

## Custom Response Parsing

Parse complex API responses.

### Nested JSON

```python
from slm_builder import SLMBuilder

def parse_nested_response(response):
    """Extract QA pairs from nested JSON."""
    data = response.json()
    qa_pairs = []
    
    for item in data['results']['items']:
        qa_pairs.append({
            'question': item['content']['question'],
            'answer': item['content']['answer'],
            'metadata': {
                'category': item['metadata']['category'],
                'confidence': item['scores']['confidence']
            }
        })
    
    return qa_pairs

builder = SLMBuilder(project_name="custom-parser")

builder.load_from_api(
    endpoint="https://api.example.com/complex-data",
    headers={"Authorization": "Bearer TOKEN"},
    response_parser=parse_nested_response
)

builder.train(epochs=3, method="lora")
```

---

## Rate Limiting

Handle API rate limits with automatic retry.

### Example

```python
from slm_builder import SLMBuilder
import time
import requests

def load_with_rate_limit(builder, endpoint, headers, max_requests_per_minute=60):
    """Load data respecting rate limits."""
    request_interval = 60 / max_requests_per_minute
    
    page = 0
    total_loaded = 0
    
    while True:
        try:
            response = builder.load_from_api(
                endpoint=f"{endpoint}?page={page}",
                headers=headers,
                append=True
            )
            
            if len(response) == 0:
                break
            
            total_loaded += len(response)
            page += 1
            
            # Respect rate limit
            time.sleep(request_interval)
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:  # Rate limit exceeded
                retry_after = int(e.response.headers.get('Retry-After', 60))
                print(f"Rate limit hit. Waiting {retry_after}s...")
                time.sleep(retry_after)
                continue
            else:
                raise
    
    return total_loaded

# Usage
builder = SLMBuilder(project_name="rate-limited")
total = load_with_rate_limit(
    builder,
    endpoint="https://api.example.com/data",
    headers={"Authorization": "Bearer TOKEN"},
    max_requests_per_minute=30
)

print(f"✅ Loaded {total} samples")
builder.train(epochs=3, method="lora")
```

---

## Webhook Integration

Process webhook payloads for training.

### Example

```python
from flask import Flask, request
from slm_builder import SLMBuilder
import json

app = Flask(__name__)

builder = SLMBuilder(
    project_name="webhook-training",
    base_model="google/flan-t5-small"
)

@app.route('/webhook', methods=['POST'])
def handle_webhook():
    """Receive and process webhook data."""
    data = request.json
    
    # Extract QA pairs from webhook
    qa_pairs = []
    for item in data.get('items', []):
        qa_pairs.append({
            'question': item['question'],
            'answer': item['answer']
        })
    
    # Add to training data
    builder.add_training_samples(qa_pairs)
    
    # Trigger training if threshold reached
    if builder.dataset_size >= 1000:
        builder.train(epochs=2, method="lora")
        builder.save_model("./webhook-model")
        builder.clear_dataset()  # Reset for next batch
    
    return {"status": "success", "samples_added": len(qa_pairs)}

if __name__ == '__main__':
    app.run(port=5000)
```

---

## Multi-API Integration

Combine data from multiple APIs.

### Example

```python
from slm_builder import SLMBuilder

builder = SLMBuilder(
    project_name="multi-api",
    base_model="google/flan-t5-base"
)

# API 1: Internal knowledge base
builder.load_from_api(
    endpoint="https://internal-api.company.com/kb",
    headers={"Authorization": "Bearer INTERNAL_TOKEN"}
)

# API 2: External data provider
builder.load_from_api(
    endpoint="https://external-api.com/qa-data",
    headers={"X-API-Key": "EXTERNAL_KEY"},
    append=True
)

# API 3: Third-party service
builder.load_from_api(
    endpoint="https://service.com/training-data",
    auth=("username", "password"),
    append=True
)

print(f"Total samples from all APIs: {builder.dataset_size}")
builder.train(epochs=5, method="lora")
```

---

## Streaming API

Handle streaming API responses.

### Example

```python
from slm_builder import SLMBuilder
import requests
import json

builder = SLMBuilder(project_name="streaming-api")

url = "https://api.example.com/stream"
headers = {"Authorization": "Bearer TOKEN"}

# Stream large datasets
with requests.get(url, headers=headers, stream=True) as response:
    response.raise_for_status()
    
    batch = []
    batch_size = 100
    
    for line in response.iter_lines():
        if line:
            item = json.loads(line)
            batch.append({
                'question': item['question'],
                'answer': item['answer']
            })
            
            # Process in batches
            if len(batch) >= batch_size:
                builder.add_training_samples(batch)
                batch = []
                print(f"Processed {builder.dataset_size} samples...")

    # Add remaining samples
    if batch:
        builder.add_training_samples(batch)

print(f"✅ Total samples: {builder.dataset_size}")
builder.train(epochs=3, method="lora")
```

---

## Caching API Responses

Cache responses to avoid repeated API calls.

### Example

```python
from slm_builder import SLMBuilder
import requests
import json
import hashlib
from pathlib import Path

def load_with_cache(endpoint, headers, cache_dir="./api_cache"):
    """Load from API with disk caching."""
    # Create cache key
    cache_key = hashlib.md5(
        f"{endpoint}{json.dumps(headers)}".encode()
    ).hexdigest()
    
    cache_file = Path(cache_dir) / f"{cache_key}.json"
    cache_file.parent.mkdir(exist_ok=True)
    
    # Check cache
    if cache_file.exists():
        print("✅ Loading from cache...")
        with open(cache_file) as f:
            return json.load(f)
    
    # Fetch from API
    print("📡 Fetching from API...")
    response = requests.get(endpoint, headers=headers)
    response.raise_for_status()
    data = response.json()
    
    # Save to cache
    with open(cache_file, 'w') as f:
        json.dump(data, f)
    
    return data

# Usage
builder = SLMBuilder(project_name="cached-api")

data = load_with_cache(
    endpoint="https://api.example.com/training-data",
    headers={"Authorization": "Bearer TOKEN"}
)

builder.load_data(data)
builder.train(epochs=3, method="lora")
```

---

## Next Steps

- [Advanced Examples](advanced-examples) - Production use cases
- [Data Sources](../features/data-sources) - All data source options
- [Configuration](../reference/configuration) - API configuration

## Need Help?

- [GitHub Issues](https://github.com/isathish/slm/issues)
- [Discussions](https://github.com/isathish/slm/discussions)
