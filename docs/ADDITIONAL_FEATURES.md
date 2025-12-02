---
layout: default
title: Additional Features
nav_order: 6
---

# Additional Features Documentation

This document covers the advanced features added to the SLM Builder package, including dataset splitting, database loaders, API loaders, and model comparison.

## Table of Contents
- [Dataset Splitting and Validation](#dataset-splitting-and-validation)
- [Database Loaders](#database-loaders)
- [API Data Loaders](#api-data-loaders)
- [Model Comparison and Benchmarking](#model-comparison-and-benchmarking)
- [Usage Examples](#usage-examples)

---

## Dataset Splitting and Validation

### Overview
The dataset splitting module provides utilities for splitting datasets with stratification support and validating dataset quality.

### Key Features
- **Train/Test Split**: Basic two-way splitting
- **Train/Val/Test Split**: Three-way splitting for validation
- **K-Fold Cross-Validation**: Create multiple train/val splits
- **Stratified Splitting**: Maintain class distribution across splits
- **Dataset Validation**: Check dataset quality and consistency
- **Class Balance Checking**: Identify imbalanced datasets

### Classes

#### DatasetSplitter
```python
from slm_builder.data import DatasetSplitter

splitter = DatasetSplitter()

# Train/test split
train, test = splitter.train_test_split(
    dataset,
    test_size=0.2,
    stratify_by='label.label',  # For classification
    shuffle=True,
    random_state=42
)

# Train/val/test split
train, val, test = splitter.train_val_test_split(
    dataset,
    val_size=0.15,
    test_size=0.15,
    stratify_by='label.label',
    shuffle=True,
    random_state=42
)

# K-fold cross-validation
folds = splitter.k_fold_split(
    dataset,
    n_folds=5,
    shuffle=True,
    random_state=42
)

for fold_idx, (train_fold, val_fold) in enumerate(folds):
    print(f"Fold {fold_idx}: Train={len(train_fold)}, Val={len(val_fold)}")
```

#### DatasetValidator
```python
from slm_builder.data import DatasetValidator

validator = DatasetValidator()

# Validate dataset
report = validator.validate_dataset(
    dataset,
    task='qa',  # or 'classification', 'instruction', 'generation'
    strict=False  # Set to True to raise on errors
)

print(f"Valid samples: {report['valid_samples']}/{report['total_samples']}")
print(f"Validity rate: {report['validity_rate']:.2%}")
print(f"Errors: {len(report['errors'])}")
print(f"Warnings: {len(report['warnings'])}")

# Check class balance
distribution = validator.check_class_balance(
    dataset,
    stratify_by='label.label'
)

print(f"Classes: {distribution['n_classes']}")
print(f"Balanced: {distribution['is_balanced']}")
print(f"Imbalance ratio: {distribution['imbalance_ratio']:.2f}")
```

### Convenience Functions
```python
from slm_builder.data import split_dataset, validate_dataset

# Simple split
train, test = split_dataset(dataset, test_size=0.2)

# With validation set
train, val, test = split_dataset(dataset, test_size=0.15, val_size=0.15)

# Validate
report = validate_dataset(dataset, task='qa', strict=False)
```

---

## Database Loaders

### Overview
Load data directly from SQL and NoSQL databases.

### Supported Databases
- **SQL**: PostgreSQL, MySQL, SQLite, and other SQLAlchemy-compatible databases
- **NoSQL**: MongoDB

### SQL Loader

#### Requirements
```bash
pip install sqlalchemy psycopg2-binary  # For PostgreSQL
pip install sqlalchemy pymysql  # For MySQL
```

#### Usage
```python
from slm_builder.data import load_from_sql

# Connection parameters
connection_params = {
    'dialect': 'postgresql',  # or 'mysql', 'sqlite'
    'host': 'localhost',
    'port': 5432,
    'database': 'mydb',
    'user': 'username',
    'password': 'password'
}

# SQL query
query = """
SELECT 
    id,
    question_text as text,
    answer_text,
    metadata
FROM qa_table
WHERE status = 'active'
LIMIT 1000
"""

# Column mapping (maps database columns to canonical format)
column_mapping = {
    'text': 'question_text',
    'label.answer': 'answer_text'
}

# Load data
records = load_from_sql(
    query=query,
    connection_params=connection_params,
    task='qa',
    column_mapping=column_mapping
)

print(f"Loaded {len(records)} records from SQL database")
```

#### SQLite Example
```python
connection_params = {
    'dialect': 'sqlite',
    'database': '/path/to/database.db'
}

query = "SELECT * FROM training_data"

records = load_from_sql(query, connection_params, task='classification')
```

### MongoDB Loader

#### Requirements
```bash
pip install pymongo
```

#### Usage
```python
from slm_builder.data import load_from_mongodb

# Connection parameters
connection_params = {
    'host': 'localhost',
    'port': 27017,
    'database': 'mydb',
    'username': 'user',  # Optional
    'password': 'pass'   # Optional
}

# Query filter (MongoDB query syntax)
query_filter = {
    'status': 'active',
    'category': {'$in': ['tech', 'science']}
}

# Projection (fields to include/exclude)
projection = {
    'text': 1,
    'label': 1,
    'metadata': 1,
    '_id': 0
}

# Load data
records = load_from_mongodb(
    collection_name='training_data',
    connection_params=connection_params,
    task='qa',
    query_filter=query_filter,
    projection=projection,
    limit=1000
)

print(f"Loaded {len(records)} records from MongoDB")
```

---

## API Data Loaders

### Overview
Load data from REST APIs with support for pagination, authentication, and rate limiting.

### Requirements
```bash
pip install requests tqdm
```

### Features
- **Multiple Authentication Methods**: Bearer, Basic, API Key, OAuth2
- **Pagination Support**: Offset-based, page-based, cursor-based
- **Rate Limiting**: Control request frequency
- **Custom Response Parsers**: Handle different API response structures
- **Progress Tracking**: Visual progress with tqdm

### Usage

#### Basic API Loading
```python
from slm_builder.data import load_from_api

# Simple API without authentication
records = load_from_api(
    base_url='https://api.example.com',
    endpoint='/v1/data',
    task='qa',
    max_pages=10
)
```

#### With Bearer Token Authentication
```python
auth = {
    'type': 'bearer',
    'token': 'your-api-token-here'
}

records = load_from_api(
    base_url='https://api.example.com',
    endpoint='/v1/training-data',
    task='classification',
    auth=auth,
    max_pages=20
)
```

#### With API Key Authentication
```python
auth = {
    'type': 'api_key',
    'key_name': 'X-API-Key',
    'key_value': 'your-api-key'
}

records = load_from_api(
    base_url='https://api.example.com',
    endpoint='/data',
    auth=auth
)
```

#### With Pagination
```python
# Offset-based pagination
pagination = {
    'type': 'offset',
    'param': 'offset',
    'size_param': 'limit',
    'page_size': 100
}

# Page-based pagination
pagination = {
    'type': 'page',
    'param': 'page',
    'size_param': 'per_page',
    'page_size': 50
}

# Cursor-based pagination
pagination = {
    'type': 'cursor',
    'param': 'cursor',
    'page_size': 100
}

records = load_from_api(
    base_url='https://api.example.com',
    endpoint='/items',
    task='instruction',
    pagination=pagination,
    max_pages=50
)
```

#### With Rate Limiting
```python
records = load_from_api(
    base_url='https://api.example.com',
    endpoint='/data',
    task='qa',
    rate_limit=2.0,  # 2 requests per second
    max_pages=100
)
```

#### Custom Response Parser
```python
def custom_parser(response):
    """Extract data from custom API response format."""
    if 'payload' in response:
        return response['payload']['items']
    return []

loader = APILoader(task='qa')
records = loader.load(
    base_url='https://api.example.com',
    endpoint='/custom',
    response_parser=custom_parser,
    max_pages=10
)
```

---

## Model Comparison and Benchmarking

### Overview
Compare multiple models on the same dataset and track experiments.

### Classes

#### ModelComparator
```python
from slm_builder.models import ModelComparator

comparator = ModelComparator(output_dir='./model_comparisons')

# Load models
models = [
    ('gpt2-base', model1, tokenizer1),
    ('gpt2-medium', model2, tokenizer2),
    ('distilgpt2', model3, tokenizer3)
]

# Compare models
results = comparator.compare_models(
    models=models,
    dataset=test_dataset,
    metrics=['perplexity', 'accuracy', 'bleu'],
    batch_size=8
)

# View results
print(f"Models compared: {results['n_models']}")
for model_name, model_results in results['models'].items():
    print(f"\n{model_name}:")
    print(f"  Perplexity: {model_results['metrics']['perplexity']:.4f}")
    print(f"  Accuracy: {model_results['metrics']['accuracy']:.4f}")
    print(f"  Time: {model_results['evaluation_time']:.2f}s")

# Generate report
markdown_report = comparator.generate_comparison_report(results, format='markdown')
print(markdown_report)

# Save as HTML
html_report = comparator.generate_comparison_report(results, format='html')
with open('comparison_report.html', 'w') as f:
    f.write(html_report)
```

#### ExperimentTracker
```python
from slm_builder.models import ExperimentTracker

tracker = ExperimentTracker(tracking_dir='./experiments')

# Log experiment
experiment_id = tracker.log_experiment(
    experiment_name='baseline_training',
    model_name='gpt2',
    hyperparameters={
        'learning_rate': 5e-5,
        'batch_size': 16,
        'epochs': 3,
        'lora_r': 8,
        'lora_alpha': 16
    },
    metrics={
        'train_loss': 2.35,
        'eval_loss': 2.42,
        'perplexity': 11.2,
        'accuracy': 0.87
    },
    notes='Baseline training with LoRA'
)

print(f"Experiment logged: {experiment_id}")

# List all experiments
experiments = tracker.list_experiments()
print(f"Total experiments: {len(experiments)}")

# Filter by model
gpt2_experiments = tracker.list_experiments(model_filter='gpt2')
print(f"GPT-2 experiments: {len(gpt2_experiments)}")

# Get best experiment
best_exp = tracker.get_best_experiment(
    metric='perplexity',
    minimize=True,  # Lower perplexity is better
    model_filter='gpt2'
)

print(f"Best experiment: {best_exp['id']}")
print(f"Best perplexity: {best_exp['metrics']['perplexity']:.4f}")
```

### Convenience Function
```python
from slm_builder.models import compare_models

results = compare_models(
    models=[
        ('model1', model1, tokenizer1),
        ('model2', model2, tokenizer2)
    ],
    dataset=test_data,
    metrics=['perplexity', 'accuracy'],
    output_dir='./comparisons'
)
```

---

## Usage Examples

### Complete Workflow Example

```python
from slm_builder import SLMBuilder
from slm_builder.data import (
    load_from_sql,
    split_dataset,
    validate_dataset
)
from slm_builder.models import compare_models

# 1. Load data from database
connection_params = {
    'dialect': 'postgresql',
    'host': 'localhost',
    'port': 5432,
    'database': 'training_db',
    'user': 'user',
    'password': 'pass'
}

query = "SELECT text, question, answer FROM qa_data WHERE status = 'verified'"

dataset = load_from_sql(
    query=query,
    connection_params=connection_params,
    task='qa'
)

# 2. Validate dataset
report = validate_dataset(dataset, task='qa', strict=False)
print(f"Dataset quality: {report['validity_rate']:.2%}")

# 3. Split dataset
train, val, test = split_dataset(
    dataset,
    test_size=0.15,
    val_size=0.15,
    stratify_by='label.label',
    random_state=42
)

# 4. Train multiple models
models_to_compare = []

for base_model in ['gpt2', 'distilgpt2']:
    builder = SLMBuilder(
        project_name=f'{base_model}_qa',
        base_model=base_model,
        task='qa'
    )
    
    builder.prepare_data(train)
    
    builder.train(
        epochs=3,
        learning_rate=5e-5,
        use_lora=True,
        lora_config={'r': 8, 'alpha': 16}
    )
    
    models_to_compare.append((
        base_model,
        builder.model,
        builder.tokenizer
    ))

# 5. Compare models
comparison = compare_models(
    models=models_to_compare,
    dataset=test,
    metrics=['perplexity', 'accuracy', 'bleu'],
    output_dir='./model_comparisons'
)

print("\nComparison Results:")
for model_name, results in comparison['models'].items():
    print(f"{model_name}:")
    print(f"  Perplexity: {results['metrics']['perplexity']:.4f}")
    print(f"  Accuracy: {results['metrics']['accuracy']:.4f}")
```

### API Data Loading Example

```python
from slm_builder.data import load_from_api
from slm_builder import SLMBuilder

# Load from API with authentication and pagination
auth = {
    'type': 'bearer',
    'token': 'your-api-token'
}

pagination = {
    'type': 'offset',
    'param': 'offset',
    'size_param': 'limit',
    'page_size': 100
}

dataset = load_from_api(
    base_url='https://api.yourcompany.com',
    endpoint='/training-data',
    task='instruction',
    auth=auth,
    pagination=pagination,
    rate_limit=5.0,  # 5 requests per second
    max_pages=50
)

print(f"Loaded {len(dataset)} records from API")

# Use with SLMBuilder
builder = SLMBuilder(
    project_name='api_trained_model',
    base_model='gpt2',
    task='instruction'
)

builder.prepare_data(dataset)
builder.train(epochs=3)
```

---

## Performance Considerations

### Dataset Splitting
- Stratified splitting ensures balanced class distribution
- Random state ensures reproducibility
- K-fold cross-validation provides robust evaluation

### Database Loaders
- Use column mapping for efficient data extraction
- Apply filters in SQL/MongoDB queries for better performance
- Use projections to load only needed fields
- Consider pagination for large datasets

### API Loaders
- Use rate limiting to avoid API throttling
- Set appropriate page sizes based on API limits
- Consider max_pages to limit data volume
- Cache responses when possible

### Model Comparison
- Use smaller batch sizes if memory is limited
- Compare models on same hardware for fair comparison
- Track evaluation time for deployment considerations
- Save comparison reports for documentation

---

## Dependencies

Additional dependencies for these features:

```bash
# Database support
pip install sqlalchemy psycopg2-binary pymongo

# API loading
pip install requests tqdm

# All features
pip install sqlalchemy psycopg2-binary pymongo requests tqdm
```

---

## Best Practices

1. **Data Validation**: Always validate datasets before training
2. **Stratification**: Use stratified splits for imbalanced datasets
3. **Cross-Validation**: Use K-fold for robust model evaluation
4. **Database Queries**: Optimize queries with indexes and filters
5. **API Rate Limits**: Respect API rate limits to avoid throttling
6. **Model Comparison**: Compare on consistent test sets
7. **Experiment Tracking**: Log all hyperparameters and metrics
8. **Reproducibility**: Set random seeds for reproducible splits

---

## Summary

These additional features provide:
- ✅ **Dataset splitting** with stratification and K-fold CV
- ✅ **Dataset validation** for quality assurance
- ✅ **SQL database** loading (PostgreSQL, MySQL, SQLite)
- ✅ **MongoDB** loading with query filters
- ✅ **REST API** loading with authentication and pagination
- ✅ **Model comparison** with multiple metrics
- ✅ **Experiment tracking** for reproducibility
- ✅ **Report generation** in multiple formats

All features integrate seamlessly with the existing SLM Builder workflow!
