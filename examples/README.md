# SLM Builder Examples

This directory contains comprehensive examples demonstrating all features of the SLM Builder package.

## 📋 Example Overview

### Basic Usage

#### [build_qa_from_csv.py](build_qa_from_csv.py)
**Purpose**: Basic QA model training from CSV files

**What it demonstrates**:
- Loading data from CSV files
- Building a QA model with LoRA fine-tuning
- Basic configuration and training workflow

**Prerequisites**: CSV file with `question` and `answer` columns

**Run**:
```bash
python examples/build_qa_from_csv.py
```

---

### Database Integration

#### [build_from_database.py](build_from_database.py)
**Purpose**: Load training data from SQL databases (PostgreSQL, MySQL, SQLite)

**What it demonstrates**:
- SQL database connections with different dialects
- Custom SQL queries for data extraction
- Column mapping for QA tasks
- Error handling and connection management

**Prerequisites**:
- PostgreSQL/MySQL/SQLite database running
- Database credentials
- `sqlalchemy`, `psycopg2-binary` (PostgreSQL), or `pymysql` (MySQL) installed

**Configuration**:
```python
connection_params = {
    "dialect": "postgresql",  # or "mysql", "sqlite"
    "host": "localhost",
    "port": 5432,
    "database": "your_db",
    "user": "your_user",
    "password": "your_password"
}
```

**Run**:
```bash
pip install sqlalchemy psycopg2-binary
python examples/build_from_database.py
```

---

#### [build_from_mongodb.py](build_from_mongodb.py)
**Purpose**: Load training data from MongoDB collections

**What it demonstrates**:
- MongoDB connection and authentication
- Query filters and field projections
- Document-to-record conversion
- Handling nested MongoDB documents

**Prerequisites**:
- MongoDB instance running
- `pymongo` installed

**Configuration**:
```python
connection_params = {
    "host": "localhost",
    "port": 27017,
    "database": "qa_database",
    "user": "admin",
    "password": "password"
}
```

**Run**:
```bash
pip install pymongo
python examples/build_from_mongodb.py
```

---

### API Integration

#### [build_from_api.py](build_from_api.py)
**Purpose**: Load training data from REST APIs

**What it demonstrates**:
- Multiple authentication methods (Bearer token, API key, Basic auth)
- Pagination strategies (offset, page-based, cursor-based)
- Rate limiting and throttling
- Custom response parsers
- Error handling and retries

**Prerequisites**:
- REST API endpoint
- API credentials
- `requests`, `tqdm` installed

**Supported Authentication**:
- Bearer token authentication
- API key authentication (header or query param)
- Basic HTTP authentication
- OAuth2 (token-based)

**Supported Pagination**:
- Offset-based (e.g., `?offset=0&limit=100`)
- Page-based (e.g., `?page=1&per_page=100`)
- Cursor-based (e.g., `?cursor=abc123`)

**Run**:
```bash
pip install requests tqdm
python examples/build_from_api.py
```

---

### Data Preparation

#### [dataset_splitting.py](dataset_splitting.py)
**Purpose**: Dataset validation, splitting, and quality analysis

**What it demonstrates**:
- Dataset quality validation with detailed reports
- Train/validation/test splitting with stratification
- K-fold cross-validation
- Class balance checking and imbalance detection
- Random seed control for reproducibility

**Features**:
- **Validation**: Check dataset quality, missing values, duplicates
- **Splitting**: Multiple strategies (simple, stratified, time-based)
- **K-Fold CV**: Cross-validation with configurable folds
- **Balance Analysis**: Detect class imbalances and distribution issues

**Run**:
```bash
python examples/dataset_splitting.py
```

**Output**:
- Validation reports with quality metrics
- Split datasets (train/val/test)
- Class distribution analysis
- Recommendations for handling imbalances

---

### Model Comparison

#### [model_comparison.py](model_comparison.py)
**Purpose**: Compare multiple models and track experiments

**What it demonstrates**:
- Training multiple models with different configurations
- Comprehensive metric comparison (perplexity, accuracy, BLEU, ROUGE, F1)
- Model ranking by performance
- Report generation (Markdown, HTML, text, JSON)
- Experiment tracking with hyperparameter logging
- Finding best performing models

**Features**:
- **Multi-Model Evaluation**: Compare 2+ models on same dataset
- **Metrics**: Perplexity, Accuracy, BLEU, ROUGE, F1, training time
- **Rankings**: Sort models by any metric
- **Reports**: Generate comparison reports in multiple formats
- **Tracking**: Log experiments with hyperparameters and metadata

**Prerequisites**:
- `nltk`, `rouge-score` for BLEU and ROUGE metrics (optional)

**Run**:
```bash
pip install nltk rouge-score  # Optional for advanced metrics
python examples/model_comparison.py
```

**Output**:
- Comparison reports in `comparison_reports/`
- Experiment logs in `experiments/`
- Rankings by metric
- Performance summaries

---

## 🔧 Installation

### Core Dependencies
```bash
pip install torch transformers datasets peft accelerate bitsandbytes
```

### Optional Dependencies

For **database support**:
```bash
pip install sqlalchemy psycopg2-binary pymongo pymysql
```

For **API loading**:
```bash
pip install requests tqdm
```

For **advanced metrics**:
```bash
pip install nltk rouge-score
```

For **all features**:
```bash
pip install sqlalchemy psycopg2-binary pymongo requests tqdm nltk rouge-score
```

---

## 📊 Example Workflow

### Complete Training Pipeline

```python
from slm_builder import SLMBuilder

# Step 1: Initialize builder
builder = SLMBuilder(project_name="my-slm", base_model="gpt2")

# Step 2: Load data from database
result = builder.build_from_database(
    query="SELECT question, answer FROM qa_table WHERE quality > 0.8",
    connection_params={
        "dialect": "postgresql",
        "host": "localhost",
        "database": "mydb",
        "user": "user",
        "password": "pass"
    },
    db_type="sql",
    task="qa",
    recipe="lora"
)

# Step 3: Validate and split data
from slm_builder.data import validate_dataset, split_dataset

validation_report = validate_dataset(result['dataset'], task='qa')
splits = split_dataset(
    result['dataset'],
    test_size=0.2,
    val_size=0.15,
    stratify_by='label.label'
)

# Step 4: Compare models
from slm_builder.models import compare_models

models = [
    {'model': 'gpt2', 'name': 'GPT2-Base'},
    {'model': 'distilgpt2', 'name': 'DistilGPT2'}
]

comparison = compare_models(
    models=models,
    test_dataset=splits['test'],
    metrics=['perplexity', 'accuracy', 'bleu']
)

print(f"Best model: {comparison.get_rankings()[0]['name']}")
```

---

## 🎯 Use Cases

### Use Case 1: Customer Support QA Bot
**Goal**: Build a QA model from existing support tickets

**Steps**:
1. Load tickets from PostgreSQL: `build_from_database.py`
2. Validate data quality: `dataset_splitting.py`
3. Train with LoRA: `build_qa_from_csv.py`
4. Compare with baseline: `model_comparison.py`

### Use Case 2: API-Driven Content Model
**Goal**: Build a model from API-sourced content

**Steps**:
1. Load from REST API: `build_from_api.py`
2. Split and validate: `dataset_splitting.py`
3. Train and evaluate: `build_qa_from_csv.py`

### Use Case 3: Multi-Source Knowledge Base
**Goal**: Combine data from multiple sources

**Steps**:
1. Load from MongoDB: `build_from_mongodb.py`
2. Load from API: `build_from_api.py`
3. Merge datasets and validate: `dataset_splitting.py`
4. Compare models: `model_comparison.py`

---

## 📝 Tips and Best Practices

### Data Loading
- Always validate connection parameters before loading
- Use query filters to reduce data transfer
- Handle authentication securely (environment variables)
- Implement retry logic for API calls

### Dataset Preparation
- Always validate before splitting
- Use stratification for imbalanced datasets
- Set random seeds for reproducibility
- Check for duplicates and missing values

### Model Training
- Start with LoRA for faster iterations
- Use quantization for limited resources
- Monitor validation metrics during training
- Save checkpoints regularly

### Experiment Tracking
- Log all hyperparameters
- Track multiple metrics (not just loss)
- Compare with baseline models
- Generate reports for documentation

---

## 🐛 Troubleshooting

### Database Connection Issues
```python
# Test connection first
import sqlalchemy
engine = sqlalchemy.create_engine("postgresql://user:pass@localhost/db")
with engine.connect() as conn:
    result = conn.execute("SELECT 1")
    print("Connection successful")
```

### API Authentication Issues
```python
# Test API endpoint
import requests
response = requests.get(
    "https://api.example.com/data",
    headers={"Authorization": "Bearer YOUR_TOKEN"}
)
print(response.status_code, response.json())
```

### Import Errors
```bash
# Install missing dependencies
pip install sqlalchemy psycopg2-binary pymongo requests tqdm nltk rouge-score
```

---

## 📚 Additional Resources

- **[FEATURES.md](../FEATURES.md)** - Dynamic model loading and core features
- **[ADDITIONAL_FEATURES.md](../ADDITIONAL_FEATURES.md)** - Advanced features documentation
- **[CONTRIBUTING.md](../CONTRIBUTING.md)** - Contributing guidelines
- **[README.md](../README.md)** - Main project documentation

---

## 🤝 Need Help?

- Check the main documentation in the parent directory
- Review ADDITIONAL_FEATURES.md for detailed API documentation
- Open an issue on GitHub for bugs or feature requests
- See CONTRIBUTING.md for development guidelines

---

**Happy Building! 🚀**
