---
layout: default
title: Getting Started Examples
parent: Examples
nav_order: 3
---

# Getting Started Examples
{: .no_toc }

Simple examples to get you started with SLM Builder.
{: .fs-6 .fw-300 }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Basic QA Model from CSV

Build a question-answering model from a CSV file.

### Prerequisites

- CSV file with `question` and `answer` columns
- Python 3.8+
- SLM Builder installed

### Code

```python
from slm_builder import SLMBuilder

# Initialize builder
builder = SLMBuilder(
    project_name="qa-from-csv",
    base_model="google/flan-t5-small"
)

# Load data from CSV
builder.load_data_from_csv("qa_data.csv")

# Train model
builder.train(
    epochs=3,
    learning_rate=2e-5,
    method="lora",
    batch_size=8
)

# Evaluate
metrics = builder.evaluate()
print(f"Accuracy: {metrics['accuracy']:.2%}")
print(f"F1 Score: {metrics['f1']:.3f}")

# Test the model
questions = [
    "What is Python?",
    "How do I install packages?",
    "What is pip?"
]

for question in questions:
    answer = builder.generate(question)
    print(f"Q: {question}")
    print(f"A: {answer}\n")

# Export
builder.export_model("./output", format="huggingface")
```

### Expected Output

```
Training: 100%|██████████| 300/300 [02:45<00:00, 1.81it/s]
Evaluating: 100%|██████████| 50/50 [00:15<00:00, 3.25it/s]

Accuracy: 92.50%
F1 Score: 0.934

Q: What is Python?
A: Python is a high-level programming language...

✅ Model exported to ./output
```

---

## JSON/JSONL Data Loading

Load training data from JSON or JSONL files.

### JSON Format

```json
[
  {
    "question": "What is machine learning?",
    "answer": "Machine learning is a subset of AI..."
  },
  {
    "question": "What is deep learning?",
    "answer": "Deep learning uses neural networks..."
  }
]
```

### JSONL Format

```jsonl
{"question": "What is machine learning?", "answer": "Machine learning is..."}
{"question": "What is deep learning?", "answer": "Deep learning uses..."}
```

### Code

```python
from slm_builder import SLMBuilder

builder = SLMBuilder(
    project_name="qa-from-json",
    base_model="google/flan-t5-small"
)

# Load from JSON
builder.load_data_from_json("data.json")

# Or load from JSONL
builder.load_data_from_jsonl("data.jsonl")

# Train and evaluate
builder.train(epochs=3, method="lora")
metrics = builder.evaluate()

print(f"✅ Training complete! Accuracy: {metrics['accuracy']:.2%}")
```

---

## Multi-Format Data Loading

Load data from multiple sources and merge them.

### Code

```python
from slm_builder import SLMBuilder

builder = SLMBuilder(
    project_name="multi-source",
    base_model="google/flan-t5-base"
)

# Load from multiple sources
builder.load_data_from_csv("dataset1.csv")
builder.load_data_from_json("dataset2.json")
builder.load_data_from_jsonl("dataset3.jsonl")

# View dataset stats
print(f"Total samples: {builder.dataset_size}")
print(f"Train samples: {builder.train_size}")
print(f"Test samples: {builder.test_size}")

# Train on combined data
builder.train(
    epochs=5,
    learning_rate=3e-5,
    method="lora"
)

# Evaluate
metrics = builder.evaluate()
print(f"Combined model accuracy: {metrics['accuracy']:.2%}")
```

---

## Custom Data Preprocessing

Apply custom preprocessing to your data.

### Code

```python
from slm_builder import SLMBuilder
import pandas as pd

def preprocess_data(df):
    """Custom preprocessing function."""
    # Convert to lowercase
    df['question'] = df['question'].str.lower()
    df['answer'] = df['answer'].str.lower()
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['question'])
    
    # Filter by length
    df = df[df['question'].str.len() > 10]
    df = df[df['answer'].str.len() > 20]
    
    return df

# Load and preprocess
builder = SLMBuilder(
    project_name="custom-preprocessing",
    base_model="google/flan-t5-small"
)

# Load raw data
raw_data = pd.read_csv("raw_data.csv")

# Apply preprocessing
processed_data = preprocess_data(raw_data)

# Save and load
processed_data.to_csv("processed_data.csv", index=False)
builder.load_data_from_csv("processed_data.csv")

# Train
builder.train(epochs=3, method="lora")
```

---

## Train-Test Split Configuration

Configure how your data is split for training and testing.

### Code

```python
from slm_builder import SLMBuilder
from slm_builder.data import DatasetSplitter

builder = SLMBuilder(
    project_name="custom-split",
    base_model="google/flan-t5-small"
)

# Load data
builder.load_data_from_csv("data.csv")

# Custom split configuration
splitter = DatasetSplitter()

# 80% train, 20% test
train_data, test_data = splitter.train_test_split(
    builder.dataset,
    test_size=0.2,
    shuffle=True,
    random_state=42
)

# Or 70% train, 15% val, 15% test
train_data, val_data, test_data = splitter.train_val_test_split(
    builder.dataset,
    val_size=0.15,
    test_size=0.15,
    random_state=42
)

# Apply to builder
builder.set_datasets(train=train_data, test=test_data)

# Train
builder.train(epochs=3, method="lora")
```

---

## Batch Inference

Process multiple inputs efficiently.

### Code

```python
from slm_builder import SLMBuilder

builder = SLMBuilder(
    project_name="batch-inference",
    base_model="google/flan-t5-small"
)

# Load and train
builder.load_data_from_csv("data.csv")
builder.train(epochs=3, method="lora")

# Prepare batch of questions
questions = [
    "What is Python?",
    "How do I create a list?",
    "What is a dictionary?",
    "How do I read a file?",
    "What is a function?"
]

# Batch inference
responses = builder.generate_batch(
    questions,
    batch_size=8,
    max_length=128
)

# Display results
for q, a in zip(questions, responses):
    print(f"Q: {q}")
    print(f"A: {a}\n")
```

---

## Model Saving and Loading

Save trained models and load them later.

### Saving

```python
from slm_builder import SLMBuilder

# Train model
builder = SLMBuilder(project_name="saved-model")
builder.load_data_from_csv("data.csv")
builder.train(epochs=3, method="lora")

# Save model
builder.save_model("./my-model")
print("✅ Model saved to ./my-model")
```

### Loading

```python
from slm_builder import SLMBuilder

# Load saved model
builder = SLMBuilder.load_model("./my-model")

# Use immediately
response = builder.generate("What is Python?")
print(response)

# Continue training if needed
builder.load_data_from_csv("new_data.csv")
builder.train(epochs=2, method="lora")
```

---

## Configuration File

Use configuration files for reproducible experiments.

### config.yaml

```yaml
project:
  name: "qa-model"
  base_model: "google/flan-t5-small"

data:
  source: "qa_data.csv"
  test_size: 0.2
  random_state: 42

training:
  method: "lora"
  epochs: 5
  learning_rate: 2e-5
  batch_size: 8
  warmup_steps: 100

evaluation:
  metrics:
    - accuracy
    - f1
    - bleu
    - rouge

export:
  format: "huggingface"
  output_dir: "./output"
```

### Code

```python
from slm_builder import SLMBuilder
from slm_builder.config import load_config

# Load configuration
config = load_config("config.yaml")

# Initialize from config
builder = SLMBuilder.from_config(config)

# Load data
builder.load_data_from_csv(config['data']['source'])

# Train
builder.train(**config['training'])

# Evaluate
metrics = builder.evaluate(metrics=config['evaluation']['metrics'])

# Export
builder.export_model(
    config['export']['output_dir'],
    format=config['export']['format']
)
```

---

## Next Steps

### Try More Advanced Examples

- [Database Integration](database-integration) - Load from databases
- [API Integration](api-integration) - Fetch data from APIs
- [Advanced Examples](advanced-examples) - Production use cases

### Learn About Features

- [Training Methods](../features/training) - LoRA, QLoRA, full fine-tuning
- [Evaluation Metrics](../features/evaluation) - Understand metrics
- [Model Export](../features/export) - Export formats

### Get Help

- [GitHub Issues](https://github.com/isathish/slm/issues)
- [Discussions](https://github.com/isathish/slm/discussions)
