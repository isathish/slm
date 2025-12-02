---
layout: default
title: Quick Start
parent: Guides
nav_order: 1
---

# Quick Start Guide
{: .no_toc }

Get started with SLM Builder in under 5 minutes.
{: .fs-6 .fw-300 }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Installation

```bash
pip install slm-builder
```

For specific features:

```bash
# Database support
pip install slm-builder[db]

# All features
pip install slm-builder[all]
```

---

## Your First Model

### 1. Import and Initialize

```python
from slm_builder import SLMBuilder

builder = SLMBuilder(
    project_name="my-first-model",
    base_model="google/flan-t5-small"
)
```

### 2. Load Data

From a CSV file:

```python
builder.load_data_from_csv("training_data.csv")
```

Your CSV should have these columns:
- `question` or `input`
- `answer` or `output`

### 3. Train the Model

```python
builder.train(
    epochs=3,
    learning_rate=2e-5,
    method="lora"  # Efficient fine-tuning
)
```

### 4. Evaluate

```python
metrics = builder.evaluate()
print(f"Accuracy: {metrics['accuracy']:.2%}")
print(f"F1 Score: {metrics['f1']:.3f}")
```

### 5. Use Your Model

```python
response = builder.generate("What is machine learning?")
print(response)
```

### 6. Export

```python
# Export to HuggingFace format
builder.export_model("./output", format="huggingface")

# Or export to ONNX
builder.export_model("./output", format="onnx")
```

---

## Complete Example

```python
from slm_builder import SLMBuilder

# Initialize
builder = SLMBuilder(
    project_name="qa-bot",
    base_model="google/flan-t5-small"
)

# Load data
builder.load_data_from_csv("qa_data.csv")

# Train
builder.train(
    epochs=3,
    learning_rate=2e-5,
    method="lora",
    batch_size=8
)

# Evaluate
metrics = builder.evaluate()
print(f"✅ Model trained! Accuracy: {metrics['accuracy']:.2%}")

# Test
response = builder.generate("How do I install Python?")
print(f"Bot: {response}")

# Export
builder.export_model("./qa-bot-model", format="huggingface")
```

---

## What's Next?

### Learn More About Features

- [Model Loading](../features/model-loading) - Load from different sources
- [Data Sources](../features/data-sources) - Use databases, APIs, and more
- [Training Methods](../features/training) - Configure training options
- [Evaluation Metrics](../features/evaluation) - Understand metrics

### Try More Examples

- [Getting Started Examples](../examples/getting-started) - More basic examples
- [Database Integration](../examples/database-integration) - Use databases
- [API Integration](../examples/api-integration) - Fetch from APIs

### Configure Your Project

- [Configuration Reference](../reference/configuration) - All options
- [Best Practices](../guides/best-practices) - Production tips

---

## Common Issues

### Import Error

```python
ModuleNotFoundError: No module named 'slm_builder'
```

**Solution**: Make sure you've installed the package:
```bash
pip install slm-builder
```

### CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solution**: Reduce batch size or use CPU:
```python
builder.train(batch_size=4, device="cpu")
```

### Model Download Fails

```
ConnectionError: Failed to download model
```

**Solution**: Check your internet connection or use a local model:
```python
builder = SLMBuilder(base_model="/path/to/local/model")
```

---

## Need Help?

- [GitHub Issues](https://github.com/isathish/slm/issues)
- [Discussions](https://github.com/isathish/slm/discussions)
- [Full Documentation](../)
