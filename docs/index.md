---
layout: default
title: Home
nav_order: 1
description: "SLM Builder - Build, train, and deploy specialized language models"
permalink: /
---

# SLM Builder
{: .fs-9 }

A comprehensive Python toolkit for creating, training, and deploying specialized language models with support for multiple data sources, training methods, and deployment formats.
{: .fs-6 .fw-300 }

[Get Started](guides/quick-start){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 .mr-2 }
[View on GitHub](https://github.com/isathish/slm){: .btn .fs-5 .mb-4 .mb-md-0 }

---

{: .warning }
> **Version 1.0.0** - Production Ready

## Quick Start

```python
from slm_builder import SLMBuilder

# Create a QA model
builder = SLMBuilder(
    project_name="my-qa-model",
    base_model="google/flan-t5-small"
)

# Load data and train
builder.load_data_from_csv("qa_data.csv")
builder.train(epochs=3, learning_rate=2e-5)

# Evaluate and export
metrics = builder.evaluate()
builder.export_model("./output", format="huggingface")
```

---

## 📚 Documentation

### Getting Started
{: .text-delta }

| Guide | Description |
|:------|:------------|
| [**Installation**](INSTALLATION) | Complete installation instructions |
| [**Quick Start**](guides/quick-start) | Build your first model in 5 minutes |
| [**Basic Concepts**](guides/concepts) | Core concepts and architecture |
| [**Quick Reference**](QUICK_REFERENCE) | Cheat sheet for common tasks |

### Features
{: .text-delta }

| Feature | Description |
|:--------|:------------|
| [**Model Loading**](features/model-loading) | Load from HuggingFace, local, Ollama, GGUF |
| [**Data Sources**](features/data-sources) | 14 data sources: SQL, MongoDB, APIs, files |
| [**Training**](features/training) | LoRA, QLoRA, full fine-tuning |
| [**Evaluation**](features/evaluation) | Comprehensive metrics and benchmarks |
| [**Quantization**](features/quantization) | 4-bit and 8-bit compression |
| [**Model Comparison**](features/model-comparison) | Benchmark multiple models |
| [**Export**](features/export) | ONNX, TorchScript, HuggingFace |

### Examples
{: .text-delta }

| Category | Examples |
|:---------|:---------|
| [**Getting Started**](examples/getting-started) | Basic usage patterns |
| [**Database Integration**](examples/database-integration) | PostgreSQL, MySQL, MongoDB, Redis |
| [**API Integration**](examples/api-integration) | REST APIs, authenticated endpoints |
| [**Advanced**](examples/advanced-examples) | Production use cases |

### Reference
{: .text-delta }

| Document | Description |
|:---------|:------------|
| [**API Reference**](reference/api) | Complete API documentation |
| [**Configuration**](reference/configuration) | All configuration options |
| [**CLI Reference**](reference/cli) | Command-line interface |

### Development
{: .text-delta }

| Guide | Description |
|:------|:------------|
| [**Developer Guide**](DEVELOPER_GUIDE) | Contributing and development |
| [**Changelog**](CHANGELOG) | Version history |

---

## ✨ Key Features

<div class="code-example" markdown="1">

### 🎯 **14 Data Sources**
CSV, JSON, JSONL, Parquet, SQL (PostgreSQL, MySQL, SQLite), MongoDB, Redis, Elasticsearch, REST APIs, and more.

### 🚀 **Flexible Training**
LoRA fine-tuning, QLoRA (4-bit), full fine-tuning, custom configurations.

### 📊 **Advanced Evaluation**
Perplexity, BLEU, ROUGE, F1, accuracy, and task-specific metrics.

### ⚡ **Optimized Performance**
Quantization, GPU/CPU support, batch processing, efficient inference.

### 📦 **Multiple Exports**
HuggingFace Hub, ONNX, TorchScript, TensorFlow, local deployment.

### 🔧 **Production Ready**
MLflow tracking, dataset splitting, model comparison, comprehensive logging.

</div>

---

## 🎯 Use Cases

### Customer Support Bot
Build an intelligent QA bot from support tickets:
```python
builder.load_from_database(
    connection_params={"dialect": "postgresql"},
    query="SELECT question, answer FROM support_tickets"
)
```

### Document Intelligence  
Extract insights from document collections:
```python
builder.load_from_elasticsearch(
    host="localhost:9200",
    index="documents"
)
```

### API-Driven Models
Train on data from REST APIs:
```python
builder.load_from_api(
    endpoint="https://api.example.com/data",
    headers={"Authorization": "Bearer TOKEN"}
)
```

---

## 📦 Installation

```bash
# Basic installation
pip install slm-builder

# With database support
pip install slm-builder[db]

# With all features
pip install slm-builder[all]
```

See the [Installation Guide](INSTALLATION) for detailed instructions.

---

## 🤝 Community & Support

### Get Help
- [GitHub Issues](https://github.com/isathish/slm/issues) - Report bugs
- [Discussions](https://github.com/isathish/slm/discussions) - Ask questions
- [Documentation](https://isathish.github.io/slm/) - Guides and references

### Contribute
We welcome contributions! See our [Developer Guide](DEVELOPER_GUIDE) to get started.

---

## 📊 Project Status

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-passing-success)

**Last Updated**: December 2, 2025

---

## License

SLM Builder is released under the MIT License. See [LICENSE](https://github.com/isathish/slm/blob/main/LICENSE) for details.
