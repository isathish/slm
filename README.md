# SLM-Builder

[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/isathish/slm/releases)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-wiki-orange.svg)](https://github.com/isathish/slm/wiki)

**Build Small/Specialized Language Models from any dataset, source, or topic.**

SLM-Builder is an end-to-end Python toolkit for creating, training, and deploying specialized language models optimized for specific domains.

## 🚀 Quick Start

```bash
# Install
pip install slm-builder

# Build your first model
from slm_builder import SLMBuilder

builder = SLMBuilder(project_name="my-slm")
result = builder.build_from_csv("data.csv", task="qa", recipe="lora")
```

## ✨ Key Features

- 📥 **14 Data Sources**: CSV, JSONL, SQL, MongoDB, REST APIs, and more
- 🎯 **Multiple Tasks**: QA, classification, generation, instruction-tuning
- 🚀 **Easy Training**: Pre-configured recipes (LoRA, full fine-tuning)
- 💻 **CPU & GPU**: Optimized for both environments
- 📊 **Model Comparison**: Benchmark multiple models
- 🔬 **Advanced Metrics**: Perplexity, BLEU, ROUGE, F1
- ⚡ **Quantization**: 4-bit and 8-bit model compression
- 📦 **Export**: ONNX, TorchScript, HuggingFace format

## 📚 Documentation

Complete documentation is available in multiple formats:

- **[GitHub Wiki](https://github.com/isathish/slm/wiki)** - Comprehensive guides and tutorials
- **[GitHub Pages](https://isathish.github.io/slm/)** - Formatted documentation site
- **[docs/](./docs/)** - Source markdown files

### Quick Links

- [Installation Guide](./docs/INSTALLATION.md)
- [Quick Reference](./docs/QUICK_REFERENCE.md)
- [Core Features](./docs/FEATURES.md)
- [Additional Features](./docs/ADDITIONAL_FEATURES.md)
- [Examples](./docs/EXAMPLES.md)
- [Contributing](./docs/CONTRIBUTING.md)
- [Changelog](./docs/CHANGELOG.md)

## 📦 Installation

```bash
# Basic installation
pip install slm-builder

# With database support
pip install slm-builder[db]

# With all features
pip install slm-builder[all]
```

## 🎯 Examples

### Build from CSV

```python
from slm_builder import SLMBuilder

builder = SLMBuilder(project_name="customer-support")
result = builder.build_from_csv(
    path="support_qa.csv",
    task="qa",
    recipe="lora"
)
```

### Build from Database

```python
result = builder.build_from_database(
    query="SELECT question, answer FROM qa_table",
    connection_params={
        "dialect": "postgresql",
        "host": "localhost",
        "database": "mydb"
    },
    db_type="sql",
    task="qa"
)
```

### Build from API

```python
result = builder.build_from_api(
    base_url="https://api.example.com",
    endpoint="/data",
    auth={"type": "bearer", "token": "YOUR_TOKEN"},
    task="qa"
)
```

More examples in [examples/](./examples/) directory.

## 🔧 CLI Usage

```bash
# Build model
slm build --source data.csv --task qa --recipe lora

# Launch annotation UI
slm annotate --source data.csv --task qa

# Export model
slm export --model output/best --format onnx

# Serve model
slm serve --model output/best --port 8080
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](./docs/CONTRIBUTING.md).

## 📊 Version & Releases

Current version: **1.0.0**

We follow [Semantic Versioning](https://semver.org/):
- **Major**: Breaking changes
- **Minor**: New features (backward compatible)
- **Patch**: Bug fixes

See [CHANGELOG](./docs/CHANGELOG.md) for release history.

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Built with ❤️ using:
- [PyTorch](https://pytorch.org/)
- [Transformers](https://huggingface.co/transformers/)
- [PEFT](https://github.com/huggingface/peft)
- [Datasets](https://huggingface.co/docs/datasets/)

---

**[Documentation](https://github.com/isathish/slm/wiki)** | **[Examples](./examples/)** | **[Issues](https://github.com/isathish/slm/issues)** | **[Releases](https://github.com/isathish/slm/releases)**
