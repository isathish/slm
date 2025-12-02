---
layout: default
title: Home
nav_order: 1
---

# SLM Builder Documentation

Welcome to the comprehensive documentation for **SLM Builder** - an end-to-end Python toolkit for creating, training, and deploying specialized language models.

## 🚀 Getting Started

New to SLM Builder? Start here:

1. **[Installation Guide](INSTALLATION.md)** - Get SLM Builder up and running
2. **[Quick Reference](QUICK_REFERENCE.md)** - Cheat sheet for common tasks
3. **[Examples](EXAMPLES.md)** - Learn by example

## 📚 Core Documentation

### Features

- **[Core Features](FEATURES.md)** - Dynamic model loading, quantization, evaluation
- **[Additional Features](ADDITIONAL_FEATURES.md)** - Database loading, API integration, model comparison

### Development

- **[Developer Guide](DEVELOPER_GUIDE.md)** - Contributing, development setup, release process
- **[Changelog](CHANGELOG.md)** - Version history and changes

## 🎯 Quick Navigation

### By Use Case

- **Building QA Models**: [Quick Reference](QUICK_REFERENCE.md#qa-model) → [Examples](EXAMPLES.md#qa-examples)
- **Database Integration**: [Additional Features](ADDITIONAL_FEATURES.md#database-loaders) → [Examples](EXAMPLES.md#database-examples)
- **API Data Loading**: [Additional Features](ADDITIONAL_FEATURES.md#api-loaders) → [Examples](EXAMPLES.md#api-examples)
- **Model Comparison**: [Additional Features](ADDITIONAL_FEATURES.md#model-comparison) → [Examples](EXAMPLES.md#comparison-examples)

### For Developers

- **Contributing**: [Developer Guide](DEVELOPER_GUIDE.md#contributing) - How to contribute
- **Development Setup**: [Developer Guide](DEVELOPER_GUIDE.md#development-setup) - Set up your dev environment
- **Release Process**: [Developer Guide](DEVELOPER_GUIDE.md#release-process) - Creating releases
- **Testing**: [Developer Guide](DEVELOPER_GUIDE.md#testing) - Running and writing tests

### By Feature

- **Data Sources**: [Core Features](FEATURES.md#data-sources)
- **Model Loading**: [Core Features](FEATURES.md#model-loading)
- **Training Methods**: [Core Features](FEATURES.md#training)
- **Evaluation Metrics**: [Core Features](FEATURES.md#evaluation)
- **Export Formats**: [Core Features](FEATURES.md#export)

## 🔧 Advanced Topics

- **Dataset Splitting & Validation**: [Additional Features](ADDITIONAL_FEATURES.md#dataset-splitting)
- **Experiment Tracking**: [Additional Features](ADDITIONAL_FEATURES.md#experiment-tracking)
- **Quantization**: [Core Features](FEATURES.md#quantization)
- **Custom Preprocessing**: [Examples](EXAMPLES.md#custom-preprocessing)

## 🤝 Community

- **[GitHub Repository](https://github.com/isathish/slm)**
- **[Issue Tracker](https://github.com/isathish/slm/issues)**
- **[Discussions](https://github.com/isathish/slm/discussions)**

## 📦 Installation

```bash
# Basic installation
pip install slm-builder

# With database support
pip install slm-builder[db]

# With all features
pip install slm-builder[all]
```

See [Installation Guide](INSTALLATION.md) for detailed instructions.

## ✨ Key Features

- 📥 **14 Data Sources**: CSV, JSONL, SQL, MongoDB, REST APIs, and more
- 🎯 **Multiple Tasks**: QA, classification, generation, instruction-tuning
- 🚀 **Easy Training**: Pre-configured recipes (LoRA, full fine-tuning)
- 💻 **CPU & GPU Support**: Optimized for both environments
- 📊 **Model Comparison**: Benchmark multiple models
- 🔬 **Advanced Metrics**: Perplexity, BLEU, ROUGE, F1
- ⚡ **Quantization**: 4-bit and 8-bit compression
- 📦 **Multiple Exports**: ONNX, TorchScript, HuggingFace

## 🎓 Learning Path

### Beginner
1. Read [Installation Guide](INSTALLATION.md)
2. Try [Quick Reference](QUICK_REFERENCE.md) examples
3. Explore [Basic Examples](EXAMPLES.md#basic-examples)

### Intermediate
1. Learn about [Core Features](FEATURES.md)
2. Try [Database Loading](EXAMPLES.md#database-examples)
3. Experiment with [Model Comparison](EXAMPLES.md#comparison-examples)

### Advanced
1. Study [Additional Features](ADDITIONAL_FEATURES.md)
2. Read [Developer Guide](DEVELOPER_GUIDE.md)
3. Create custom implementations

### Contributing
1. Review [Developer Guide](DEVELOPER_GUIDE.md#contributing)
2. Set up [Development Environment](DEVELOPER_GUIDE.md#development-setup)
3. Understand [Release Process](DEVELOPER_GUIDE.md#release-process)

## 📝 Documentation

This documentation is automatically published to GitHub Pages:

- **Live Site**: [https://isathish.github.io/slm/](https://isathish.github.io/slm/)
- **Source Files**: [docs/](https://github.com/isathish/slm/tree/main/docs)

Documentation is automatically updated on every push to the `main` branch.

---

**Version**: 1.0.0  
**Last Updated**: December 2, 2025  
**Status**: ✅ Production Ready
