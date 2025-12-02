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

- **[Contributing Guide](CONTRIBUTING.md)** - How to contribute to the project
- **[Changelog](CHANGELOG.md)** - Version history and changes
- **[Release Guide](RELEASE_GUIDE.md)** - How to create new releases

## 📊 Project Status

### Implementation Reports

- **[Implementation Summary](IMPLEMENTATION_SUMMARY.md)** - Overview of implemented features
- **[Completion Report](COMPLETION_REPORT.md)** - Final completion status
- **[TODO Completion](TODO_COMPLETION.md)** - Completed tasks checklist
- **[Development Checklist](CHECKLIST.md)** - Feature implementation checklist
- **[Setup Complete](SETUP_COMPLETE.md)** - Setup and configuration status

## 🎯 Quick Navigation

### By Use Case

- **Building QA Models**: [Quick Reference](QUICK_REFERENCE.md#qa-model) → [Examples](EXAMPLES.md#qa-examples)
- **Database Integration**: [Additional Features](ADDITIONAL_FEATURES.md#database-loaders) → [Examples](EXAMPLES.md#database-examples)
- **API Data Loading**: [Additional Features](ADDITIONAL_FEATURES.md#api-loaders) → [Examples](EXAMPLES.md#api-examples)
- **Model Comparison**: [Additional Features](ADDITIONAL_FEATURES.md#model-comparison) → [Examples](EXAMPLES.md#comparison-examples)

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
2. Read [Contributing Guide](CONTRIBUTING.md)
3. Create custom implementations

## 📝 Documentation Format

This documentation is available in multiple formats:

- **GitHub Pages**: [https://isathish.github.io/slm/](https://isathish.github.io/slm/)
- **GitHub Wiki**: [https://github.com/isathish/slm/wiki](https://github.com/isathish/slm/wiki)
- **Source Files**: [docs/](https://github.com/isathish/slm/tree/main/docs)

## 🔄 Updates

Documentation is automatically updated:
- **GitHub Pages**: On every push to `main` branch
- **GitHub Wiki**: On every documentation change
- **Version**: Updated on each release

---

**Version**: 1.0.0  
**Last Updated**: December 2, 2025  
**Status**: ✅ Production Ready
