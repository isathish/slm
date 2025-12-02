---
layout: default
title: Installation
nav_order: 3
---

# Installation Guide

## Prerequisites

- Python 3.8 or higher
- pip or conda

## Installation Options

### Option 1: CPU-Only (Minimal)

For development or CPU-only inference:

```bash
pip install slm-builder
```

This installs the base dependencies without heavy ML libraries.

### Option 2: Full Installation (Recommended)

For training and GPU support:

```bash
pip install slm-builder[full]
```

This includes:
- PyTorch
- Transformers
- PEFT (for LoRA)
- Accelerate
- ONNX Runtime
- All training dependencies

### Option 3: Development Installation

For contributing to the project:

```bash
git clone https://github.com/isathish/slm.git
cd slm
pip install -e ".[full,dev]"
```

This installs in editable mode with development tools (pytest, black, flake8, etc.)

## Verifying Installation

```python
python -c "import slm_builder; print(slm_builder.__version__)"
```

Or test the CLI:

```bash
slm --version
```

## Optional Dependencies

### For GPU Training

If you have NVIDIA GPU:

```bash
# Install PyTorch with CUDA support first
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Then install SLM Builder
pip install slm-builder[full]
```

### For HTML Scraping

```bash
pip install beautifulsoup4
```

### For Database Connectors

```bash
# PostgreSQL
pip install psycopg2-binary

# MongoDB
pip install pymongo
```

## Troubleshooting

### Issue: `transformers` not found

**Solution:** Install full version:
```bash
pip install slm-builder[full]
```

### Issue: CUDA out of memory

**Solution:** Use CPU or reduce batch size:
```bash
slm build --source data.csv --device cpu
```

Or in config:
```yaml
training:
  batch_size: 4  # Reduce from default 8
```

### Issue: `peft` not available

**Solution:** Install full version with PEFT:
```bash
pip install slm-builder[full]
```

## Platform-Specific Notes

### macOS

```bash
# For M1/M2 Macs, use MPS acceleration
pip install slm-builder[full]
```

SLM Builder will automatically detect and use MPS when available.

### Windows

```bash
pip install slm-builder[full]
```

For best performance on Windows, use WSL2 with Ubuntu.

### Linux

```bash
# Standard installation works well
pip install slm-builder[full]
```

## Docker Installation

```bash
# CPU version
docker pull slmbuilder/slm-builder:latest

# GPU version (requires nvidia-docker)
docker pull slmbuilder/slm-builder:gpu
```

## Next Steps

After installation, check out:
- [Quick Start Guide](../README.md#quick-start)
- [Examples](../examples/)
- [API Documentation](./API.md)
