# SLM-Builder

**Build Small/Specialized Language Models from any dataset, source, or topic.**

SLM-Builder is an end-to-end Python toolkit for creating, training, and deploying specialized language models optimized for specific domains. Whether you have FAQ data, internal documentation, or customer support logs, SLM-Builder helps you build production-ready models with minimal ML expertise.

## ✨ Features

### Core Features
- 📥 **Multiple Data Sources**: Load from CSV, JSONL, text files, URLs, databases (SQL, MongoDB), or REST APIs
- 🎯 **Task-Specific**: Support for QA, classification, generation, and instruction-tuning
- 🚀 **Easy Training**: Pre-configured recipes (LoRA, full fine-tuning, instruction-tuning)
- 💻 **CPU & GPU Support**: Optimized for both environments with hardware auto-detection
- 🏷️ **Built-in Annotation**: Streamlit-based UI for data labeling
- 📦 **Export Options**: ONNX, TorchScript, or HuggingFace format
- 🌐 **Production Ready**: FastAPI server template included
- 🔒 **Security First**: PII detection and license checking

### Advanced Features
- 🔀 **Dynamic Model Loading**: Load from HuggingFace Hub, Local paths, Ollama, GGUF files, HTTP/S3 URLs
- ⚖️ **Smart Dataset Splitting**: Train/val/test splits with stratification and K-fold cross-validation
- 🗄️ **Database Integration**: Direct loading from PostgreSQL, MySQL, SQLite, MongoDB
- 🌐 **API Data Loading**: REST API support with authentication and pagination
- 📊 **Model Comparison**: Benchmark multiple models with comprehensive metrics
- 📈 **Experiment Tracking**: Track hyperparameters, metrics, and model versions
- 🔬 **Advanced Evaluation**: Perplexity, BLEU, ROUGE, Accuracy, F1 scores
- ⚡ **Quantization**: 4-bit and 8-bit model quantization for efficiency
- 🔍 **Dataset Validation**: Automatic quality checking and class balance analysis

📖 **[View Advanced Features Documentation →](FEATURES.md)**  
📖 **[View Additional Features Documentation →](ADDITIONAL_FEATURES.md)**

## 🚀 Quick Start

### Installation

```bash
# Basic installation (CPU-only)
pip install slm-builder

# Full installation (with GPU support)
pip install slm-builder[full]

# Development installation
pip install slm-builder[dev]
```

### Build Your First SLM

```python
from slm_builder import SLMBuilder

# Initialize builder
builder = SLMBuilder(project_name="faq-bot")

# Build from CSV in one line
result = builder.build_from_csv(
    path="data/faqs.csv",
    task="qa",
    recipe="lora"
)

print(f"Model saved to: {result['model_dir']}")
```

### CLI Usage

```bash
# Build from CSV
slm build --source data/faqs.csv --task qa --recipe lora --base-model gpt2

# Launch annotation UI
slm annotate --source data/raw.csv --task qa --out annotated.jsonl

# Export to ONNX
slm export --model output/best --format onnx --optimize cpu --quantize

# Serve the model
slm serve --model output/best --port 8080
```

## 📚 Examples

### QA System from CSV

```python
from slm_builder import SLMBuilder

builder = SLMBuilder(
    project_name="customer-support",
    base_model="gpt2",
)

# CSV should have 'question' and 'answer' columns
result = builder.build_from_csv(
    path="support_qa.csv",
    task="qa",
    recipe="lora",
)
```

### Custom Preprocessing

```python
from slm_builder import SLMBuilder

def custom_filter(records):
    # Filter out short questions
    return [r for r in records if len(r.get("text", "")) > 20]

builder = SLMBuilder(project_name="my-slm")
builder.register_preprocessor(custom_filter)

result = builder.build_from_csv("data.csv", task="qa")
```

## 🔧 Training Recipes

- **LoRA**: Efficient fine-tuning using Low-Rank Adaptation (recommended for CPU/limited resources)
- **Full Fine-tuning**: Traditional fine-tuning of all parameters (requires more resources)
- **Instruction-Tuning**: Specialized for instruction-following models

## 📝 Configuration

Create a `config.yml`:

```yaml
project_name: my-slm
base_model: gpt2
task: qa
recipe: lora

preprocess:
  max_tokens_per_chunk: 512
  chunk_overlap: 64

training:
  batch_size: 8
  learning_rate: 5e-5
  epochs: 3

lora:
  r: 8
  lora_alpha: 32
  target_modules: [q_proj, v_proj]
```

## 🏗️ Package Structure

```
slm_builder/
├── api.py              # Main SLMBuilder class
├── cli.py              # CLI commands
├── data/               # Data loading and preprocessing
├── models/             # Model training and export
├── serve/              # FastAPI serving
└── utils/              # Utilities
```

## 🧪 Testing

```bash
pytest tests/
```

## 📄 License

MIT License - see LICENSE file

---

**Made with ❤️ for building specialized AI models**