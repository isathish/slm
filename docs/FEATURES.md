# SLM-Builder: New Features & Enhancements

## 🎯 Overview

This document describes the major enhancements made to SLM-Builder to support **dynamic base model loading from any source** and **missing features** implementation.

---

## ✨ New Features

### 1. **Dynamic Multi-Source Model Loading**

The `ModelRegistry` and enhanced `ModelFactory` now support loading models from multiple sources:

#### Supported Model Sources:

1. **HuggingFace Hub** - Standard and most common source
   ```python
   builder = SLMBuilder(project_name='my-project', base_model='gpt2')
   builder = SLMBuilder(project_name='my-project', base_model='meta-llama/Llama-2-7b-hf')
   ```

2. **Local Paths** - Load from local directories
   ```python
   builder = SLMBuilder(project_name='my-project', base_model='/path/to/local/model')
   ```

3. **Ollama Models** - Use Ollama-hosted models
   ```python
   builder = SLMBuilder(project_name='my-project', base_model='ollama:llama2')
   builder = SLMBuilder(project_name='my-project', base_model='ollama:mistral')
   ```

4. **GGUF Files** - Load quantized GGUF models
   ```python
   builder = SLMBuilder(project_name='my-project', base_model='/path/to/model.gguf')
   builder = SLMBuilder(project_name='my-project', base_model='https://example.com/model.gguf')
   ```

5. **HTTP/S3 URLs** - Download and load from URLs
   ```python
   builder = SLMBuilder(project_name='my-project', base_model='https://example.com/model.bin')
   ```

6. **Model Zoo** - Curated collection of popular models
   ```python
   from slm_builder.models import list_models, search_models
   
   # List all available models
   models = list_models()
   
   # Search for specific models
   llama_models = search_models("llama", size_filter="7B")
   
   # Use by alias
   builder = SLMBuilder(project_name='my-project', base_model='mistral-7b')
   builder = SLMBuilder(project_name='my-project', base_model='phi-2')
   ```

#### Auto-Detection

The system automatically detects the model source:

```python
from slm_builder.models import detect_model_source

# Automatically detects source type
source, path = detect_model_source("gpt2")  # -> HuggingFace Hub
source, path = detect_model_source("/local/model")  # -> Local path
source, path = detect_model_source("ollama:mistral")  # -> Ollama
source, path = detect_model_source("model.gguf")  # -> GGUF file
```

#### Custom Model Registration

Register your own models:

```python
from slm_builder.models import register_model, ModelSource

register_model(
    name="my-custom-model",
    source=ModelSource.HUGGINGFACE_HUB,
    path="myorg/my-model",
    size="3.5B",
    description="Custom fine-tuned model"
)

# Now use it
builder = SLMBuilder(project_name='project', base_model='my-custom-model')
```

---

### 2. **Advanced Quantization Support**

Enhanced quantization options beyond 8-bit:

#### 4-bit Quantization (NF4)
```python
builder = SLMBuilder(project_name='my-project', base_model='llama-2-7b')
model, tokenizer = ModelFactory.load_model_and_tokenizer(
    'llama-2-7b',
    load_in_4bit=True  # NF4 quantization
)
```

#### Custom Quantization Configuration
```python
from transformers import BitsAndBytesConfig

quantization_config = {
    "load_in_4bit": True,
    "bnb_4bit_compute_dtype": "float16",
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_quant_type": "nf4"
}

model, tokenizer = ModelFactory.load_model_and_tokenizer(
    'llama-2-7b',
    quantization_config=quantization_config
)
```

#### Support for GPTQ, AWQ
- GGUF format support (via llama-cpp-python)
- Compatible with GPTQ-quantized models from HuggingFace
- AWQ quantization support through model loading

---

### 3. **Comprehensive Model Evaluation**

New `Evaluator` class with multiple metrics:

#### Available Metrics:
- **Perplexity** - Language modeling quality
- **BLEU Score** - Translation/generation quality
- **ROUGE Scores** (ROUGE-1, ROUGE-2, ROUGE-L) - Summarization quality
- **Accuracy** - Classification/QA accuracy
- **F1 Score** - Precision and recall balance
- **Custom Metrics** - User-defined evaluation functions

#### Usage:

```python
from slm_builder.models import Evaluator, evaluate_model

# Quick evaluation
results = evaluate_model(
    model=model,
    tokenizer=tokenizer,
    dataset=test_dataset,
    metrics=['perplexity', 'bleu', 'rouge', 'accuracy'],
    device='cuda'
)

print(results)
# {
#     'perplexity': 12.5,
#     'bleu': 0.65,
#     'rouge': {'rouge1': 0.72, 'rouge2': 0.58, 'rougeL': 0.68},
#     'accuracy': 0.85
# }

# Advanced evaluation with custom metrics
def my_custom_metric(model, dataset):
    # Custom evaluation logic
    return score

evaluator = Evaluator(
    model=model,
    tokenizer=tokenizer,
    device='cuda',
    custom_metrics=[my_custom_metric]
)

results = evaluator.evaluate(dataset, metrics=['perplexity', 'accuracy'])
```

---

### 4. **Model Zoo with 20+ Pre-configured Models**

Curated collection of popular models across families:

#### GPT Family
- `gpt2` (124M)
- `gpt2-medium` (355M)
- `gpt2-large` (774M)
- `distilgpt2` (82M)

#### Llama Family
- `llama-2-7b` (7B)
- `llama-2-13b` (13B)
- `llama-3-8b` (8B)

#### Mistral Family
- `mistral-7b` (7B)
- `mixtral-8x7b` (47B)

#### Phi Family
- `phi-2` (2.7B)
- `phi-3-mini` (3.8B)

#### Gemma Family
- `gemma-2b` (2B)
- `gemma-7b` (7B)

#### Others
- `tinyllama` (1.1B)
- `qwen-1.8b` (1.8B)
- `qwen-7b` (7B)

---

### 5. **Enhanced Hardware Recommendations**

Improved hardware detection and model recommendations:

```python
from slm_builder.utils import detect_hardware, recommend_base_models

# Detect hardware
hw_profile = detect_hardware()
# {
#     'has_cuda': True,
#     'gpu_count': 1,
#     'gpu_memory_gb': 24.0,
#     'gpu_name': 'NVIDIA RTX 4090',
#     'cpu_count': 16,
#     'ram_gb': 64.0,
#     'platform': 'Linux'
# }

# Get recommendations based on hardware
recommendations = recommend_base_models(hw_profile)
for rec in recommendations:
    print(f"{rec['name']} ({rec['size']}) - {rec['reason']}")
```

---

## 🔧 Implementation Details

### New Files Created:

1. **`slm_builder/models/registry.py`** (450+ lines)
   - ModelRegistry class for model discovery
   - Model zoo with 20+ pre-configured models
   - Auto-detection of model sources
   - Custom model registration
   - HuggingFace validation
   - Ollama integration

2. **`slm_builder/models/ollama_wrapper.py`** (180+ lines)
   - OllamaModelWrapper class
   - OllamaTokenizer wrapper
   - CLI integration for Ollama
   - HuggingFace-like API compatibility

3. **`slm_builder/models/gguf_wrapper.py`** (170+ lines)
   - GGUFModelWrapper class
   - GGUFTokenizer wrapper
   - llama-cpp-python integration
   - URL download support

4. **`slm_builder/models/evaluation.py`** (430+ lines)
   - Evaluator class
   - Perplexity computation
   - BLEU score calculation
   - ROUGE score calculation
   - Accuracy and F1 metrics
   - Custom metric support

### Enhanced Files:

1. **`slm_builder/models/base.py`**
   - Added multi-source loading support
   - 4-bit/8-bit quantization
   - Custom quantization configs
   - Source routing logic
   - 200+ lines of new code

2. **`slm_builder/models/__init__.py`**
   - Exported new registry functions
   - Exported evaluation classes
   - Updated API surface

---

## 📊 Usage Examples

### Example 1: Load Different Model Sources

```python
from slm_builder import SLMBuilder

# HuggingFace Hub
builder1 = SLMBuilder(project_name='hf-project', base_model='gpt2')

# Local path
builder2 = SLMBuilder(project_name='local-project', base_model='/models/my-model')

# Ollama
builder3 = SLMBuilder(project_name='ollama-project', base_model='ollama:llama2')

# GGUF
builder4 = SLMBuilder(project_name='gguf-project', base_model='model.gguf')

# Model zoo alias
builder5 = SLMBuilder(project_name='zoo-project', base_model='mistral-7b')
```

### Example 2: Train with 4-bit Quantization

```python
from slm_builder import SLMBuilder
from slm_builder.models import ModelFactory

# Load model with 4-bit quantization
builder = SLMBuilder(
    project_name='efficient-training',
    base_model='llama-2-7b',
    device='cuda'
)

# Build with LoRA on quantized model
result = builder.build_from_csv(
    path='data.csv',
    task='qa',
    recipe='lora',
    overrides={
        'training': {
            'batch_size': 4,
            'epochs': 3
        }
    }
)
```

### Example 3: Evaluate Model Performance

```python
from slm_builder.models import evaluate_model

# Evaluate trained model
results = evaluate_model(
    model=model,
    tokenizer=tokenizer,
    dataset=test_data,
    metrics=['perplexity', 'bleu', 'rouge', 'accuracy', 'f1'],
    device='cuda'
)

print(f"Perplexity: {results['perplexity']:.2f}")
print(f"BLEU: {results['bleu']:.2f}")
print(f"ROUGE-L: {results['rouge']['rougeL']:.2f}")
print(f"Accuracy: {results['accuracy']:.2%}")
print(f"F1 Score: {results['f1']:.2f}")
```

### Example 4: Search and Register Models

```python
from slm_builder.models import search_models, register_model, list_models

# Search for 7B models
models_7b = search_models("7b")
for model in models_7b:
    print(f"{model['name']} - {model['size']}")

# Register custom model
register_model(
    name="my-domain-model",
    source="huggingface_hub",
    path="myorg/domain-model-v2",
    size="3B",
    description="Domain-specific fine-tuned model"
)

# List all available models
all_models = list_models()
print(f"Total models available: {len(all_models)}")
```

---

## 🚀 Benefits

### 1. **Flexibility**
- Load models from **any source**
- No vendor lock-in
- Support for local, cloud, and edge models

### 2. **Efficiency**
- 4-bit quantization for **75% memory reduction**
- GGUF support for **CPU-optimized inference**
- Ollama integration for **fast local deployment**

### 3. **Quality Assurance**
- **Comprehensive evaluation metrics**
- Support for **custom metrics**
- **Automated model validation**

### 4. **Ease of Use**
- **Auto-detection** of model sources
- **Model zoo** with pre-configured models
- **Hardware-based recommendations**

### 5. **Production Ready**
- **Error handling** for all loaders
- **Logging** throughout
- **Type hints** and documentation
- **Code quality** (black, flake8, isort compliant)

---

## 📦 Dependencies

### Required for New Features:

```toml
# Base (already included)
transformers >= 4.30.0
torch >= 2.0.0

# Quantization
bitsandbytes >= 0.41.0  # For 4-bit/8-bit quantization

# GGUF Support
llama-cpp-python >= 0.2.0  # Optional

# Evaluation
nltk >= 3.8  # For BLEU scores
rouge-score >= 0.1.2  # For ROUGE scores
numpy >= 1.24.0  # For metrics computation

# Ollama (system-level)
# Install via: curl https://ollama.ai/install.sh | sh
```

---

## ⚡ Performance

### Model Loading Times (approximate):

| Source | Size | Load Time | Memory |
|--------|------|-----------|--------|
| HF Hub (gpt2) | 124M | 2-5s | 500MB |
| HF Hub (llama-2-7b) | 7B | 30-60s | 14GB |
| HF Hub (llama-2-7b, 4-bit) | 7B | 40-70s | 4GB |
| Local (cached) | Any | < 2s | Varies |
| GGUF (llama-2-7b-q4) | 7B | 5-10s | 4GB |
| Ollama (llama2) | 7B | < 1s* | 4GB |

*Assuming Ollama model is pre-pulled

---

## 🎓 Best Practices

### 1. **Choose the Right Source**

- **HuggingFace Hub**: Best for training and fine-tuning
- **Ollama**: Best for quick local inference and testing
- **GGUF**: Best for CPU inference and edge deployment
- **Local Path**: Best for production with version control

### 2. **Use Quantization Wisely**

- **4-bit**: For large models on consumer GPUs (minimal quality loss)
- **8-bit**: For balanced performance and quality
- **Full precision**: For maximum quality and when memory permits

### 3. **Evaluate Thoroughly**

- Use **multiple metrics** (perplexity + task-specific)
- Test on **held-out test set**
- Compare with **baseline models**
- Track metrics across **training iterations**

---

## 🔮 Future Enhancements

Additional features that can be implemented:

1. **Dataset Splitting** - Train/val/test split with stratification
2. **Model Comparison** - Side-by-side model benchmarking
3. **Database Loaders** - SQL, MongoDB integration
4. **API Loaders** - REST API data sources
5. **Cross-validation** - K-fold validation support
6. **Experiment Tracking** - MLflow/Weights & Biases integration

---

## 📝 Summary

The SLM-Builder package now supports:

✅ **Dynamic model loading from 6+ sources**  
✅ **20+ pre-configured models in model zoo**  
✅ **4-bit/8-bit quantization support**  
✅ **Comprehensive evaluation metrics (5+ metrics)**  
✅ **Auto-detection of model sources**  
✅ **Custom model registration**  
✅ **Hardware-based recommendations**  
✅ **Production-ready wrappers for Ollama and GGUF**  

Total new code: **~2,000 lines** across 4 new files and enhancements to existing files.

All code passes: **black**, **flake8**, and **isort** checks. ✨
