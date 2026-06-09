# THEMIS — Technology Stack

## Overview

THEMIS is built on a minimal, cost-effective stack optimized for free compute (Kaggle T4 GPU) and offline CLI deployment.

---

## Technology Matrix

| Layer | Technology | Version | Purpose | Why This Choice |
|-------|-----------|---------|---------|-----------------|
| **Base Model** | Mistral 7B Instruct v0.3 | v0.3 | Foundation model | Strong instruction following, 7B optimal for T4 |
| **Fine-tuning** | LoRA (Low-Rank Adaptation) | — | Parameter-efficient training | Fits in 16GB VRAM, 2x faster with Unsloth |
| **Training Framework** | Unsloth | latest | LoRA training | 2x faster, 60% less memory, free |
| **Training Platform** | Kaggle | — | GPU compute | Free T4 GPU (16GB VRAM) |
| **ML Framework** | PyTorch | 2.x | Tensor operations | Industry standard, CUDA support |
| **Model Hub** | HuggingFace Transformers | 4.x | Model loading/inference | Standard for LLM deployment |
| **PEFT** | HuggingFace PEFT | latest | LoRA adapter loading | Official LoRA implementation |
| **Quantization** | bitsandbytes | 4-bit | Model compression | Reduces VRAM from 14GB to 5GB |
| **Dataset Format** | Alpaca JSON | — | Instruction tuning | Standard SFT format, widely supported |
| **CLI Framework** | Typer | latest | Command-line interface | Fast, type-safe, auto-generated help |
| **Terminal UI** | Rich | latest | Formatted output | Beautiful panels, tables, progress bars |
| **Scraping** | requests + BeautifulSoup | — | Web scraping | Simple, reliable for static sites |
| **Synthetic Data** | Claude API | — | Q&A pair generation | High-quality legal text generation |
| **Evaluation** | custom + rouge-score | — | Model evaluation | Citation accuracy + ROUGE-L |
| **Packaging** | pyproject.toml | — | Package distribution | Modern Python packaging standard |

---

## Version Constraints

### Python
- **Required:** Python 3.11+
- **Reason:** Type hints, match statements, performance improvements

### PyTorch
- **Required:** PyTorch 2.x with CUDA support
- **Reason:** bitsandbytes 4-bit quantization requires PyTorch 2.x

### HuggingFace Stack
- **transformers:** 4.x (recent enough for Mistral support)
- **peft:** latest (LoRA adapter support)
- **bitsandbytes:** 0.41+ (4-bit quantization)

### Unsloth
- **Required:** Latest version from GitHub
- **Reason:** Most recent optimizations and Mistral support

---

## Hardware Requirements

### Training (Kaggle)
| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA T4 (16GB VRAM) |
| RAM | 16GB system RAM |
| Storage | ~20GB for model + data |
| Training Time | ~2-3 hours for 3 epochs |

### Inference (User Machine)
| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 8GB | 16GB+ |
| GPU | None (CPU works) | NVIDIA GPU with 6GB+ VRAM |
| Storage | 5GB (model weights) | 10GB (with cache) |
| OS | Windows, Linux, macOS | Any with Python 3.11+ |

---

## Dependency Management

### pyproject.toml (Planned)

```toml
[project]
name = "themis-law"
version = "1.0.0"
requires-python = ">=3.11"
dependencies = [
    "transformers>=4.35.0",
    "peft>=0.6.0",
    "bitsandbytes>=0.41.0",
    "torch>=2.0.0",
    "typer>=0.9.0",
    "rich>=13.0.0",
    "requests>=2.31.0",
    "beautifulsoup4>=4.12.0",
]

[project.optional-dependencies]
training = [
    "unsloth",
    "rouge-score",
]
dev = [
    "pytest",
    "ruff",
]
```

---

## Development Tools

| Tool | Purpose | When Used |
|------|---------|-----------|
| ruff | Python linter | Pre-commit, CI |
| pytest | Test framework | Unit tests |
| mypy | Type checker | Optional, for strict typing |
| black | Code formatter | Consistent style (or ruff format) |

---

## Training Pipeline Details

### Unsloth Configuration

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="mistralai/Mistral-7B-Instruct-v0.3",
    max_seq_length=2048,
    dtype=None,  # Auto-detect
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing=True,
)
```

### LoRA Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| r (rank) | 16 | Balance between capacity and VRAM |
| lora_alpha | 32 | Scaling factor (2x rank) |
| target_modules | q,k,v,o_proj | Attention layers only |
| lora_dropout | 0.05 | Prevent overfitting |
| bias | none | Simplify training |
| task_type | CAUSAL_LM | Language modeling task |

### Training Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| epochs | 3 | Enough for convergence, not overfit |
| batch_size | 2 | T4 VRAM constraint |
| gradient_accumulation | 4 | Effective batch size = 8 |
| learning_rate | 2e-4 | Standard for LoRA |
| warmup_ratio | 0.03 | Stable early training |
| lr_scheduler | cosine | Smooth decay |
| max_seq_length | 2048 | Fits legal responses |
| fp16 | true | T4 compatible, faster training |

---

## Inference Pipeline Details

### Model Loading

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import bitsandbytes as bnb

# Load base model in 4-bit
base_model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.3",
    quantization_config=bnb BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    ),
    device_map="auto",
)

# Attach LoRA adapter
model = PeftModel.from_pretrained(base_model, "./model/themis-lora")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
```

### Generation Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| temperature | 0.3 | Low creativity, high factual accuracy |
| top_p | 0.9 | Nucleus sampling |
| max_new_tokens | 1024 | Enough for detailed legal responses |
| repetition_penalty | 1.1 | Prevent repetitive outputs |

---

## Alternative Technologies Considered

| Alternative | Why Not Chosen |
|-------------|----------------|
| Llama 2 7B | Mistral has better instruction following at 7B |
| Full fine-tuning | Requires 40GB+ VRAM, not feasible on T4 |
| QLoRA | bitsandbytes 4-bit is sufficient, QLoRA adds complexity |
| Flask/FastAPI | CLI chosen for offline-first, zero-infrastructure goal |
| LangChain | Overkill for simple inference pipeline |
| vLLM | Optimized for serving, not needed for single-user CLI |

---

## Platform-Specific Notes

### Kaggle
- Unsloth pre-installed in most notebooks
- T4 GPU available with free account
- Session limit: ~6 hours (enough for 3 epochs)
- Save adapter to output directory, download manually

### Windows
- bitsandbytes requires WSL2 for CUDA support
- Alternative: Use CPU-only mode (slower but works)
- Path handling: Use pathlib for cross-platform compatibility

### macOS
- No CUDA support, CPU-only inference
- Apple Silicon (M1/M2) works with MPS backend
- Slower but functional

---

## Future Technology Considerations

| Version | Technology Change | Reason |
|---------|-------------------|--------|
| v1.2 | Hindi tokenizer | Multilingual support for bilingual fine-tune |
| v2.0 | FastAPI | Web API for HECTOR integration |
| v2.1 | Next.js | Web UI frontend |
| v3.0 | vLLM or TGI | Production-grade inference serving |
