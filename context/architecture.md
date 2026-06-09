# THEMIS — System Architecture

## Overview

THEMIS is a parametric legal intelligence engine for Indian law. Unlike RAG-based systems (HECTOR), THEMIS encodes legal knowledge directly into model weights via LoRA fine-tuning. The system is designed for offline CLI deployment with no runtime document retrieval.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     USER LAYER                          │
│  CLI (Typer + Rich) → themis ask / themis chat          │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│                   INFERENCE LAYER                       │
│  infer.py → Model Loader → LoRA Adapter → Generator     │
│  - Loads Mistral 7B Instruct v0.3 base model           │
│  - Attaches LoRA adapter (rank=16, alpha=32)            │
│  - Applies chat template + system prompt                │
│  - Generates response with citations                    │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│                  CONFIGURATION LAYER                    │
│  config.py → ModelPath, GenerationParams, DeviceConfig  │
│  - Paths, hyperparameters, device selection              │
│  - Generation: temperature, top_p, max_new_tokens        │
└─────────────────────────────────────────────────────────┘
```

---

## Component Design

### 1. CLI Entry Point (`cli.py`)

**Framework:** Typer + Rich

**Commands:**
- `themis ask "<question>"` — Single-shot Q&A, returns formatted response
- `themis chat` — Interactive multi-turn session with conversation history
- `themis eval` — Run 50-question evaluation harness
- `themis info` — Model metadata (training date, dataset size, LoRA config)
- `themis version` — Version string and config summary

**Responsibilities:**
- Parse user input
- Display Rich-formatted output (tables, panels, progress bars)
- Manage conversation history for `chat` mode
- Handle exit signals gracefully

### 2. Inference Engine (`infer.py`)

**Responsibilities:**
- Load base model (Mistral 7B Instruct v0.3) from local path or HuggingFace Hub
- Apply LoRA adapter weights on top of base model
- Construct system prompt with legal domain instructions
- Apply chat template formatting
- Generate response with controlled parameters
- Post-process output to extract citations and format

**Key Functions:**
```python
def load_model(model_path: str, lora_path: str) -> PeftModelForCausalLM
def generate_response(query: str, history: list[dict], config: Config) -> str
def apply_system_prompt(query: str) -> str
```

**Model Loading Strategy:**
1. Load base model in 4-bit quantization (QLoRA-compatible)
2. Attach LoRA adapter via PEFT
3. Set to eval mode, disable gradients
4. Move to available device (CUDA if available, else CPU)

### 3. Configuration (`config.py`)

**Dataclass: `Config`**
```python
@dataclass
class Config:
    # Model
    base_model_path: str = "mistralai/Mistral-7B-Instruct-v0.3"
    lora_adapter_path: str = "./model/themis-lora"
    
    # Generation
    temperature: float = 0.3
    top_p: float = 0.9
    max_new_tokens: int = 1024
    repetition_penalty: float = 1.1
    
    # Device
    device: str = "auto"  # auto | cuda | cpu
    load_in_4bit: bool = True
    
    # System
    system_prompt: str = "You are THEMIS, a legal intelligence engine..."
    disclaimer: str = "This is legal orientation, not legal advice..."
```

### 4. Evaluation Harness (`eval/`)

**run_eval.py:**
- Loads `eval_set.json` (50 ground-truth Q&A pairs)
- Runs each question through the model
- Collects responses for metric computation

**metrics.py:**
- `citation_accuracy(predicted, ground_truth)` — Check if correct section numbers are cited
- `refusal_rate(query, response)` — Verify model refuses out-of-scope queries
- `rouge_l(predicted, ground_truth)` — Lexical overlap score
- `hallucination_check(response)` — Manual/automated check for fabricated sections

**eval_set.json:**
- 50 held-out questions across 5 categories
- Ground truth answers with correct citations
- Category labels for per-category analysis

---

## Data Flow

### Inference Flow
```
User Input
    │
    ▼
System Prompt Injection
    │  "You are THEMIS... Answer based on BNS/IPC/BNSS..."
    ▼
Chat Template Formatting
    │  [INST] {system_prompt}\n{user_query} [/INST]
    ▼
Model Forward Pass
    │  LoRA-augmented Mistral 7B generation
    ▼
Post-Processing
    │  Extract citations, add disclaimer, format output
    ▼
Rich Terminal Output
    │  Panel with legal orientation + citations footer
    ▼
Response to User
```

### Training Data Flow
```
Raw Sources
    │
    ├─── India Code (Bare Acts) ──┐
    ├─── Indian Kanoon (Judgments)┤
    └─── Groq API / Templates ─┘
                                  │
                                  ▼
                         Data Preprocessing
                         │  parse.py → clean, dedup, format
                         │  generate.py → Q&A pair generation
                         ▼
                    dataset.json (Alpaca format)
                         │
                         ▼
                    Training Pipeline
                    │  Unsloth + LoRA on Kaggle T4
                    │  3 epochs, batch_size=2, lr=2e-4
                    ▼
                    LoRA Adapter Weights
                    │  Saved to HuggingFace Hub
                    │  danieldeshmukh/themis-mistral-7b-lora
                    ▼
                    Inference Engine
                    │  Load base + adapter → Serve CLI
                    ▼
                    User
```

---

## File Structure

```
themis/
├── cli.py                  # Rich-powered CLI entry point
├── infer.py                # Model loading and inference engine
├── config.py               # Model path, generation params, device config
├── eval/
│   ├── run_eval.py         # Evaluation harness (50 held-out questions)
│   ├── metrics.py          # Citation accuracy, refusal rate, ROUGE-L
│   └── eval_set.json       # Ground truth evaluation dataset
├── data/
│   ├── scraper/
│   │   ├── kanoon.py       # Indian Kanoon judgment scraper
│   │   └── indiacode.py    # India Code Bare Acts parser
│   ├── synthetic/
│   │   └── generate.py     # Groq/template Q&A pair generation
│   ├── preprocess.py       # Cleaning, deduplication, formatting
│   └── dataset.json        # Final Alpaca-format training dataset
├── training/
│   ├── finetune.py         # Unsloth + LoRA training script (Kaggle)
│   ├── config.yaml         # LoRA rank, alpha, target modules, epochs
│   └── push_to_hub.py      # HuggingFace Hub upload
├── model/                  # Local model weights (gitignored)
├── tests/
│   └── test_infer.py       # Unit tests for inference pipeline
├── context/                # Project context docs (this directory)
│   ├── architecture.md
│   ├── roadmap.md
│   ├── progress-tracker.md
│   ├── security-audit.md
│   ├── tech-stack.md
│   └── data-pipeline.md
└── pyproject.toml          # Package config (pip install themis-law)
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| LoRA over full fine-tuning | Free T4 GPU (16GB VRAM) cannot fit full 7B fine-tuning |
| Mistral 7B as base | Strong instruction following, 7B is optimal for single T4 |
| CLI over web UI | Zero infrastructure cost, offline-capable, simpler deployment |
| Alpaca format | Standard SFT format, compatible with Unsloth/HuggingFace |
| 4-bit quantization | Reduces VRAM usage from ~14GB to ~5GB, leaves room for LoRA |
| System prompt engineering | Guides model to cite specific sections, refuse out-of-scope |
| Separate eval harness | Offline evaluation, no runtime dependency on eval infrastructure |

---

## Relationship to HECTOR

In v2.0, a unified router will dispatch queries:
- **THEMIS** → Citizen Q&A, quick legal orientation
- **HECTOR** → Deep legal research requiring source-level citations

```
User Query → Query Classifier → THEMIS or HECTOR
```

THEMIS handles simple statutory questions. HECTOR handles complex research needing document retrieval.

---

## Constraints

1. **No internet at inference time** — All knowledge must be in model weights
2. **Single GPU training** — T4 on Kaggle (16GB VRAM max)
3. **No case law in v1** — Only statutes and procedural law
4. **English only in v1** — Hindi support planned for v1.2
5. **No state-specific laws** — Stamp duty, tenancy, land records vary by state
