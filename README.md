<div align="center">

```
████████╗██╗  ██╗███████╗███╗   ███╗██╗███████╗
╚══██╔══╝██║  ██║██╔════╝████╗ ████║██║██╔════╝
   ██║   ███████║█████╗  ██╔████╔██║██║███████╗
   ██║   ██╔══██║██╔══╝  ██║╚██╔╝██║██║╚════██║
   ██║   ██║  ██║███████╗██║ ╚═╝ ██║██║███████║
   ╚═╝   ╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝╚═╝╚══════╝
```

**The Parametric Legal Intelligence Engine for Indian Law**

<p>
  <img src="https://img.shields.io/badge/model-Mistral%207B%20LoRA-blueviolet" />
  <img src="https://img.shields.io/badge/domain-Indian%20Law%20%7C%20BNS%20%7C%20IPC-crimson" />
  <img src="https://img.shields.io/badge/python-3.11+-brightgreen" />
  <img src="https://img.shields.io/badge/fine--tuning-LoRA%20%7C%20Unsloth-orange" />
  <img src="https://img.shields.io/badge/license-MIT-lightgrey" />
</p>

*"Not retrieval. Not lookup. Pure legal reasoning, baked into weights."*

</div>

---

## What is THEMIS?

THEMIS is a domain-specific large language model fine-tuned on Indian statutory law. It is **not** a retrieval system, a search engine, or a chatbot wrapper around an existing API. It is a **parametric knowledge model** — one where legal understanding of the Bharatiya Nyaya Sanhita (BNS), the Indian Penal Code (IPC), the Bharatiya Nagarik Suraksha Sanhita (BNSS), and allied statutes has been baked directly into the model weights through supervised fine-tuning.

Where HECTOR retrieves — THEMIS reasons.

---

## The Core Difference: THEMIS vs HECTOR

| Dimension | HECTOR | THEMIS |
|-----------|--------|--------|
| Architecture | RAG (Retrieval-Augmented Generation) | Parametric Fine-Tuning (LoRA) |
| Knowledge source | External vector database (Qdrant) | Model weights |
| Needs documents at runtime | Yes — ingests Bare Acts and commentaries | No — knowledge is in the model |
| Hallucination control | Chain-of-Verification + source grounding | Training data quality + eval harness |
| Response style | Structured legal research output | Conversational plain-language with citations |
| Best for | Deep legal research, IPC↔BNS mapping | Citizen Q&A, quick legal orientation |
| Deployment | FastAPI + Next.js frontend | CLI (offline capable) |

They are complementary systems. HECTOR is the researcher. THEMIS is the explainer.

---

## What THEMIS Does

- Answers questions about Indian law in plain, citizen-readable language
- Cites specific BNS / IPC / BNSS / Consumer Protection Act sections in every response
- Maps legacy IPC sections to their BNS equivalents where applicable
- Handles criminal law, civil disputes, property rights, consumer rights, RTI, and procedural queries
- Runs entirely offline via CLI — no API calls, no internet dependency after setup
- Refuses to speculate — trained to say "consult a lawyer" when the query is beyond statutory scope

---

## What THEMIS Does NOT Do

- **Not a lawyer substitute.** THEMIS provides legal orientation, not legal advice. Every response includes a disclaimer directing users to seek qualified counsel for their specific situation.
- **Not a retrieval system.** It does not search documents at runtime. It cannot cite page numbers or PDFs. That is HECTOR's job.
- **Not trained on case law.** THEMIS knows statutes and procedural law. Judgments, precedents, and High Court/Supreme Court rulings are outside its training scope in v1.
- **Not multilingual in v1.** English only. Hindi support is on the roadmap.
- **Not a general-purpose LLM.** Do not expect it to write code, summarize articles, or answer non-legal queries. It will decline.
- **Not accurate for state-specific laws.** Stamp duty, tenancy, land records — these vary by state and are not covered.

---

## Architecture

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
│   └── dataset.json        # Final Alpaca-format training dataset (1,939 pairs)
├── training/
│   ├── finetune.py         # Unsloth + LoRA training script (Kaggle)
│   ├── config.yaml         # LoRA rank, alpha, target modules, epochs
│   └── push_to_hub.py      # HuggingFace Hub upload
├── notebooks/
│   └── themis_inference.ipynb  # Kaggle inference notebook
├── model/                  # Local model weights (gitignored)
└── tests/
    └── test_infer.py       # Unit tests for inference pipeline
```

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Base Model | Mistral 7B Instruct v0.3 (4-bit) | Foundation — strong instruction following |
| Fine-tuning Method | LoRA rank=8, alpha=16 | Parameter-efficient training on free compute |
| Training Framework | Unsloth | 2x faster LoRA, fits in 16GB VRAM |
| Training Platform | Kaggle (free T4 GPU) | Zero cost fine-tuning |
| Dataset | 1,939 Alpaca-format Q&A pairs | BNS, BNSS, IPC, Consumer Law, RTI |
| Data Sources | Indian Kanoon + India Code + Synthetic | Scraping + generation pipeline |
| CLI Framework | Typer + Rich | Beautiful terminal interface |
| Inference | Transformers + PEFT + BitsAndBytes | 4-bit quantized loading + LoRA |
| Notebook | Kaggle T4 GPU | Interactive demo + evaluation |
| Model Hosting | HuggingFace Hub | Free public model hosting |

---

## CLI Interface

THEMIS runs as a Rich-powered terminal application. No browser, no server, no API key required at inference time.

### Installation

```bash
pip install themis-law
```

### Quick Start

**Option A — Kaggle (recommended):** Run the [THEMIS Inference Notebook](https://www.kaggle.com/datasets/daniel2503/themis-inference) on a free T4 GPU. No local setup needed.

**Option B — Local CLI:**
```bash
pip install themis-law
themis ask "What is BNS Section 118?"
themis chat
```

> **Note:** Requires a CUDA GPU for reasonable performance. The base model is ~4GB in 4-bit quantization.

### Commands

| Command | Description |
|---------|-------------|
| `themis ask "your question"` | Single-shot legal Q&A |
| `themis chat` | Interactive multi-turn session |

### Terminal Experience

```
████████╗██╗  ██╗███████╗███╗   ███╗██╗███████╗
╚══██╔══╝██║  ██║██╔════╝████╗ ████║██║██╔════╝
   ██║   ███████║█████╗  ██╔████╔██║██║███████╗
   ██║   ██╔══██║██╔══╝  ██║╚██╔╝██║██║╚════██║
   ██║   ██║  ██║███████╗██║ ╚═╝ ██║██║███████║
   ╚═╝   ╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝╚═╝╚══════╝

THEMIS v1.0.0  | Model: Daniel2503/themis-mistral-7b-lora
Domain: BNS 2023 | BNSS 2023 | IPC | Consumer Law | RTI Act

? Ask a legal question (or 'exit' to quit): If my employer hasn't paid salary for 2 months, what can I do?

Loading model...
THEMIS is thinking...

LEGAL ORIENTATION
============================================================

Non-payment of wages is actionable under multiple statutes in India:

[1] Payment of Wages Act, 1936 — Section 15
    File a complaint before the Authority appointed
    under this Act (typically a Labour Commissioner).
    Claim includes wages + compensation up to 10x.

[2] BNS 2023 — Section 316 (Criminal Breach of Trust)
    If deliberate withholding is proven, this may
    attract criminal liability on the employer.

[3] Industrial Disputes Act, 1947 — Section 33C
    Workmen can recover dues directly through the
    Labour Court without filing a civil suit.

DISCLAIMER: This is legal orientation, not legal advice.
Consult a qualified advocate for your specific situation.

Tokens: 24 in / 187 out | Device: cuda
```

---

## Dataset

### Sources

| Source | Type | Volume |
|--------|------|--------|
| India Code (indiacode.nic.in) | BNS, BNSS, BSA, Consumer Protection Act, RTI Act — full text | ~800 sections parsed |
| Indian Kanoon (indiankanoon.org) | Selected judgment summaries across criminal/civil/consumer domains | ~500 judgments |
| Synthetic (Groq/template) | Q&A pairs generated from parsed Bare Act sections | ~1,500 pairs |
| **Total** | | **~1,939 instruction pairs** |

### Format

All training data follows the Alpaca instruction tuning format:

```json
{
  "instruction": "What is the punishment for causing hurt by dangerous weapons under the new Indian law?",
  "input": "",
  "output": "Under Section 118 of the Bharatiya Nyaya Sanhita (BNS) 2023, whoever causes grievous hurt by means of any instrument for shooting, stabbing or cutting, or any instrument which, used as a weapon of offence, is likely to cause death, or by means of fire or any heated substance, or by means of any poison or any corrosive substance, shall be punished with imprisonment for life, or with imprisonment of either description for a term which may extend to ten years, and shall also be liable to fine. This replaces Section 326 of the Indian Penal Code (IPC) 1860."
}
```

---

## Fine-Tuning Configuration

```yaml
# config.yaml
base_model: unsloth/mistral-7b-instruct-v0.3-bnb-4bit
method: lora

lora:
  r: 8                            # LoRA rank
  lora_alpha: 16                  # Scaling factor
  target_modules:                 # Attention layers only
    - q_proj
    - v_proj
  lora_dropout: 0.05
  bias: none
  task_type: CAUSAL_LM

training:
  epochs: 3
  batch_size: 2
  gradient_accumulation_steps: 4
  learning_rate: 2e-4
  warmup_ratio: 0.03
  lr_scheduler: cosine
  max_seq_length: 512
  fp16: true                     # T4 compatible

unsloth:
  use_gradient_checkpointing: true
  random_state: 42
```

---

## Evaluation

THEMIS ships with a 50-question held-out evaluation set covering:

| Category | Questions | Metric |
|----------|-----------|--------|
| Criminal Law (BNS) | 15 | Citation accuracy |
| Civil / Property | 10 | Factual consistency |
| Consumer Rights | 10 | Response completeness |
| Procedural (BNSS) | 8 | Step accuracy |
| IPC → BNS mapping | 7 | Section mapping correctness |

### Metrics

- **Citation Accuracy** — Does the response cite the correct section number?
- **Refusal Rate** — Does the model correctly refuse out-of-scope queries?
- **ROUGE-L** — Overlap with ground truth responses
- **Hallucination Rate** — Manual spot-check of 20 responses for fabricated sections

Run evaluation:

```bash
themis eval --verbose
```

---

## Roadmap

| Version | Milestone |
|---------|-----------|
| v1.0 | BNS, BNSS, BSA, Consumer Protection Act, RTI Act — criminal + civil + consumer |
| v1.1 | Add Supreme Court landmark judgment summaries |
| v1.2 | Hindi language support (bilingual fine-tune) |
| v2.0 | THEMIS-HECTOR bridge — RAG + parametric hybrid inference |
| v2.1 | Web UI wrapper for non-technical users |

---

## Relationship to HECTOR

THEMIS and HECTOR are designed to eventually operate as a hybrid system:

```
User Query
    │
    ▼
┌─────────────────────────────────────┐
│         Query Classifier            │
│  "Does this need retrieval or       │
│   parametric reasoning?"            │
└──────────────┬──────────────────────┘
               │
       ┌───────┴───────┐
       ▼               ▼
  ┌─────────┐     ┌─────────┐
  │  THEMIS │     │ HECTOR  │
  │ (quick  │     │ (deep   │
  │  Q&A)   │     │research)│
  └─────────┘     └─────────┘
```

In v2.0, a unified router will dispatch queries to THEMIS for citizen-level Q&A and to HECTOR for deep legal research requiring source-level citations.

---

## Why This Exists

India has 1.4 billion people. Fewer than 2 million are lawyers. The gap between legal literacy and legal need is enormous. THEMIS is a step toward making statutory law accessible to anyone with a terminal — not as a replacement for lawyers, but as a first layer of orientation that helps people understand what laws exist, what they say, and what options they have.

---

## License

MIT License — See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- Base model: [Mistral 7B Instruct v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) by Mistral AI
- Fine-tuning framework: [Unsloth](https://github.com/unslothai/unsloth)
- Legal data: [India Code](https://indiacode.nic.in) (public domain) · [Indian Kanoon](https://indiankanoon.org)
- CLI: [Typer](https://typer.tiangolo.com) + [Rich](https://github.com/Textualize/rich)

---

<div align="center">
<i>THEMIS — Greek goddess of law, justice, and order.<br>
Because justice should not require a law degree to understand.</i>
</div>
