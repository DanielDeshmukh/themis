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
  <img src="https://img.shields.io/badge/status-v1%20trained%20%7C%20v2%20data%20blocked-orange" />
  <img src="https://img.shields.io/badge/python-3.11+-brightgreen" />
  <img src="https://img.shields.io/badge/license-MIT-lightgrey" />
</p>

*"Not retrieval. Not lookup. Pure legal reasoning, baked into weights."*

**HuggingFace:** [`Daniel2503/themis-mistral-7b-lora`](https://huggingface.co/Daniel2503/themis-mistral-7b-lora)

</div>

---

## What is THEMIS?

THEMIS is a domain-specific large language model fine-tuned on Indian statutory law. It is **not** a retrieval system, a search engine, or a chatbot wrapper around an existing API. It is a **parametric knowledge model** — one where legal understanding of the Bharatiya Nyaya Sanhita (BNS), the Indian Penal Code (IPC), the Bharatiya Nagarik Suraksha Sanhita (BNSS), and allied statutes is baked directly into the model weights through supervised fine-tuning.

Where HECTOR retrieves — THEMIS reasons.

---

## Current Status (June 2026)

### What's Built and Working

| Component | Status | Details |
|-----------|--------|---------|
| **Scraper** | Done | 1,230 sections across 8 laws from India Code |
| **v1 Synthetic Generator** | Done | Template + Groq API (Mixtral), ~2,000 pairs |
| **v2 Synthetic Generator** | Done | Multi-template (15+ categories), IPC-BNS mapping, abbreviations |
| **v3 Synthetic Generator** | Done | Chunked 5k LLM processing with checkpoint/resume |
| **Preprocessing Pipeline** | Done | Clean, dedup, merge, train/eval split → `dataset.json` |
| **CLI** | Done | 7 commands via Typer + Rich (`ask`, `chat`, `scrape`, `generate`, `preprocess`, `eval`, `info`) |
| **Evaluation Harness** | Done | 4 metrics: citation accuracy, ROUGE-L, refusal rate, hallucination rate |
| **Unit Tests** | Done | 13 tests across config, inference, metrics, preprocessing |
| **v1 Model** | Trained | LoRA rank=8, alpha=16, 1,939 pairs, published to HF Hub |
| **Packaging** | Done | `pip install themis-law`, CLI entry point `themis` |

### What's Blocked

| Blocker | Impact | Next Step |
|---------|--------|-----------|
| **Groq API rate limits** | v3/60k LLM generation fails immediately with 429 errors | Use multiple API keys, switch to paid tier, or use alternative free APIs (Together, OpenRouter) |
| **Training data** | v2 needs 10-15k pairs, v3 needs 50-90k. Currently only 8,264 template pairs (no LLM augmentation) | Resolve API bottleneck or expand template engine |
| **No local model weights** | `model/` is empty; adapter hosted on HF Hub only | Download adapter for offline use |

### What v1 Actually Achieved

**Honest assessment** — v1 was a proof of concept, not a production model:

**What works:**
- End-to-end fine-tuning pipeline on Kaggle free T4 GPU
- LoRA adapter published to HuggingFace Hub
- Model learns Alpaca instruction format and legal assistant style
- Disclaimer behavior trained correctly
- Response structure (citations, recommendations) partially learned

**What doesn't work:**
- BNS abbreviation recognition — model confuses "BNS" with unrelated expansions
- Section number citation — hallucinates on specific queries (~40% accuracy)
- Deep statutory knowledge — 1,939 pairs was insufficient for domain grounding
- IPC → BNS mapping — transition knowledge not retained at this scale

**Root cause:** Mistral 7B has near-zero BNS 2023 knowledge in pretraining (BNS enacted Dec 2023, after training cutoff). The fine-tune taught the model *how to respond like a lawyer* but not *what Indian law says*.

---

## Project Goals

### v2 — Knowledge Foundation (Current Target)

**Target:** 10,000–15,000 training pairs | LoRA rank 16 | Sequence 1,024 | T4 GPU

| Goal | Metric | Current |
|------|--------|---------|
| Training pairs | 15,000 | 8,264 (template only, LLM augmentation blocked) |
| BNS abbreviation recognition | >70% correct | ~20% |
| Section citation accuracy | >70% | ~40% |
| Hallucination rate | <30% | ~60% |
| IPC → BNS mapping | Working | Not tested |

**What v2 adds over v1:**
- 15+ question template categories (definition, punishment, procedure, offence, scenario, jurisdiction, limitation, burden of proof, etc.)
- 200+ IPC → BNS section mapping pairs
- 20 abbreviation disambiguation pairs
- LoRA rank 16, all 4 attention modules (q,k,v,o)
- Sequence length 1,024 for longer statutory text
- 50-question held-out evaluation set

### v3 — Production Grade (Future)

**Target:** 50,000–90,000 pairs | LoRA rank 32 | Sequence 2,048 | A100 GPU

| Legal Domain | Target Pairs | Sources |
|-------------|-------------|---------|
| BNS 2023 — Criminal Law | 15,000 | India Code full text |
| IPC 1860 — Legacy Criminal Law | 10,000 | India Code + IPC↔BNS mapping |
| BNSS 2023 — Criminal Procedure | 8,000 | India Code full text |
| BSA 2023 — Evidence Act | 5,000 | India Code full text |
| Consumer Protection Act 2019 | 6,000 | India Code + NCDRC summaries |
| RTI Act 2005 | 3,000 | India Code + CIC decisions |
| Indian Contract Act 1872 | 5,000 | India Code full text |
| Transfer of Property Act 1882 | 4,000 | India Code full text |
| Supreme Court landmark judgments | 10,000 | Indian Kanoon |
| IPC → BNS transition mapping | 8,000 | Section-level comparison |
| **Total** | **74,000** | |

**Success criteria:** Citation accuracy >85%. Hallucination rate <10%.

### v4 — THEMIS-HECTOR Hybrid (Vision)

Unify parametric reasoning (THEMIS) with retrieval grounding (HECTOR):

```
User Query
    │
    ▼
┌─────────────────────────────────────┐
│         Query Classifier            │
│  "Parametric or retrieval?"         │
└──────────────┬──────────────────────┘
               │
       ┌───────┴───────┐
       ▼               ▼
  ┌─────────┐     ┌─────────┐
  │  THEMIS │     │ HECTOR  │
  │ (reason)│     │(retrieve│
  │         │     │+ verify)│
  └────┬────┘     └────┬────┘
       └───────┬───────┘
              ▼
      Unified Legal Response
      with citations + reasoning
```

---

## Architecture

```
themis/
├── __init__.py              # Package exports, v1.0.0
├── config.py                # All project config (paths, model, training, generation)
├── cli.py                   # Rich CLI: ask, chat, scrape, generate, preprocess, eval, info
├── infer.py                 # ThemisInference class — model loading + generation
├── data/
│   ├── preprocess.py        # Clean, dedup, merge, train/eval split
│   ├── dataset.json         # Training dataset (27,167 lines)
│   ├── scraper/
│   │   ├── indiacode.py     # India Code Bare Acts scraper (resilient, retry/backoff)
│   │   ├── kanoon.py        # Indian Kanoon judgment scraper
│   │   └── raw/             # 8 scraped law JSONs (1,230 sections)
│   └── synthetic/
│       ├── generate.py      # v1 generator (template + Groq Mixtral)
│       ├── generate_v2.py   # v2 generator (15+ categories, IPC mapping)
│       ├── generate_v3.py   # v3 generator (chunked 5k LLM, checkpoint/resume)
│       └── synthetic_qa*.json # Generated outputs
├── training/
│   ├── config.yaml          # LoRA v2 config (rank=16, alpha=32)
│   ├── finetune.py          # Unsloth + SFTTrainer (Kaggle T4)
│   └── push_to_hub.py       # HuggingFace Hub upload
├── eval/
│   ├── metrics.py           # Citation accuracy, ROUGE-L, refusal, hallucination
│   ├── run_eval.py          # Evaluation runner
│   └── eval_set.json        # 50 held-out evaluation questions
└── tests/
    └── test_infer.py        # 13 unit tests
```

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Base Model | Mistral 7B Instruct v0.3 (4-bit NF4) | Foundation |
| Fine-tuning | LoRA via PEFT + Unsloth | Parameter-efficient training |
| Training Platform | Kaggle T4 (v1/v2) → A100 (v3) | Compute |
| Dataset Format | Alpaca instruction tuning | Standard SFT format |
| Data Sources | India Code + Indian Kanoon + Synthetic | Scraping + generation |
| Synthetic Generation | Groq API (Llama 3.1 8B) + templates | Q&A pair generation |
| CLI | Typer + Rich | Terminal interface |
| Inference | Transformers + PEFT + BitsAndBytes | LoRA adapter loading |
| Evaluation | Custom 4-metric harness | Quality measurement |
| Packaging | pyproject.toml + setuptools | `pip install themis-law` |
| Model Hosting | HuggingFace Hub | Public model access |

---

## Training Configuration

### v1 (Completed)

```yaml
base_model: unsloth/mistral-7b-instruct-v0.3-bnb-4bit
lora_r: 8
lora_alpha: 16
target_modules: [q_proj, v_proj]
lora_dropout: 0
epochs: 3
batch_size: 1
gradient_accumulation: 8
learning_rate: 2e-4
max_seq_length: 512
platform: Kaggle T4 (free)
training_pairs: 1,939
```

### v2 (Target)

```yaml
lora_r: 16
lora_alpha: 32
target_modules: [q_proj, k_proj, v_proj, o_proj]
lora_dropout: 0.05
max_seq_length: 1024
training_pairs: 15,000
platform: Kaggle T4
```

### v3 (Future)

```yaml
lora_r: 32
lora_alpha: 64
target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
max_seq_length: 2048
training_pairs: 74,000
platform: RunPod A100 (40GB)
```

---

## Evaluation Framework

**4-metric evaluation harness:**

| Metric | What It Measures | v1 Result | v2 Target | v3 Target |
|--------|-----------------|-----------|-----------|-----------|
| **Citation Accuracy** | Correct section numbers cited | ~40% | >70% | >85% |
| **ROUGE-L** | Text overlap with ground truth | TBD | >0.4 | >0.6 |
| **Refusal Rate** | Correctly declines out-of-scope queries | TBD | >80% | >95% |
| **Hallucination Rate** | Fabricated section numbers / act names | ~60% | <30% | <10% |

---

## Data Pipeline Status

### Scraped Data (1,230 sections)

| Law | Sections | File |
|-----|----------|------|
| Bharatiya Nyaya Sanhita 2023 | ~358 | `the_bharatiya_nyaya_sanhita_2023.json` |
| BNSS 2023 | ~531 | `the_bharatiya_nagarik_suraksha_sanhita_2023.json` |
| BSA 2023 | ~170 | `the_bharatiya_sakshya_adhiniyam_2023.json` |
| Consumer Protection Act 2019 | ~100+ | `the_consumer_protection_act_2019.json` |
| RTI Act 2005 | ~31 | `the_right_to_information_act_2005.json` |
| Industrial Disputes Act | ~10+ | `the_industrial_disputes_banking_and_insurance_companies_act_1949.json` |
| Repealing and Amending Act 2016 | Small | `the_repealing_and_amending_act_2016.json` |
| Terrorist Affected Areas Act 1984 | Small | `the_terrorist_affected_areas_special_courts_act_1984.json` |

### Synthetic Generation Versions

| Version | Strategy | Output | Status |
|---------|----------|--------|--------|
| v1 | 1 Q&A per section via Groq (Mixtral) | `synthetic_qa.json` | Done |
| v2 | 3-5 Q&A per section + IPC-BNS mapping (65 entries) + abbreviations (8) | `synthetic_qa_v2.json` | Done |
| v3 | 10+ Q&A per section, chunked 5k processing, 2-min breaks, checkpoint/resume | `synthetic_qa_v3.json` | **Rate limited** |

### Current Dataset

- **Template baseline:** 8,264 pairs (instant, no API needed)
- **LLM-augmented:** 0 pairs (Groq rate limits blocking)
- **Total in `dataset.json`:** 27,167 lines (v1 preprocessing output)
- **Needed for v2:** 10,000–15,000 pairs
- **Needed for v3:** 50,000–90,000 pairs

---

## Relationship to HECTOR

| | THEMIS | HECTOR |
|---|---|---|
| Architecture | Parametric fine-tune (LoRA) | RAG (Qdrant + Chain-of-Verification) |
| Knowledge | Model weights | External vector database |
| Runtime documents | Not needed | Required |
| Best for | Citizen Q&A | Deep legal research |
| Citations | Parametric (may hallucinate) | Source-grounded (verified) |
| Status | v1 trained, v2 blocked on data | Production-ready |

---

## How to Use

### Install

```bash
pip install themis-law
```

### CLI Commands

```bash
# Ask a legal question
themis ask "What is the punishment for theft under BNS?"

# Interactive chat
themis chat

# Scrape legal data
themis scrape --law all
themis scrape --law bns

# Generate synthetic Q&A pairs
themis generate --v2
themis generate --v3

# Preprocess dataset
themis preprocess

# Run evaluation
themis eval --verbose

# Model info
themis info
themis version
```

### Python API

```python
from themis import load_model, generate_response

load_model()
response = generate_response("What is Section 303 of BNS?")
print(response)
```

---

## Known Limitations

- **BNS abbreviation confusion** — spell out "Bharatiya Nyaya Sanhita 2023" for best results
- **Section hallucination** — ~60% hallucination rate on BNS queries in v1
- **No case law knowledge** — statutes only, no judgments
- **English only** — Hindi support planned for v3
- **State-specific laws not covered** — only central acts
- **Groq rate limits** — LLM data augmentation blocked
- **No offline model** — requires internet for first run (HF Hub download)

---

## Why This Exists

India has 1.4 billion people. Fewer than 2 million are lawyers. The gap between legal literacy and legal need is enormous. THEMIS is a step toward making statutory law accessible to anyone — not as a replacement for lawyers, but as a first layer of orientation that helps people understand what laws exist, what they say, and what options they have.

At 90,000 training pairs, a model can genuinely know Indian law. That is the goal.

---

## License

MIT License

---

## Citation

```bibtex
@misc{themis2026,
  author = {Daniel Deshmukh},
  title = {THEMIS: Parametric Legal Intelligence Engine for Indian Law},
  year = {2026},
  publisher = {HuggingFace},
  url = {https://huggingface.co/Daniel2503/themis-mistral-7b-lora}
}
```

---

*THEMIS — Greek goddess of law, justice, and order.*
*Because justice should not require a law degree to understand.*
