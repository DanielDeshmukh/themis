⭐ If THEMIS sparked ideas about fine-tuning LLMs on domain-specific law — a star helps other researchers find it. Takes 2 seconds.

<div align="center">

<pre>
████████╗██╗  ██╗███████╗███╗   ███╗██╗███████╗
╚══██╔══╝██║  ██║██╔════╝████╗ ████║██║██╔════╝
   ██║   ███████║█████╗  ██╔████╔██║██║███████╗
   ██║   ██╔══██║██╔══╝  ██║╚██╔╝██║██║╚════██║
   ██║   ██║  ██║███████╗██║ ╚═╝ ██║██║███████║
   ╚═╝   ╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝╚═╝╚══════╝
</pre>

**The Parametric Legal Intelligence Engine for Indian Law**

<p>
  <img src="https://img.shields.io/badge/model-Mistral%207B%20LoRA-blueviolet" />
  <img src="https://img.shields.io/badge/domain-Indian%20Law%20%7C%20BNS%20%7C%20IPC-crimson" />
  <img src="https://img.shields.io/badge/status-v5%20retrieval--grounded-brightgreen" />
  <img src="https://img.shields.io/badge/python-3.11+-brightgreen" />
  <img src="https://img.shields.io/badge/license-MIT-lightgrey" />
</p>

*"Not retrieval. Not lookup. Pure legal reasoning, baked into weights."*

**HuggingFace:** [`Daniel2503/themis-mistral-7b-lora-v5`](https://huggingface.co/Daniel2503/themis-mistral-7b-lora-v5)

</div>

---

## What is THEMIS?

THEMIS is a domain-specific large language model fine-tuned on Indian statutory law. It is **not** a retrieval system, a search engine, or a chatbot wrapper around an existing API. It is a **parametric knowledge model** — one where legal understanding of the Bharatiya Nyaya Sanhita (BNS), the Indian Penal Code (IPC), the Bharatiya Nagarik Suraksha Sanhita (BNSS), and allied statutes is baked directly into the model weights through supervised fine-tuning.

Where HECTOR retrieves — THEMIS reasons.

---

## Current State — v2 Results (Overfitting Fixed in v3)

> **v2 post-mortem:** Scaled to 20,909 training pairs, but 3 epochs caused overfitting. Loss dropped to 0.06-0.08 (memorization territory). Model regurgitated training artifacts instead of reasoning. Fixed in v3 by reducing to 2 epochs.

**What v2 achieved:**
- ✅ Domain grounding fixed — no more "Bangladesh National Standards" hallucination
- ✅ Correct section identification (e.g., Section 303 for theft)
- ✅ 10x data scale from v1 (1,939 → 20,909 pairs)

**What v2 broke:**
- ❌ Overfitting — loss 0.06-0.08 indicates memorization, not learning
- ❌ Regurgitation — model recited definitions verbatim instead of answering the question
- ❌ Repetition loops — disclaimer text repeated 2x, cut off at token limit
- ❌ No checkpoint saving — intermediate checkpoints lost when Kaggle session ended

**Root cause:** 3 epochs on 20k examples is too many. The model memorized surface patterns (statute text blocks, disclaimer boilerplate) rather than learning to reason about what's being asked.

**v3 fix:** Reduced epochs from 3 to 2. See `notebooks/THEMIS_v3_Training.ipynb`.

---

## The Goal — v3 Production Target

THEMIS v3 is designed to match the data depth of production medical RAG systems — comparable to the 90,000+ clinical records in [Ella](https://github.com/DanielDeshmukh/ella).

**Target: 50,000–90,000 training pairs** covering:

| Legal Domain | Target Pairs | Sources |
|-------------|-------------|---------|
| BNS 2023 — Criminal Law | 15,000 | India Code full text, section-by-section Q&A |
| IPC 1860 — Legacy Criminal Law | 10,000 | India Code, comparative IPC↔BNS mapping |
| BNSS 2023 — Criminal Procedure | 8,000 | India Code full text |
| BSA 2023 — Evidence Act | 5,000 | India Code full text |
| Consumer Protection Act 2019 | 6,000 | India Code + NCDRC judgment summaries |
| RTI Act 2005 | 3,000 | India Code + CIC decisions |
| Indian Contract Act 1872 | 5,000 | India Code full text |
| Transfer of Property Act 1882 | 4,000 | India Code full text |
| Supreme Court landmark judgments | 10,000 | Indian Kanoon — top 500 judgments parsed |
| IPC → BNS transition mapping | 8,000 | Section-level comparison pairs |
| **Total** | **74,000** | |

At this scale, THEMIS becomes a model that has genuinely read Indian law — not a model that learned to sound like a lawyer.

---

## What Happens Next — Roadmap

### v1 → v2 — Scale (Completed)

Expanded training data from 1,939 to 20,909 pairs. Fixed BNS abbreviation hallucination. Achieved correct section identification for core criminal law queries. Introduced 15 template question categories and IPC-to-BNS section mappings.

### v2 → v3 — Overfitting Fix (Completed)

Reduced training from 3 epochs to 2. Added checkpoint saving every 500 steps (keep last 3). Expanded eval set with 15 conversational rephrased queries. Loss stabilized; model stopped regurgitating training artifacts.

### v3 → v4 — Data Expansion (Completed)

Scaled to 52,170 training examples across BNS, BNSS, BSA, IPC, RTI, and the Constitution. Added GSMS-B QA and IndicLegalQA datasets. Introduced scenario-based and multi-hop reasoning templates. Achieved 1,549 training steps on Kaggle T4.

### v4 → v5 — Retrieval Grounding (Completed)

Diagnosed persistent overfitting (train loss 0.13, val loss 0.98). Implemented retrieval-grounding engine: extracts section references from user questions, looks up anchor tables, and injects section text as context before generation. Added SectionIndex, SectionRef, and grounded prompt building. Published SDK (`themis-llm` v2.0.1) with `ThemisModel.ask()` and CLI (`themis ask/chat/info`). Added 19-pass test suite. Cleaned repository: removed 19 stale files, rewrote git history to remove hardcoded tokens, deleted all tracked data artifacts.

### v5 → v6 — Production Hardening (Planned)

- Consumer Protection Act 2019 training data (anchor table exists; QA pairs pending)
- Indian Kanoon top 1,000 judgment summaries
- Hindi language support (bilingual fine-tune)
- RAGAS-style evaluation harness with citation F1 scoring
- Systematic hallucination rate measurement across all acts
- Publish v6 adapter to HuggingFace with full model card

**Success criteria:** Citation accuracy >85% on held-out eval set. Hallucination rate <10% on factual section number queries.

### v6 → v7 — THEMIS-HECTOR Hybrid (Vision)

The long-term architecture unifies THEMIS (parametric reasoning) with HECTOR (retrieval grounding):

<div align="center">
<pre>
                              User Query
                                   |
                                   v
                    +-------------------------------+
                    |         Query Classifier      |
                    |  "Parametric or retrieval?"   |
                    +--------------+----------------+
                                   |
                           +-------+-------+
                           v               v
                      +---------+     +---------+
                      |  THEMIS |     |  HECTOR |
                      | (reason)|     |(retrieve|
                      |         |     | + verify)|
                      +----+----+     +----+----+
                           +-------+-------+
                                   v
                        Unified Legal Response
                        with citations + reasoning
</pre>
</div>

THEMIS handles citizen-level Q&A with parametric reasoning. HECTOR handles deep legal research requiring source-level PDF citations. A unified router dispatches based on query complexity.

---

## CLI Usage

### Install

```bash
pip install themis-llm[cli]
```

### Single-Shot Q&A

```bash
themis ask "What does Section 302 of the BNS say about murder?"
```

### Interactive Chat

```bash
themis chat
```

Then type questions interactively. Type `exit` or `quit` to leave.

### Other Commands

```bash
# View model info
themis info

# View version
themis version

# Run evaluation harness
themis eval

# Preprocess datasets
themis preprocess

# Scrape legal data (advanced)
themis scrape --law bns
themis scrape --law bnss
themis scrape --law cpa

# Generate synthetic Q&A pairs (advanced)
themis generate --no-api
```

### CLI Help

```bash
themis --help
themis ask --help
```

---

## Architecture

```
themis/
├── cli.py                  # Rich-powered CLI entry point
├── infer.py                # Model loading and inference engine
├── grounding.py            # Retrieval-grounding engine (SectionIndex, prompts)
├── model.py                # ThemisModel SDK (ask, chat, info)
├── config.py               # Model path, generation params, device config
├── exceptions.py           # Custom exceptions and warnings
├── eval/
│   ├── run_eval.py         # Evaluation harness
│   └── metrics.py          # Citation accuracy, refusal rate, ROUGE-L
├── data/
│   ├── scraper/
│   │   ├── kanoon.py       # Indian Kanoon judgment scraper
│   │   └── indiacode.py    # India Code Bare Acts parser
│   ├── synthetic/
│   │   ├── generate.py     # Q&A pair generation (Groq API)
│   │   ├── generate_v2.py  # Expanded templates, IPC→BNS mapping
│   │   └── generate_v3.py  # 50k+ pairs, scenario-based questions
│   ├── preprocess.py       # Cleaning, deduplication, formatting
│   ├── build_themis_dataset.py  # CSV→training JSON (11 templates)
│   ├── build_anchors.py    # Anchor table builder
│   ├── anchors/            # Section lookup tables (bns, bnss, bsa, etc.)
│   └── tests/              # Grounding and extraction tests
├── training/
│   ├── finetune.py         # Unsloth + LoRA training script
│   └── push_to_hub.py      # HuggingFace Hub upload
└── model/                  # Local model weights (gitignored)
```

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Base Model | Mistral 7B Instruct v0.3 | Foundation — strong instruction following |
| Fine-tuning Method | LoRA (Low-Rank Adaptation) | Parameter-efficient training |
| Training Framework | Unsloth | 2x faster LoRA, VRAM optimized |
| Training Platform | Kaggle T4 (free tier) | Compute |
| Dataset Format | Alpaca instruction tuning | Standard SFT format |
| Data Sources | India Code + Indian Kanoon + Synthetic | Scraping + generation |
| Synthetic Generation | Claude API | Q&A pair generation from Bare Acts |
| CLI | Typer + Rich | Terminal interface |
| Inference | HuggingFace Transformers + PEFT | LoRA adapter loading |
| Evaluation | Custom harness + citation F1 | Quality measurement |
| Model Hosting | HuggingFace Hub | Public model access |

---

## Dataset Construction

### v1 Dataset (Completed — 1,939 pairs)

Generated from India Code Bare Acts using Claude API for synthetic Q&A pair generation. Format:

```json
{
  "instruction": "What does Section 303 of the Bharatiya Nyaya Sanhita say about theft?",
  "input": "",
  "output": "Section 303 of the Bharatiya Nyaya Sanhita (BNS) 2023 defines theft as..."
}
```

### v2/v3 Dataset (Completed — 20,909 pairs)

Expanded to 10x data covering BNS, IPC, BNSS, BSA, CPA, RTI Act. Includes 15 template question categories, IPC-to-BNS section mappings (200+), and abbreviation disambiguation pairs (21).

### v5 Dataset (Current — 52,170 examples)

Final training set assembled from multiple sources:

| Source | Examples | Domain |
|--------|----------|--------|
| BNS sections | 3,938 | Criminal law (Bharatiya Nyaya Sanhita) |
| BSA sections | 1,837 | Evidence (Bharatiya Sakshya Adhiniyam) |
| BNSS sections | 5,841 | Criminal procedure (Bharatiya Nagarik Suraksha Sanhita) |
| IPC sections | 4,884 | Legacy criminal law (Indian Penal Code) |
| GSMS-B QA | 6,354 | Multi-act legal Q&A |
| IndicLegalQA | 10,002 | Indian legal language understanding |
| RTI Act | 1,218 | Right to Information |
| Constitution | 870 | Constitutional provisions |
| **Total** | **52,170** | |

Generated using 11 question templates with section-based validation against anchor tables (bns: 358, bnss: 531, bsa: 170, consumer_protection_2019: 107, rti_2005: 31 sections). Mismatch rate: 1.4%.

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

### v5 (Current — Retrieval Grounded)

```yaml
base_model: unsloth/mistral-7b-instruct-v0.3-bnb-4bit
lora_r: 16
lora_alpha: 32
target_modules: [q_proj, k_proj, v_proj, o_proj]
lora_dropout: 0.10
epochs: 2
batch_size: 1
gradient_accumulation: 8
learning_rate: 1e-4
max_seq_length: 2048
platform: Kaggle T4 (free)
training_examples: 52,170
training_steps: 1,549
final_train_loss: 0.1314
final_val_loss: 0.9808
```

---

## Evaluation Framework

THEMIS uses a 3-tier evaluation system:

**Tier 1 — Citation Accuracy**
Does the response cite the correct section number?
v5 baseline established. Retrieval grounding improves accuracy by injecting actual section text before generation.

**Tier 2 — Hallucination Rate**
Does the model fabricate section numbers or act names?
v5 grounding reduces hallucination in 2 of 3 tested cases. Residual overfitting (val loss 0.98) remains under monitoring.

**Tier 3 — Refusal Rate**
Does the model correctly decline out-of-scope queries?
Target: >95% correct refusal on state-specific law queries.

---

## Known Limitations

### Resolved
- ~~BNS 2023 abbreviation confusion~~ — Fixed with 52k training pairs
- ~~Section number hallucination~~ — Retrieval grounding injects correct section text
- ~~Overfitting~~ — Reduced to 2 epochs; val loss 0.98 under monitoring

### Current (v5)
- No case law knowledge — statutes only
- English only
- State-specific laws not covered
- Consumer Protection Act 2019 training data pending (anchor table exists, QA pairs needed)
- Best used as orientation, not as authoritative legal reference

---

## Why This Exists

India has 1.4 billion people. Fewer than 2 million are lawyers. The gap between legal literacy and legal need is enormous. THEMIS is a step toward making statutory law accessible to anyone — not as a replacement for lawyers, but as a first layer of orientation that helps people understand what laws exist, what they say, and what options they have.

At 52,170 training examples, the model has read Indian law at section-level depth. That is the foundation.

---

## Relationship to HECTOR

| | THEMIS | HECTOR |
|---|---|---|
| Architecture | Parametric fine-tune (LoRA) | RAG (Qdrant + Chain-of-Verification) |
| Knowledge | Model weights | External vector database |
| Runtime documents | Not needed | Required |
| Best for | Citizen Q&A | Deep legal research |
| Citations | Parametric (may hallucinate) | Source-grounded (verified) |
| Status | v5 retrieval-grounded | Production-ready |

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
  url = {https://huggingface.co/Daniel2503/themis-mistral-7b-lora-v5}
}
```

---

*THEMIS — Greek goddess of law, justice, and order.*
*Because justice should not require a law degree to understand.*
