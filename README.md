⭐ If THEMIS sparked ideas about fine-tuning LLMs on domain-specific law — a star helps other researchers find it. Takes 2 seconds.

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
  <img src="https://img.shields.io/badge/status-v1%20trained%20%7C%20v2%20in%20progress-orange" />
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

## Current State — v1 (Honest Assessment)

> This section exists because honest evaluation is more valuable than a polished demo.

**What v1 demonstrates:**
- ✅ End-to-end fine-tuning pipeline working on Kaggle free T4 GPU
- ✅ LoRA adapter trained and published to HuggingFace Hub
- ✅ Alpaca instruction format learned — model responds in legal assistant style
- ✅ Disclaimer behavior trained correctly
- ✅ Response structure (citations, recommendations, disclaimers) partially learned

**What v1 does NOT do well:**
- ❌ BNS 2023 abbreviation recognition — model confuses "BNS" with unrelated expansions
- ❌ Accurate section number citation — hallucinates section numbers on specific queries
- ❌ Deep statutory knowledge retention — 1,939 pairs was insufficient for domain grounding
- ❌ IPC → BNS mapping — transition knowledge not retained at this data scale

**Root cause analysis:**

Mistral 7B Instruct v0.3 has near-zero BNS 2023 knowledge in its pretraining data — BNS was enacted in December 2023, at or after Mistral's training cutoff. This means there is no foundation for the LoRA to build on. The fine-tune taught the model *how to respond like a lawyer* but not *what Indian law says*.

The technical constraints that caused this:

| Parameter | v1 Value | Minimum Needed | v2 Target |
|-----------|----------|----------------|-----------|
| LoRA rank | 8 | 16 | 32 |
| Sequence length | 512 tokens | 1,024 tokens | 2,048 tokens |
| Training pairs | 1,939 | 10,000+ | 50,000–90,000 |
| Target modules | q_proj, v_proj | q,k,v,o proj | q,k,v,o + MLP |
| Epochs | 3 | 3–5 | 3–5 |

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

### v2 — Knowledge Foundation (In Progress)

**Target:** 10,000–15,000 pairs | LoRA rank 16 | Sequence 1,024 | T4 x2

- [ ] Expand dataset to 15,000 pairs (BNS + IPC + BNSS full coverage)
- [ ] Retrain with rank 16, all 4 attention modules (q,k,v,o)
- [ ] Sequence length 1,024 for longer statutory text
- [ ] Add BNS abbreviation disambiguation pairs explicitly
- [ ] Evaluate on 100-question held-out set with citation accuracy metric
- [ ] Publish v2 adapter to HuggingFace

**Success criteria:** Model correctly identifies BNS as Bharatiya Nyaya Sanhita and cites accurate section numbers on 70%+ of criminal law queries.

---

### v3 — Production Grade (Planned)

**Target:** 50,000–90,000 pairs | LoRA rank 32 | Sequence 2,048 | A100 (Colab Pro or RunPod)

- [ ] Full India Code corpus ingestion (all central acts)
- [ ] Indian Kanoon top 1,000 judgment summaries
- [ ] IPC → BNS complete transition mapping (all 511 sections)
- [ ] Hindi language support (bilingual fine-tune)
- [ ] RAGAS-style evaluation harness with citation F1 scoring
- [ ] Systematic hallucination rate measurement
- [ ] Publish v3 adapter to HuggingFace with full model card

**Success criteria:** Citation accuracy >85% on held-out eval set. Hallucination rate <10% on factual section number queries.

---

### v4 — THEMIS-HECTOR Hybrid (Vision)

The long-term architecture unifies THEMIS (parametric reasoning) with HECTOR (retrieval grounding):

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

THEMIS handles citizen-level Q&A with parametric reasoning. HECTOR handles deep legal research requiring source-level PDF citations. A unified router dispatches based on query complexity.

---

## Architecture

```
themis/
├── cli.py                  # Rich-powered CLI entry point
├── infer.py                # Model loading and inference engine
├── config.py               # Model path, generation params, device config
├── eval/
│   ├── run_eval.py         # Evaluation harness
│   ├── metrics.py          # Citation accuracy, refusal rate, ROUGE-L
│   └── eval_set.json       # Ground truth evaluation dataset
├── data/
│   ├── scraper/
│   │   ├── kanoon.py       # Indian Kanoon judgment scraper
│   │   └── indiacode.py    # India Code Bare Acts parser
│   ├── synthetic/
│   │   └── generate.py     # Claude-assisted Q&A pair generation
│   ├── preprocess.py       # Cleaning, deduplication, formatting
│   └── dataset.json        # Training dataset (v1: 1,939 pairs)
├── training/
│   ├── finetune.py         # Unsloth + LoRA training script
│   ├── config.yaml         # LoRA hyperparameters
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
| Training Platform | Kaggle free T4 (v1) → RunPod A100 (v3) | Compute |
| Dataset Format | Alpaca instruction tuning | Standard SFT format |
| Data Sources | India Code + Indian Kanoon + Synthetic | Scraping + generation |
| Synthetic Generation | Claude API | Q&A pair generation from Bare Acts |
| CLI | Typer + Rich | Terminal interface |
| Inference | HuggingFace Transformers + PEFT | LoRA adapter loading |
| Evaluation | Custom harness + citation F1 | Quality measurement |
| Model Hosting | HuggingFace Hub | Public model access |

---

## Dataset Construction

### v1 Dataset (Current — 1,939 pairs)

Generated from India Code Bare Acts using Claude API for synthetic Q&A pair generation. Format:

```json
{
  "instruction": "What does Section 303 of the Bharatiya Nyaya Sanhita say about theft?",
  "input": "",
  "output": "Section 303 of the Bharatiya Nyaya Sanhita (BNS) 2023 defines theft as..."
}
```

### v2 Dataset Plan (10,000–15,000 pairs)

Expanding sources to cover full BNS, IPC, BNSS, BSA statutory text with explicit abbreviation disambiguation pairs.

### v3 Dataset Plan (50,000–90,000 pairs)

Full India Code corpus + Indian Kanoon judgment summaries + complete IPC→BNS transition mapping. At this scale, the dataset size matches the clinical corpus depth of production medical AI systems.

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

### v2 (Planned)

```yaml
lora_r: 16
target_modules: [q_proj, k_proj, v_proj, o_proj]
max_seq_length: 1024
training_pairs: 15,000
platform: Kaggle T4 x2
```

### v3 (Planned)

```yaml
lora_r: 32
target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
max_seq_length: 2048
training_pairs: 74,000
platform: RunPod A100 (40GB)
```

---

## Evaluation Framework

THEMIS uses a 3-tier evaluation system:

**Tier 1 — Citation Accuracy**
Does the response cite the correct section number?
Target: >85% on criminal law queries by v3.

**Tier 2 — Hallucination Rate**
Does the model fabricate section numbers or act names?
Target: <10% hallucination rate by v3.
Current v1 rate: ~60% on BNS-specific queries (abbreviation confusion).

**Tier 3 — Refusal Rate**
Does the model correctly decline out-of-scope queries?
Target: >95% correct refusal on state-specific law queries.

---

## Known Limitations (v1)

- BNS 2023 abbreviation confusion — spell out "Bharatiya Nyaya Sanhita 2023" for best results
- Section number hallucination on specific criminal law queries
- No case law knowledge — statutes only
- English only
- State-specific laws not covered
- Best used as orientation, not as authoritative legal reference

---

## Why This Exists

India has 1.4 billion people. Fewer than 2 million are lawyers. The gap between legal literacy and legal need is enormous. THEMIS is a step toward making statutory law accessible to anyone — not as a replacement for lawyers, but as a first layer of orientation that helps people understand what laws exist, what they say, and what options they have.

At 90,000 training pairs, a model can genuinely know Indian law. That is the goal.

---

## Relationship to HECTOR

| | THEMIS | HECTOR |
|---|---|---|
| Architecture | Parametric fine-tune (LoRA) | RAG (Qdrant + Chain-of-Verification) |
| Knowledge | Model weights | External vector database |
| Runtime documents | Not needed | Required |
| Best for | Citizen Q&A | Deep legal research |
| Citations | Parametric (may hallucinate) | Source-grounded (verified) |
| Status | v1 trained, v2 in progress | Production-ready |

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
