# THEMIS — Progress Tracker

## Current Status: PLANNING

**Phase:** Pre-development
**Last Updated:** 2026-06-09

---

## Phase Status

| Phase | Status | Start | End | Notes |
|-------|--------|-------|-----|-------|
| Phase 1: Data Pipeline | Not Started | — | — | Scrape + generate training data |
| Phase 2: Training | Not Started | — | — | Unsloth + LoRA on Kaggle |
| Phase 3: Inference | Not Started | — | — | Model loading + CLI |
| Phase 4: Evaluation | Not Started | — | — | Eval harness + metrics |
| Phase 5: Packaging | Not Started | — | — | pip install + PyPI |

---

## Task Breakdown

### Phase 1: Data Pipeline

| # | Task | Assignee | Status | Notes |
|---|------|----------|--------|-------|
| 1.1 | Set up data/ directory structure | — | Not Started | |
| 1.2 | Build India Code scraper (indiacode.py) | — | Not Started | Parse BNS, BNSS, BSA, CPA, RTI sections |
| 1.3 | Build Indian Kanoon scraper (kanoon.py) | — | Not Started | 500 judgment summaries |
| 1.4 | Set up Claude synthetic generation (generate.py) | — | Not Started | ~1500 Q&A pairs |
| 1.5 | Build preprocessing pipeline (preprocess.py) | — | Not Started | Clean, dedup, format to Alpaca JSON |
| 1.6 | Create final dataset.json | — | Not Started | ~2800 instruction pairs |
| 1.7 | Data validation and quality checks | — | Not Started | Verify citations are correct |

### Phase 2: Training

| # | Task | Assignee | Status | Notes |
|---|------|----------|--------|-------|
| 2.1 | Set up Kaggle notebook environment | — | Not Started | T4 GPU, Unsloth pre-installed |
| 2.2 | Create config.yaml with LoRA hyperparameters | — | Not Started | r=16, alpha=32, 3 epochs |
| 2.3 | Write finetune.py training script | — | Not Started | Unsloth + LoRA |
| 2.4 | Run training on Kaggle | — | Not Started | ~2-3 hours on T4 |
| 2.5 | Export LoRA adapter weights | — | Not Started | Save adapter only |
| 2.6 | Upload to HuggingFace Hub | — | Not Started | danieldeshmukh/themis-mistral-7b-lora |
| 2.7 | Write push_to_hub.py | — | Not Started | Automated upload script |

### Phase 3: Inference

| # | Task | Assignee | Status | Notes |
|---|------|----------|--------|-------|
| 3.1 | Create config.py with Config dataclass | — | Not Started | Model paths, generation params |
| 3.2 | Implement model loading in infer.py | — | Not Started | 4-bit quantization + LoRA |
| 3.3 | Design and implement system prompt | — | Not Started | Legal domain instructions |
| 3.4 | Build chat template formatter | — | Not Started | [INST] format for Mistral |
| 3.5 | Implement generate_response() | — | Not Started | Core inference function |
| 3.6 | Build Rich CLI (cli.py) | — | Not Started | Typer + Rich formatting |
| 3.7 | Implement ask command | — | Not Started | Single-shot Q&A |
| 3.8 | Implement chat command | — | Not Started | Multi-turn conversation |
| 3.9 | Add info and version commands | — | Not Started | Metadata display |

### Phase 4: Evaluation

| # | Task | Assignee | Status | Notes |
|---|------|----------|--------|-------|
| 4.1 | Create eval_set.json (50 questions) | — | Not Started | 5 categories, ground truth |
| 4.2 | Implement citation_accuracy metric | — | Not Started | Check correct section numbers |
| 4.3 | Implement refusal_rate metric | — | Not Started | Out-of-scope handling |
| 4.4 | Implement ROUGE-L metric | — | Not Started | Lexical overlap |
| 4.5 | Build run_eval.py harness | — | Not Started | Run all questions, aggregate metrics |
| 4.6 | Run evaluation on v1.0 model | — | Not Started | Document results |
| 4.7 | Manual hallucination spot-check | — | Not Started | 20 responses, check for fabricated sections |

### Phase 5: Packaging

| # | Task | Assignee | Status | Notes |
|---|------|----------|--------|-------|
| 5.1 | Create pyproject.toml | — | Not Started | Package metadata + deps |
| 5.2 | Define CLI entry points | — | Not Started | `themis` command |
| 5.3 | Test pip install in clean environment | — | Not Started | Verify dependencies |
| 5.4 | Write installation documentation | — | Not Started | README section |
| 5.5 | Publish to PyPI | — | Not Started | `pip install themis-law` |
| 5.6 | Create GitHub release | — | Not Started | Tag v1.0.0 |

---

## Blockers

| # | Blocker | Impact | Status | Resolution |
|---|---------|--------|--------|------------|
| 1 | No training data yet | Cannot train model | Open | Build data pipeline first |
| 2 | No LoRA adapter weights | Cannot run inference | Open | Need Phase 2 complete |
| 3 | India Code scraping unknowns | Data quality risk | Open | Test scraper early |
| 4 | Kaggle GPU quota limits | Training speed | Open | Plan for quota resets |
| 5 | Claude API costs | Synthetic generation cost | Open | Budget for ~1500 pairs |

---

## Completed Work

| Date | Task | Notes |
|------|------|-------|
| 2026-06-09 | README.md created | Full project specification |
| 2026-06-09 | Context files created | architecture, roadmap, progress, security, tech-stack, data-pipeline |

---

## Next Actions

1. **Immediate:** Build data scraper for India Code (task 1.2)
2. **This Week:** Set up data/ directory and preprocessing pipeline
3. **Next Week:** Start Claude synthetic generation for Q&A pairs
