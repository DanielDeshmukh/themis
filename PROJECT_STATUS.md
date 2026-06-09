# THEMIS Project Status Report

**Generated:** 2026-06-08
**Current Version:** v1.0.0

---

## Executive Summary

THEMIS is a CLI-based legal intelligence engine for Indian law. The project structure is complete with all major modules implemented. However, there are **critical bugs**, **missing files**, and **incomplete features** that prevent deployment to PyPI. The project is approximately **70% complete** — the scaffolding is solid, but the pipeline cannot run end-to-end without fixes.

---

## What Works

| Component | Status | Notes |
|-----------|--------|-------|
| CLI banner + help | ✅ Works | `themis` shows ASCII art, `themis --help` shows all commands |
| `themis version` | ✅ Works | Shows v1.0.0, model name, Python version |
| `themis info` | ✅ Works | Shows model metadata table |
| `themis ask --help` | ✅ Works | Shows usage for single-shot Q&A |
| `themis chat --help` | ✅ Works | Shows usage for interactive mode |
| `themis eval --help` | ✅ Works | Shows usage for evaluation |
| `themis scrape --help` | ✅ Works | Shows law filter and delay options |
| `themis generate --help` | ✅ Works | Shows API/template toggle |
| `themis preprocess --help` | ✅ Works | Shows usage |
| `config.py` | ✅ Complete | Clean dataclass config |
| `training/config.yaml` | ✅ Complete | LoRA hyperparameters correct |
| `.gitignore` | ✅ Complete | Properly ignores weights, data, env |
| `themis.cmd` | ✅ Works | Windows launcher (requires venv) |

---

## What's Broken (Critical)

### 1. LICENSE File Missing
- **File:** `LICENSE` (does not exist)
- **Impact:** README references MIT license, `pyproject.toml` declares it, but the file doesn't exist. **Blocks PyPI deployment.**

### 2. No `__init__.py` Files
- **Impact:** Package imports will fail when installed via `pip install themis-law`. The `pyproject.toml` entry point `themis = "cli:app"` requires proper package structure.
- **Missing in:** `data/`, `data/scraper/`, `data/synthetic/`, `eval/`, `training/`, `tests/`

### 3. Bare Imports (CWD-Dependent)
- **Files:** ALL source files use `from config import config`, `from infer import load_model`, etc.
- **Impact:** Works only when CWD is project root. Breaks when installed as a package or run from another directory.

### 4. Kanoon Scraper Bug
- **File:** `data/scraper/kanoon.py:79`
- **Bug:** `resp = self._get(url, params=params)` — `_get()` method doesn't accept `params` kwarg
- **Impact:** Kanoon scraper will crash with `TypeError`

### 5. Training Script Broken
- **File:** `training/finetune.py:88`
- **Bug:** Passes raw Python `list` to `SFTTrainer` instead of HuggingFace `Dataset` object
- **Bug:** `dataset_text_field=None` with no `formatting_func` — SFTTrainer won't know how to tokenize
- **Impact:** Training will crash at runtime

### 6. `eval/eval_set.json` Missing
- **File:** `eval/eval_set.json` (does not exist)
- **Impact:** `themis eval` will fail — no ground truth data to evaluate against

### 7. `data/dataset.json` Missing
- **File:** `data/dataset.json` (does not exist)
- **Impact:** No training data exists yet. Must run full pipeline first.

### 8. No Model Weights
- **Directory:** `model/` is empty
- **Impact:** `themis ask` and `themis chat` will fail — no LoRA adapter to load
- **Note:** Expected — weights are gitignored and would be downloaded from HuggingFace Hub

### 9. Raw Scraped Data Empty
- **Directory:** `data/scraper/raw/` is empty
- **Impact:** No source material for Q&A generation

---

## What's Incomplete (Medium Priority)

### 1. Refusal Rate & Hallucination Rate Not Wired
- **File:** `eval/metrics.py`
- Functions `refusal_rate()` and `hallucination_check()` exist but are never called in `run_evaluation()`
- README claims 4 metrics, only 2 are computed (citation accuracy, ROUGE-L)

### 2. IPC→BNS Mapping Not Implemented
- **File:** `data/scraper/indiacode.py`
- `replaces` field exists but is never populated
- README claims "Maps legacy IPC sections to their BNS equivalents"

### 3. `format_instruction` Dead Code
- **File:** `training/finetune.py:76`
- Function defined but never called

### 4. Unused Dependencies
- `PyMuPDF` listed in `pyproject.toml` but never imported
- `import yaml` in `finetune.py` not listed in any dependency group

### 5. `push_to_hub.py` Import Order Bug
- **File:** `training/push_to_hub.py:6`
- `from huggingface_hub import HfApi` at top level crashes before try/except

### 6. Windows-Only Launcher
- `themis.cmd` only works on Windows
- No `themis.sh` for Linux/macOS

---

## Missing for PyPI Deployment

| Item | Status | Required |
|------|--------|----------|
| `LICENSE` file | ❌ Missing | Yes |
| `__init__.py` files | ❌ Missing | Yes |
| Fixed imports (relative) | ❌ Not done | Yes |
| `README.md` | ✅ Exists | Yes |
| `pyproject.toml` | ✅ Exists | Yes |
| Package structure | ⚠️ Partial | Yes |
| `eval/eval_set.json` | ❌ Missing | No (but needed for eval) |
| Model weights on HuggingFace | ❌ Not uploaded | Yes |
| Training data (`dataset.json`) | ❌ Not generated | Yes |
| Working training pipeline | ❌ Broken | Yes |
| Working scrapers | ⚠️ Kanoon broken | Yes |

---

## Effort Estimation to Complete

### Phase 1: Fix Critical Bugs (2-3 hours)
1. Create `LICENSE` file (MIT)
2. Add `__init__.py` to all packages
3. Fix bare imports → relative imports across all files
4. Fix Kanoon scraper `_get()` to accept `params`
5. Fix `finetune.py` to use `Dataset.from_list()` and proper `formatting_func`
6. Create `eval/eval_set.json` with 50 questions
7. Fix `push_to_hub.py` import order

### Phase 2: Generate Training Data (1-2 hours)
1. Run `themis scrape --law bns` (~395 sections, ~5-10 min)
2. Run `themis scrape --law bnss` (~530 sections, ~5-10 min)
3. Run `themis scrape --law cpa` (~120 sections, ~3 min)
4. Run `themis generate --no-api` (template mode, ~2-5 min)
5. Run `themis preprocess` (~1 min)

### Phase 3: Fine-Tune Model (4-6 hours on Kaggle T4)
1. Upload `dataset.json` to Kaggle
2. Run `finetune.py` with Unsloth
3. Push LoRA adapter to HuggingFace Hub

### Phase 4: Test End-to-End (1-2 hours)
1. `themis ask "What is BNS Section 118?"`
2. `themis chat` interactive session
3. `themis eval --verbose`
4. Verify citation accuracy > 80%

### Phase 5: PyPI Preparation (1-2 hours)
1. Fix all imports for package installation
2. Add entry points properly
3. Test `pip install -e .` works from clean environment
4. Build distribution: `python -m build`
5. Upload to PyPI: `twine upload dist/*`

**Total estimated effort: 9-15 hours**

---

## Current Pipeline Status

```
Scraping ──► Generation ──► Preprocessing ──► Training ──► Inference ──► Eval
  [✅]         [✅]            [✅]            [❌]         [⚠️]        [❌]
 India Code   Groq/Template   Clean/Dedup     Broken       No weights   No eval set
 Kanoon       Template OK     Alpaca format   SFTTrainer   CLI works    Metrics partial
 (broken)     (works)         (works)         crashes      (no model)
```

---

## What README Claims vs Reality

| README Claim | Reality |
|--------------|---------|
| `pip install themis-law` | ❌ Won't work — no `__init__.py`, broken imports |
| `themis ask "question"` | ⚠️ Shows help, but no model weights to run inference |
| `themis chat` | ⚠️ Shows help, but no model weights |
| `themis eval --verbose` | ❌ No `eval_set.json`, no model weights |
| ~2,800 instruction pairs | ❌ `dataset.json` doesn't exist yet |
| ~800 sections parsed | ❌ `data/scraper/raw/` is empty |
| ~500 judgments | ❌ Kanoon scraper is broken |
| Citation accuracy metric | ✅ Function exists |
| Refusal rate metric | ⚠️ Function exists but not wired into eval |
| Hallucination rate metric | ⚠️ Function exists but not wired into eval |
| IPC→BNS mapping | ❌ Field exists but never populated |
| Offline capable | ⚠️ CLI works offline, but needs model weights first |
| MIT License | ❌ LICENSE file missing |

---

## Recommendation

**Do not deploy to PyPI yet.** The project needs:

1. **Bug fixes** (imports, training script, Kanoon scraper)
2. **Missing files** (LICENSE, __init__.py, eval_set.json)
3. **Training data generation** (run the full pipeline)
4. **Model fine-tuning** (on Kaggle T4)
5. **End-to-end testing** (verify inference works)

Once these are complete, the project will be ready for PyPI as a **CLI tool** (not a library). Users will run `pip install themis-law` and use `themis ask` / `themis chat` commands.
