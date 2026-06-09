# THEMIS — Development Roadmap

## Vision

Make statutory law accessible to anyone with a terminal. Not as a replacement for lawyers, but as a first layer of orientation that helps people understand what laws exist, what they say, and what options they have.

---

## Version Timeline

### v1.0 — Core Legal Intelligence (Current Target)

**Goal:** Functional CLI that answers Indian legal questions with citations.

| Phase | Tasks | Status |
|-------|-------|--------|
| **Phase 1: Data Pipeline** | Scrape India Code Bare Acts, scrape Indian Kanoon judgments, generate synthetic Q&A pairs via Groq API (free) or template fallback, preprocess and format to Alpaca JSON | Pending |
| **Phase 2: Training** | Set up Unsloth training notebook for Kaggle, configure LoRA hyperparameters, run fine-tuning on T4 GPU, export LoRA adapter to HuggingFace Hub | Pending |
| **Phase 3: Inference** | Implement model loading with PEFT, build system prompt with legal domain instructions, build Rich CLI interface, implement ask/chat commands | Pending |
| **Phase 4: Evaluation** | Create 50-question held-out eval set, implement citation accuracy metric, implement ROUGE-L and refusal rate metrics, run eval and document results | Pending |
| **Phase 5: Packaging** | Create pyproject.toml for pip install, write CLI entry points, publish to PyPI, create installation documentation | Pending |

**Target Date:** TBD (depends on compute availability and data pipeline speed)

---

### v1.1 — Landmark Judgments

**Goal:** Add Supreme Court landmark judgment summaries to training data.

| Task | Description |
|------|-------------|
| Curate landmark judgments | Select 50-100 landmark SC judgments across criminal, civil, consumer domains |
| Generate summaries | Use Groq API to generate plain-language summaries of judgment holdings |
| Create Q&A pairs | Generate instruction-tuning pairs from judgment summaries |
| Re-fine-tune | Extend training with judgment-aware data |
| Update eval set | Add judgment-related evaluation questions |

**Dependencies:** v1.0 complete, judgment data source identified

---

### v1.2 — Hindi Language Support

**Goal:** Bilingual fine-tune enabling Hindi legal Q&A.

| Task | Description |
|------|-------------|
| Source Hindi legal texts | Hindi versions of BNS, BNSS, IPC from official gazette |
| Generate Hindi Q&A pairs | Groq-assisted bilingual pair generation |
| Bilingual fine-tune | Extend training with Hindi data |
| Evaluate Hindi accuracy | Create Hindi eval set, measure citation accuracy |
| Update CLI | Support language flag (--lang hi/en) |

**Dependencies:** v1.0 complete, Hindi legal text sources

---

### v2.0 — THEMIS-HECTOR Hybrid

**Goal:** Unified query routing between parametric (THEMIS) and retrieval (HECTOR) systems.

| Task | Description |
|------|-------------|
| Query classifier | Build classifier to detect if query needs retrieval or parametric reasoning |
| HECTOR integration | Connect to HECTOR's Qdrant vector database for deep research queries |
| Unified CLI | Single entry point dispatching to THEMIS or HECTOR |
| Response merging | Format responses consistently across both systems |
| Benchmark | Compare hybrid vs standalone performance |

**Dependencies:** v1.0 complete, HECTOR system available

---

### v2.1 — Web UI

**Goal:** Browser-based interface for non-technical users.

| Task | Description |
|------|-------------|
| FastAPI backend | REST API wrapping THEMIS inference |
| Next.js frontend | Chat-like interface with legal disclaimer |
| Authentication | Optional user accounts for conversation history |
| Deployment | Host on free tier (HuggingFace Spaces or Vercel) |

**Dependencies:** v2.0 complete

---

## Milestone Summary

| Version | Focus | Key Deliverable |
|---------|-------|-----------------|
| v1.0 | Core | CLI tool with BNS/IPC/BNSS Q&A |
| v1.1 | Judgments | Landmark SC judgment awareness |
| v1.2 | Hindi | Bilingual legal Q&A |
| v2.0 | Hybrid | THEMIS + HECTOR unified system |
| v2.1 | Web | Browser-based public interface |

---

## Critical Path

```
Data Pipeline → Training → Inference → Evaluation → Packaging → v1.0 Release
     │              │           │            │            │
     ▼              ▼           ▼            ▼            ▼
  Scrape +      Unsloth +    Model +     50 Q&A      pip install
  Generate      LoRA T4      CLI         eval set    themis-law
```

**Blockers:**
1. India Code scraping reliability (site may change)
2. Kaggle GPU availability (T4 quota limits)
3. API rate limits for synthetic generation
4. Model weight storage (HuggingFace Hub upload)

---

## Success Criteria (v1.0)

- [ ] CLI installs via `pip install themis-law`
- [ ] `themis ask "question"` returns formatted legal response
- [ ] Response cites correct BNS/IPC/BNSS section numbers
- [ ] Model refuses out-of-scope queries appropriately
- [ ] 50-question eval set achieves >70% citation accuracy
- [ ] Runs offline after installation
- [ ] No fabricated section numbers in outputs
- [ ] Response time <5 seconds on T4 GPU
