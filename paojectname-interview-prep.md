# THEMIS — Interview Cheat Sheet

## 1. THE 30-SECOND PITCH

THEMIS is a fine-tuned LLM for Indian law. I took Mistral 7B, scraped India's Bare Acts (BNS, BNSS, BSA, IPC, RTI), generated 52,000 legal Q&A pairs, and trained a LoRA adapter on a free Kaggle T4 GPU. The result is a 7B-parameter model that can answer questions about Indian criminal law, cite specific section numbers, and refuse out-of-scope queries — all without a retrieval pipeline. It's published on PyPI (`themis-llm`) and HuggingFace. The core insight: for domain-specific law, you can bake knowledge into weights instead of retrieving it, which gives you reasoning speed and no runtime dependency on a vector database.

## 2. THE ARCHITECTURE, WHITEBOARD-READY

Draw a pipeline with four stages:

```
[Scrape] → [Generate] → [Train] → [Serve]
```

**Stage 1 — Scrape.** India Code scraper (`indiacode.py`) hits `indiacode.nic.in` via AJAX, pulls section text for BNS (358 sections), BNSS (531), BSA (170), IPC, RTI, CPA. Handles retries, rate limiting, custom User-Agent. 996 total sections scraped.

**Stage 2 — Generate.** Three generators (v1/v2/v3). v3 uses 11 question templates (definition, punishment, procedure, comparison, scenario, etc.) plus Groq API (Mixtral-8x7b) for LLM-generated pairs. IPC→BNS mapping (511 sections). Output: Alpaca-format JSON (`instruction`, `input`, `output`).

**Stage 3 — Train.** Unsloth + LoRA on Mistral 7B Instruct v0.3 (4-bit NF4 quantization). Rank 16, alpha 32, 4 attention targets. 2 epochs, LR 1e-4, 1,549 steps. 95/5 train/val split. Adapter pushed to HuggingFace.

**Stage 4 — Serve.** CLI (`themis ask/chat/info`) loads base model in 4-bit, attaches LoRA adapter via PEFT, generates with temperature 0.3. Also a Python SDK (`ThemisInference` singleton). Disclaimer auto-injected if missing.

Key design choice: the LoRA adapter is ~40MB. The base model is ~4GB. Users download the base model once, then swap adapters for different domain versions.

## 3. TECH STACK + WHY, ONE LINE EACH

- **Mistral 7B Instruct v0.3** — Best instruction-following open model at its size; Instruct variant already tuned for Q&A format.
- **LoRA (PEFT)** — Trains ~8.4M params out of 7.3B (~0.12%); full fine-tune would need 4× A100s.
- **Unsloth** — 2x faster LoRA training, 60% less VRAM; fit 7B model + 52k examples on a free T4.
- **4-bit NF4 quantization** — Fits 7B model in 4GB VRAM; without it, T4 can't load the model at all.
- **Groq API (Mixtral-8x7b)** — Free tier for synthetic data generation; no API key cost for training data.
- **Typer + Rich** — Declarative CLI with panels, spinners, progress bars; no argparse boilerplate.
- **BitsAndBytes** — 4-bit quantization implementation; only option that works with PEFT + Mistral.
- **HuggingFace Hub** — Model hosting with versioned adapters; `PeftModel.from_pretrained` pulls automatically.
- **Kaggle T4 (free)** — Zero-cost training; 16GB VRAM sufficient for 4-bit 7B + LoRA.
- **PyPI (`themis-llm`)** — `pip install themis-llm[cli]` gives instant CLI access.
- **GitHub Actions CI** — Ruff lint + pytest on every push; prevents regressions.

## 4. THE THREE HARDECISIONS

**1. LoRA over full fine-tuning.**
Full fine-tuning of Mistral 7B would require ~28GB VRAM (fp16) or 4× A100s. LoRA trains 0.12% of parameters (8.4M) and produces a 40MB adapter. Tradeoff: LoRA can't learn entirely new capabilities, only adapt existing ones. For a domain where the base model already "knows" how to reason and I just need it to reason about Indian law, this is the right call. Cost savings: $0 (Kaggle T4) vs ~$50/run on A100.

**2. Template-based + LLM-generated hybrid data.**
Pure templates are deterministic and verifiable but robotic. Pure LLM generation is diverse but hallucination-prone. I used 11 templates for ground-truth pairs (section definitions, punishments, procedures) and Groq API for scenario-based and comparative questions. The templates gave me a verified backbone; the LLM pairs added reasoning diversity. The IPC→BNS mapping (511 entries) was hand-verified.

**3. Retrieval grounding as a patch, not a redesign.**
After v5 training (loss 0.13 train / 0.98 val), I faced a choice: keep iterating on parametric training or add retrieval. I chose retrieval grounding — extract section references from user queries, look up actual section text, inject it as context before generation. This keeps the parametric model (no new training) while fixing hallucination for section-specific questions. Tradeoff: adds a runtime dependency (anchor tables) and doesn't help with reasoning questions, only factual ones.

## 5. ONE REAL FAILURE STORY

**The "Bangladesh National Standards" incident.**

v1 had 1,939 training pairs. When asked "What is BNS?" the model confidently answered "Bangladesh National Standards" — a foreign technical standards body. Not Indian criminal law at all.

**Diagnosis:** With only 1,939 examples, the model saw "BNS" too few times to anchor it to Indian law. It fell back on whatever "BNS" meant in its pretraining data (Mistral was trained on internet text where BNS likely refers to the standards body).

**Fix:** Scaled to 20,909 pairs (v2) — 10x more data. The model never confused BNS again.

**What I'd do differently:** I should have included abbreviation disambiguation pairs from the start. v2 added 21 explicit pairs like "BNS = Bharatiya Nyaya Sanhita, not Bangladesh National Standards." These are cheap to generate and prevent the failure entirely.

## 6. LIKELY GOTCHA QUESTIONS

**Q: Why LoRA over RAG?**
A: RAG retrieves; LoRA reasons. For legal Q&A, the model needs to synthesize across sections, compare provisions, explain tradeoffs — not just look up a passage. LoRA gives that reasoning. RAG is better for citation-heavy research (that's the HECTOR companion project).

**Q: How do you handle the overfitting problem?**
A: v2 hit loss 0.06-0.08 (memorization). I reduced epochs 3→2, halved LR 2e-4→1e-4, doubled dropout 0.05→0.10, added 95/5 validation split. Final v5: train 0.13, val 0.98 — gap shows residual overfitting, but the retrieval grounding compensates by injecting ground-truth section text.

**Q: What's your hallucination rate?**
A: v1 was ~60% on BNS queries. v5 grounding reduces it — tested on 3 cases, fixed 2. Full quantification is the next milestone. The `hallucination_check()` function cross-references cited section numbers against scraped valid sections.

**Q: Why Mistral over Llama or Phi?**
A: Mistral 7B Instruct v0.3 had the best instruction-following at its size when I started. The Instruct variant means I don't need to add a chat template — it already understands Q&A format. Llama 2 was closed-commercial-use at the time.

**Q: What does the eval harness actually measure?**
A: 4 metrics: citation accuracy (section number overlap with ground truth), ROUGE-L (text similarity), refusal rate (does it decline out-of-scope queries), hallucination rate (cited sections not in valid set). 13 unit tests cover config, device resolution, prompt formatting, metric functions, and preprocessing.

**Q: This is just fine-tuning — what's the hard part?**
A: The hard part was the iteration cycle. Each version failed differently: v1 underfit (too little data), v2 overfit (too many epochs), v3 still overfit (LR too high). Finding the right combination of epochs × LR × dropout × data scale across 5 versions required systematic diagnosis, not just tuning one knob.

**Q: What's missing / unfinished?**
A: Consumer Protection Act training data (anchor table exists but no QA pairs), Hindi bilingual support, full hallucination measurement across all acts, and the THEMIS-HECTOR hybrid router (unifying parametric + retrieval).

**Q: Why 4-bit quantization?**
A: T4 has 16GB VRAM. Mistral 7B in fp16 is ~14GB — barely fits with no room for training. In 4-bit NF4 it's ~4GB, leaving room for LoRA gradients and batch processing. BitsAndBytes implements this; `bnb_4bit_use_double_quant=True` compresses twice.

## 7. NUMBERS TO HAVE READY

- **52,170** training examples across 8 legal domains
- **358** BNS sections, **531** BNSS, **170** BSA, **107** CPA, **31** RTI scraped
- **511** IPC → BNS section mappings
- **1,549** training steps, **2** epochs
- **0.1314** final train loss, **0.9808** val loss
- **8.4M** LoRA trainable params out of **7.3B** total (~0.12%)
- **~40MB** adapter size
- **4-bit** NF4 quantization
- **13** unit tests, all passing
- **11** question templates
- **30** git commits
- **4** GitHub stars
- **v2.0.1** on PyPI (`themis-llm`)
- **13** Python source files, **~3,400** lines of code
- **$0** training cost (Kaggle free tier T4)
- **7** CLI commands: ask, chat, scrape, generate, preprocess, eval, info
- **2** scraping targets: India Code (indiacode.nic.in), Indian Kanoon
- **0.3** generation temperature (low for factual consistency)
