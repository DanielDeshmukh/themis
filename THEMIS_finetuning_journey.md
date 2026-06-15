# THEMIS Fine-Tuning Journey

A living log of every fine-tuning run of Mistral 7B Instruct v0.3 on Indian statutory law — what broke, why, and what came next.

---

## v1: Too Little Data, Too Much Confidence

**Date:** June 2026
**Setup:** ~1.9k training pairs covering Indian statutory law (IPC, BNS, and related codes).
**LoRA:** rank=8, alpha=16, targets=[q_proj, v_proj]
**Epochs:** 3
**Max seq length:** 512

**Result:** Roughly a 60% hallucination rate on held-out questions. With that little data, the model didn't have enough signal to actually internalize the statutory structure — so when asked about something outside its narrow training distribution, it confidently filled the gap with plausible-sounding nonsense rather than admitting uncertainty.

**The standout failure** (and the funniest one): asked about BNS — *Bharatiya Nyaya Sanhita*, India's 2023 replacement for the IPC — the model hallucinated the acronym as **"Bangladesh National Standards"**. A core piece of Indian criminal law, rebranded by the model as a foreign technical standards body. Embarrassing if it showed up in front of anyone evaluating the model, and genuinely hilarious in isolation — a perfect illustration of what happens when a model has seen an acronym too few times to anchor it to the right domain, and instead falls back on whatever other "BNS"-shaped thing exists in its pretraining data.

**Root cause:** Insufficient training pairs (1.9k) meant the model never built a stable internal representation of BNS-specific terminology, section numbering, or scope. It was pattern-matching on vibes, not law.

**Adapter:** `Daniel2503/themis-mistral-7b-lora` on HuggingFace Hub.

---

## v2: More Data, New Problem — Overfitting

**Date:** June 2026
**Setup:** Scaled training data up roughly 10x to 20,909 examples, 3 epochs (7,842 total steps), LoRA on Mistral 7B Instruct v0.3.
**LoRA:** rank=16, alpha=32, targets=[q_proj, k_proj, v_proj, o_proj], dropout=0.05
**Max seq length:** 1024
**Platform:** Kaggle T4 x2

**Training signal:** Loss dropped to the 0.06–0.08 range by the final epoch — on paper, a strong number.

**Result on inference testing:** Asked "What is the punishment for theft under the Bharatiya Nyaya Sanhita?", the model:

- Correctly identified Section 303 (Theft) — the BNS hallucination from v1 is gone, which is real progress.
- But then recited the **definition** of theft verbatim from the statute, never addressing the actual question (the *punishment*, which is a separate subsection).
- Appended a disclaimer block, then **repeated the same disclaimer verbatim a second time**, and got cut off mid-sentence ("If you...") at the 256-token limit.

**Root cause — overfitting:** A loss of 0.06–0.08 after 3 epochs on ~21k examples is consistent with the model memorizing surface patterns from the training set — statute text blocks, disclaimer boilerplate, response structure — rather than learning to *reason* about what's being asked and select the relevant portion of the law. The repetition loop on the disclaimer is a classic overfitting symptom: the model has seen that exact phrase often enough that it becomes a high-probability continuation it loops on, especially under sampling.

**Net assessment:** v2 fixed v1's domain-grounding problem (no more "Bangladesh National Standards") but introduced a precision/relevance problem — it knows *what* law exists but not reliably *which part* answers the question, and it has a tendency to regurgitate training artifacts instead of generating a focused response.

**Adapter:** `Daniel2503/themis-mistral-7b-lora-v2` on HuggingFace Hub.

---

## v3: The 2-Epoch Fix

**Date:** June 2026
**Setup:** Same 20,909 training pairs as v2, **2 epochs** (5,228 total steps).
**LoRA:** rank=16, alpha=32, targets=[q_proj, k_proj, v_proj, o_proj], dropout=0.05
**Max seq length:** 1024
**Platform:** Kaggle T4
**Notebook:** `notebooks/THEMIS_v3_Training.ipynb`

**Key changes from v2:**

| Parameter | v2 | v3 | Why |
|-----------|-----|-----|-----|
| Epochs | 3 | **2** | Prevents memorization (loss 0.06→0.08 overfitting) |
| Checkpoints | Lost on disconnect | **Saved every 500 steps** | Keeps last 3, allows rollback |
| Eval set | 50 template questions | **65 questions** (15 conversational added) | Tests generalization, not just template matching |
| Max seq length | 1024 | 1024 | Same — sufficient for current data |

**What we're testing:**

1. Does 2 epochs prevent the overfitting symptoms from v2?
2. Can the model handle rephrased/conversational questions (not just "What is the punishment for X under Y?")?
3. Does the repetition loop on disclaimers disappear?
4. Is the loss in a healthy range (0.15–0.25) instead of memorization territory (<0.1)?

**Expected outcome:** The model should retain v2's domain grounding (correct section identification) while eliminating the regurgitation and repetition problems. A loss in the 0.15–0.25 range would indicate genuine learning rather than memorization.

**Status:** Notebook created, pending training run when GPU quota resets.

**Adapter:** Will be pushed to `Daniel2503/themis-mistral-7b-lora-v3` on HuggingFace Hub.

---

## v4: Production Scale (Planned)

**Target:** 50,000–90,000 training pairs
**LoRA:** rank=32, alpha=64, targets=[q, k, v, o, gate_proj, up_proj, down_proj]
**Max seq length:** 2048
**Platform:** RunPod A100 (40GB)

**Data sources:**
- Full India Code corpus (all central acts)
- Indian Kanoon top 1,000 judgment summaries
- Complete IPC → BNS transition mapping (all 511 sections)
- Hindi language support (bilingual fine-tune)

**Success criteria:** Citation accuracy >85%, hallucination rate <10%.

---

## Lessons Learned

| Version | Data | Epochs | Failure Mode | Fix |
|---------|------|--------|--------------|-----|
| v1 | 1.9k | 3 | Underfitting — 60% hallucination, domain confusion | More data |
| v2 | 20.9k | 3 | Overfitting — memorization, regurgitation loops | Fewer epochs |
| v3 | 20.9k | 2 | *Pending* | — |
| v4 | 50-90k | TBD | — | — |

**Pattern:** Each failure mode came from a different extreme. v1 had too little data to learn anything real. v2 had enough data but trained too long, memorizing surface patterns instead of internalizing legal reasoning. The fix isn't always "more" — sometimes it's "less, but better."

---

*Two failure modes, two data regimes, two opposite fixes — one model that's getting closer to actually knowing Indian law.*
