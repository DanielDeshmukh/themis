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

## v3: The 2-Epoch Fix That Wasn't Enough

**Date:** June 2026  
**Setup:** Same 20,909 training pairs as v2, **2 epochs** (5,228 total steps).  
**LoRA:** rank=16, alpha=32, targets=[q_proj, k_proj, v_proj, o_proj], dropout=0.05  
**Learning rate:** 2e-4  
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

**Training result:** Loss at steps 4220–4300 hovered between **0.059 and 0.094** — firmly in the overfitting zone (0.06–0.10) and fluctuating rather than converging cleanly. Training was stopped manually at step ~4300 to prevent further memorization.

**Root cause — insufficient fix:** Reducing epochs from 3 to 2 wasn't enough on its own. The learning rate (2e-4) was still aggressive enough to push the model into memorization territory within 2 epochs. Without a validation split, there was also no signal to detect the moment overfitting began — just the raw training loss, which kept falling without any counterweight.

**What didn't work:**
- 2 epochs alone didn't prevent overfitting when LR was still 2e-4
- No validation loss meant overfitting was only caught by eyeballing training loss thresholds
- lora_dropout=0.05 was too weak to regularize effectively at this LR

**Net assessment:** v3 made the right diagnosis (stop at 2 epochs) but applied too small a fix. The underlying learning rate was the bigger lever, and without eval loss tracking, there was no way to know *when* in the run the model crossed from learning to memorizing.

**Adapter:** `Daniel2503/themis-mistral-7b-lora-v3` on HuggingFace Hub.

---

## v4: Lower LR + Validation Split (In Progress)

**Date:** June 2026  
**Setup:** Same 20,909 training pairs, **2 epochs**, with a proper 95/5 train/eval split.  
**LoRA:** rank=16, alpha=32, targets=[q_proj, k_proj, v_proj, o_proj], dropout=**0.10**  
**Learning rate:** **1e-4** (halved from v3)  
**Warmup:** **5%** (up from 3%)  
**Max seq length:** 1024  
**Platform:** Kaggle T4  
**Notebook:** `notebooks/THEMIS_v4_Training.ipynb`

**Key changes from v3:**

| Parameter | v3 | v4 | Why |
|-----------|-----|-----|-----|
| **Learning rate** | **2e-4** | **1e-4** | Root cause of v3 overfitting — halved to slow memorization |
| **LoRA dropout** | **0.05** | **0.10** | Stronger regularization — forces more robust representations |
| **Warmup ratio** | **3%** | **5%** | Longer warmup prevents LR spike at start of training |
| **Validation split** | **None** | **5% holdout** | Eval loss tracked every 500 steps — detects overfitting as it happens |
| **Best checkpoint** | Last saved | **Lowest val loss** | `load_best_model_at_end=True` — saves the right checkpoint automatically |
| **Loss curve** | Not plotted | **Plotted with thresholds** | Train + eval loss on same chart with 0.10 / 0.25 reference lines |

**What we're testing:**
1. Does 1e-4 LR keep loss in the 0.15–0.25 healthy range?
2. Does doubling dropout reduce memorization without hurting domain grounding?
3. Does eval loss track close to train loss (no divergence = no overfitting)?
4. Do the v2 symptoms (repetition loops, regurgitation) stay gone?

**Expected outcome:** Training loss 0.15–0.25, eval loss within ~0.05 of train loss throughout, clean answers to both direct and rephrased questions.

**Status:** Notebook complete, pending training run.

**Adapter:** Will be pushed to `Daniel2503/themis-mistral-7b-lora-v4` on HuggingFace Hub.

---

## v5: Production Scale (Planned)

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

| Version | Data | Epochs | LR | Failure Mode | Fix |
|---------|------|--------|-----|--------------|-----|
| v1 | 1.9k | 3 | — | Underfitting — 60% hallucination, domain confusion | More data |
| v2 | 20.9k | 3 | 2e-4 | Overfitting — memorization, regurgitation loops | Fewer epochs |
| v3 | 20.9k | 2 | 2e-4 | Still overfitting — LR too high, no eval signal | Lower LR + validation |
| v4 | 20.9k | 2 | 1e-4 | *Pending* | — |
| v5 | 50–90k | TBD | TBD | — | — |

**Pattern:** Each failure came from a different extreme — or a fix that was right in direction but insufficient in magnitude. v1 had too little data. v2 had enough data but trained too long. v3 correctly cut the epochs but left the learning rate untouched, which was the bigger lever. The takeaway: overfitting isn't a single dial. It's the product of epochs × learning rate × regularization strength, and all three have to move together.

**The missing piece across v2 and v3:** No validation split. Without eval loss, there's no way to know *when* in training the model crosses from learning to memorizing — you're flying blind, reading training loss and hoping. v4 adds that signal. If eval loss tracks close to train loss throughout, the model is generalizing. If eval loss flattens or rises while train loss keeps falling, stop immediately and use the checkpoint before the divergence.

---

*Three failure modes, three fixes, one model that's getting closer to actually knowing Indian law.*