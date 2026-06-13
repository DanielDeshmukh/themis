# THEMIS Fine-Tuning Journey: v1 → v2

A log of two fine-tuning runs of Mistral 7B Instruct v0.3 on Indian statutory law, what broke, why, and what's next.

---

## v1: Too Little Data, Too Much Confidence

**Setup:** ~1.9k training pairs covering Indian statutory law (IPC, BNS, and related codes).

**Result:** Roughly a 60% hallucination rate on held-out questions. With that little data, the model didn't have enough signal to actually internalize the statutory structure — so when asked about something outside its narrow training distribution, it confidently filled the gap with plausible-sounding nonsense rather than admitting uncertainty.

**The standout failure** (and the funniest one): asked about BNS — *Bharatiya Nyaya Sanhita*, India's 2023 replacement for the IPC — the model hallucinated the acronym as **"Bangladesh National Standards"**. A core piece of Indian criminal law, rebranded by the model as a foreign technical standards body. Embarrassing if it showed up in front of anyone evaluating the model, and genuinely hilarious in isolation — a perfect illustration of what happens when a model has seen an acronym too few times to anchor it to the right domain, and instead falls back on whatever other "BNS"-shaped thing exists in its pretraining data.

**Root cause:** Insufficient training pairs (1.9k) meant the model never built a stable internal representation of BNS-specific terminology, section numbering, or scope. It was pattern-matching on vibes, not law.

---

## v2: More Data, New Problem — Overfitting

**Setup:** Scaled training data up roughly 10x to 20,909 examples, 3 epochs (7,842 total steps), LoRA on Mistral 7B Instruct v0.3.

**Training signal:** Loss dropped to the 0.06–0.08 range by the final epoch — on paper, a strong number.

**Result on inference testing:** Asked "What is the punishment for theft under the Bharatiya Nyaya Sanhita?", the model:

- Correctly identified Section 303 (Theft) — the BNS hallucination from v1 is gone, which is real progress.
- But then recited the **definition** of theft verbatim from the statute, never addressing the actual question (the *punishment*, which is a separate subsection).
- Appended a disclaimer block, then **repeated the same disclaimer verbatim a second time**, and got cut off mid-sentence ("If you...") at the 256-token limit.

**Root cause — overfitting:** A loss of 0.06–0.08 after 3 epochs on ~21k examples is consistent with the model memorizing surface patterns from the training set — statute text blocks, disclaimer boilerplate, response structure — rather than learning to *reason* about what's being asked and select the relevant portion of the law. The repetition loop on the disclaimer is a classic overfitting symptom: the model has seen that exact phrase often enough that it becomes a high-probability continuation it loops on, especially under sampling.

**Net assessment:** v2 fixed v1's domain-grounding problem (no more "Bangladesh National Standards") but introduced a precision/relevance problem — it knows *what* law exists but not reliably *which part* answers the question, and it has a tendency to regurgitate training artifacts instead of generating a focused response.

---

## Where Things Stand

- v2 adapter is pushed to HF (`Daniel2503/themis-mistral-7b-lora-v2`) but **not yet confirmed as an upgrade over v1** — pending a wider battery of test questions once GPU quota resets.
- Hypothesis: 3 epochs on this dataset size is too many. An earlier checkpoint (if saved) or a fresh 2-epoch run is the likely fix.
- Next steps: run conversational/rephrased questions (not just "what is the punishment for X under Y" — the exact phrasing the training data probably overuses), compare against v1 and any earlier checkpoints, and decide whether to retrain at 2 epochs.

---

*Two very different failure modes from two very different data regimes — one model that didn't know enough to be right, and one that knows plenty but can't always tell what's relevant. Both are fixable; neither is starting from zero.*
