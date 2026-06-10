---
language:
  - en
tags:
  - indian-law
  - bns
  - bnss
  - ipc
  - legal-ai
  - lora
  - mistral
  - fine-tuned
library_name: peft
pipeline_tag: text-generation
base_model: unsloth/mistral-7b-instruct-v0.3-bnb-4bit
---

# THEMIS v1 — Indian Legal Intelligence Engine (Proof of Pipeline)

**The Parametric Legal Intelligence Engine for Indian Law**

> **Status:** v1 trained | v2 in progress
> **Honest assessment:** This adapter proves the fine-tuning pipeline works. It does NOT yet produce reliable legal knowledge. See [Limitations](#known-limitations-v1) below.

---

## What This Is

A LoRA adapter fine-tuned on **1,939 Indian legal Q&A pairs** (BNS 2023, BNSS 2023, IPC, Consumer Protection Act, RTI Act). Must be loaded on top of `unsloth/mistral-7b-instruct-v0.3-bnb-4bit`.

## What This Demonstrates

- End-to-end fine-tuning pipeline: data scraping → synthetic generation → LoRA training (Unsloth/Kaggle T4) → HuggingFace deployment
- Instruction-following behavior in legal assistant style
- Correctly trained disclaimer behavior
- Partially learned response structure (citations, recommendations)

## What This Does NOT Demonstrate

- Accurate section number citation (~60% hallucination rate on BNS-specific queries)
- BNS abbreviation recognition (model confuses "BNS" with unrelated expansions)
- Deep statutory knowledge (1,939 pairs was insufficient for domain grounding)
- IPC → BNS section mapping

**Root cause:** Mistral 7B has near-zero BNS 2023 pretraining knowledge — BNS was enacted Dec 2023, at/after Mistral's training cutoff. The fine-tune taught style, not substance.

---

## Training Details

| Parameter | Value |
|-----------|-------|
| Base Model | `unsloth/mistral-7b-instruct-v0.3-bnb-4bit` |
| LoRA Rank | 8 |
| LoRA Alpha | 16 |
| Target Modules | `q_proj`, `v_proj` |
| Epochs | 3 |
| Batch Size | 1 |
| Gradient Accumulation | 8 |
| Learning Rate | 2e-4 |
| Max Sequence Length | 512 |
| Training Pairs | 1,939 |
| Platform | Kaggle T4 (free) |
| Framework | Unsloth + PEFT |

---

## How to Use

### With PEFT + Transformers

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained("unsloth/mistral-7b-instruct-v0.3-bnb-4bit")
base_model = AutoModelForCausalLM.from_pretrained(
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    quantization_config=bnb_config,
    device_map="auto",
)

model = PeftModel.from_pretrained(base_model, "Daniel2503/themis-mistral-7b-lora")
model.eval()

# Use FULL ACT NAMES for best results (e.g., "Bharatiya Nyaya Sanhita" not "BNS")
prompt = "### Instruction:\nWhat is the punishment for theft under the Bharatiya Nyaya Sanhita?\n\n### Response:\n"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.3)

response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(response)
```

---

## Known Limitations (v1)

- **Section hallucination** — ~60% of BNS-specific queries contain fabricated section numbers
- **Abbreviation confusion** — "BNS" not recognized; use full name "Bharatiya Nyaya Sanhita"
- **Insufficient training data** — 1,939 pairs teaches style, not statutory content
- **No case law** — statutes only, no judgments
- **English only** — Hindi on roadmap
- **State-specific laws not covered**

### For Best Results

- Use full act names: "Bharatiya Nyaya Sanhita" not "BNS"
- Ask general legal questions, not specific section numbers
- Treat output as orientation, never as authoritative legal reference

---

## v2 Roadmap

| Parameter | v1 (current) | v2 (next) |
|-----------|-------------|-----------|
| Training pairs | 1,939 | 10,000–15,000 |
| LoRA rank | 8 | 16 |
| Target modules | q_proj, v_proj | q,k,v,o proj |
| Sequence length | 512 | 1,024 |
| Expected citation accuracy | ~40% | >70% |

**Success criteria:** Model correctly identifies BNS as Bharatiya Nyaya Sanhita and cites accurate section numbers on 70%+ of criminal law queries.

---

## Citation

```bibtex
@misc{themis2026,
  title={THEMIS: Parametric Legal Intelligence Engine for Indian Law},
  author={Daniel Deshmukh},
  year={2026},
  howpublished={\url{https://huggingface.co/Daniel2503/themis-mistral-7b-lora}},
}
```

---

## License

MIT License

---

*THEMIS v1 proves the pipeline works. v2 will make the model actually know Indian law.*
