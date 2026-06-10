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

# THEMIS — Indian Legal Intelligence Engine

**The Parametric Legal Intelligence Engine for Indian Law**

*"Not retrieval. Not lookup. Pure legal reasoning, baked into weights."*

---

## Model Description

THEMIS is a domain-specific language model fine-tuned on Indian statutory law. It answers legal questions about the Bharatiya Nyaya Sanhita (BNS) 2023, Bharatiya Nagarik Suraksha Sanhita (BNSS) 2023, Indian Penal Code (IPC) 1860, Consumer Protection Act 2019, and Right to Information Act 2005.

This is a **LoRA adapter** that must be loaded on top of the base model `unsloth/mistral-7b-instruct-v0.3-bnb-4bit`.

---

## Training Details

| Parameter | Value |
|-----------|-------|
| Base Model | `unsloth/mistral-7b-instruct-v0.3-bnb-4bit` |
| Method | LoRA (Low-Rank Adaptation) |
| LoRA Rank | 8 |
| LoRA Alpha | 16 |
| Target Modules | `q_proj`, `v_proj` |
| Dropout | 0.05 |
| Epochs | 3 |
| Batch Size | 2 |
| Gradient Accumulation | 4 |
| Learning Rate | 2e-4 |
| LR Scheduler | Cosine |
| Warmup Ratio | 0.03 |
| Max Sequence Length | 512 |
| Optimizer | AdamW |
| FP16 | Yes (T4 compatible) |
| Training Platform | Kaggle (free T4 GPU) |
| Framework | Unsloth + HuggingFace PEFT |

---

## Dataset

**1,939 Alpaca-format instruction pairs** sourced from:

| Source | Type | Volume |
|--------|------|--------|
| India Code (indiacode.nic.in) | BNS, BNSS, BSA, Consumer Protection Act, RTI Act | ~800 sections |
| Indian Kanoon (indiankanoon.org) | Judgment summaries (criminal/civil/consumer) | ~500 judgments |
| Synthetic (templates + Groq API) | Q&A pairs from parsed Bare Act sections | ~639 pairs |

### Format

```json
{
  "instruction": "What is the punishment for theft under BNS?",
  "input": "",
  "output": "Under Section 303 of the Bharatiya Nyaya Sanhita (BNS) 2023..."
}
```

---

## How to Use

### With PEFT + Transformers

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    quantization_config=bnb_config,
    device_map="auto",
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
)

# Attach LoRA adapter
model = PeftModel.from_pretrained(base_model, "Daniel2503/themis-mistral-7b-lora")
model.eval()

# Format prompt (Alpaca style)
question = "What is the punishment for theft under BNS?"
prompt = f"### Instruction:\n{question}\n\n### Response:\n"

# Generate
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.3,
        top_p=0.9,
        repetition_penalty=1.1,
        do_sample=True,
    )

response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(response)
```

### With THEMIS CLI

```bash
pip install themis-law
themis ask "What is BNS Section 118?"
themis chat
```

---

## Example Outputs

**Q:** What is the punishment for theft under BNS?

**A:** Under Section 303 of the Bharatiya Nyaya Sanhita (BNS) 2023, whoever commits theft shall be punished with imprisonment of either description for a term which may extend to three years, or with fine which may extend to ten thousand rupees, or with both. This replaces Section 378 of the Indian Penal Code (IPC) 1860.

---

## Intended Use

- **Citizen legal orientation** — quick answers about Indian statutory law
- **Legal literacy** — understanding what laws exist and what they say
- **IPC to BNS mapping** — finding equivalent sections under the new code

## Limitations

- **Not a lawyer substitute.** Provides legal orientation, not legal advice.
- **Not trained on case law.** Only covers statutes, not judgments.
- **Not multilingual.** English only (Hindi on roadmap).
- **Not accurate for state-specific laws.** Stamp duty, tenancy, land records vary by state.
- **May hallucinate section numbers.** Always verify with a qualified advocate.

---

## Citation

```bibtex
@misc{themis2026,
  title={THEMIS: The Parametric Legal Intelligence Engine for Indian Law},
  author={Daniel Deshmukh},
  year={2026},
  howpublished={\\url{https://huggingface.co/Daniel2503/themis-mistral-7b-lora}},
}
```

---

## License

MIT License

---

*THEMIS — Greek goddess of law, justice, and order.*
*Because justice should not require a law degree to understand.*
