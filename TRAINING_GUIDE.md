# Task 1 & 2: Train Model on Kaggle + Push to HuggingFace

This guide walks you through every single click, paste, and command.

---

## What You're Doing

1. Upload `dataset.json` (1,939 training pairs) to Kaggle
2. Run the training script on a free T4 GPU (takes ~4-6 hours)
3. Download the trained LoRA adapter (~50MB)
4. Upload it to HuggingFace Hub so `themis ask` can download it

---

## Prerequisites

Before starting, make sure you have:
- A Kaggle account (free) → https://www.kaggle.com/account/login
- A HuggingFace account (free) → https://huggingface.co/join

---

## PART 1: Train on Kaggle

### Step 1: Get Your HuggingFace Token

You need this so the training script can download Mistral 7B.

1. Go to https://huggingface.co/settings/tokens
2. Click **"Create new token"**
3. Name it: `kaggle-training`
4. Role: **"Read"** (not write — we only need to download the base model)
5. Click **"Create token"**
6. **Copy the token** (starts with `hf_...`). You won't see it again.

### Step 2: Upload dataset.json to Kaggle

1. Go to https://www.kaggle.com/datasets
2. Click **"New Dataset"** (top right)
3. Click **"New empty dataset"**
4. Set:
   - **Title:** `themis-training-data`
   - **Slug:** `deshmukhdaniel/themis-training-data` (or whatever you prefer)
5. Click **"Files"** tab
6. Click **"Upload Files"**
7. Navigate to your project and select:
   ```
   D:\Vs Code\themis\themis\data\dataset.json
   ```
8. Click **"Save"** (top right)

### Step 3: Create the Kaggle Notebook

1. Go to https://www.kaggle.com/code
2. Click **"New Notebook"**
3. You'll see an empty notebook. We'll paste code into cells.

### Step 4: Set Up GPU

1. On the right sidebar, find **"Accelerator"**
2. Select **"GPU T4 x2"** (or just T4)
3. Click **"Turn on internet"** (so it can download packages)
4. The notebook will restart with GPU enabled.

### Step 5: Configure Secrets (HuggingFace Token)

1. On the right sidebar, find **"Secrets"** (key icon)
2. Click **"Add a new secret"**
3. Name: `HF_TOKEN`
4. Value: `hf_YourTokenHere` (the token from Step 1)
5. Click **"Add"**

### Step 6: Install Packages (Cell 1)

Click the first cell, paste this, and press **Shift+Enter**:

```python
!pip install unsloth
!pip install --upgrade transformers peft trl datasets
```

Wait for it to finish (~2 minutes). You'll see lots of output — that's normal.

### Step 7: Mount Your Dataset (Cell 2)

Click **"+" Code** to add a new cell. Paste:

```python
from kaggle_secrets import UserSecretsClient
secrets = UserSecretsClient()
hf_token = secrets.get_secret("HF_TOKEN")

import os
os.environ["HF_TOKEN"] = hf_token
print("HF_TOKEN set!")
```

Press **Shift+Enter**.

Then add another cell:

```python
import json, os

# Load dataset from Kaggle input
dataset_path = "/kaggle/input/themis-training-data/dataset.json"
with open(dataset_path, "r") as f:
    raw_data = json.load(f)

print(f"Loaded {len(raw_data)} training samples")
print(f"First instruction: {raw_data[0]['instruction'][:100]}...")
```

Press **Shift+Enter**. If you see "Loaded 1939 training samples", you're good.

### Step 8: Run Training (Cell 3 — the main one)

Add a new cell and paste this ENTIRE block:

```python
import json
from pathlib import Path

# ===== Load config =====
config = {
    "base_model": "mistralai/Mistral-7B-Instruct-v0.3",
    "lora": {
        "r": 16,
        "lora_alpha": 32,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "lora_dropout": 0.05,
        "bias": "none",
    },
    "training": {
        "epochs": 3,
        "batch_size": 2,
        "gradient_accumulation_steps": 4,
        "learning_rate": 2e-4,
        "warmup_ratio": 0.03,
        "max_seq_length": 2048,
        "fp16": True,
    },
}

# ===== Load base model with Unsloth =====
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import Dataset

print("=" * 60)
print("THEMIS Fine-tuning with Unsloth + LoRA")
print("=" * 60)

print(f"\nLoading base model: {config['base_model']}")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=config["base_model"],
    max_seq_length=config["training"]["max_seq_length"],
    dtype=None,
    load_in_4bit=True,
)

# ===== Add LoRA adapters =====
print("Adding LoRA adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r=config["lora"]["r"],
    target_modules=config["lora"]["target_modules"],
    lora_alpha=config["lora"]["lora_alpha"],
    lora_dropout=config["lora"]["lora_dropout"],
    bias=config["lora"]["bias"],
    use_gradient_checkpointing=True,
    random_state=42,
)

# ===== Format dataset =====
def format_instruction(sample):
    text = f"### Instruction:\n{sample['instruction']}\n\n"
    if sample.get("input"):
        text += f"### Input:\n{sample['input']}\n\n"
    text += f"### Response:\n{sample['output']}"
    return text

dataset = Dataset.from_list(raw_data)
print(f"Loaded {len(dataset)} training samples")

# ===== Create trainer =====
print("Setting up trainer...")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    formatting_func=format_instruction,
    max_seq_length=config["training"]["max_seq_length"],
    args=TrainingArguments(
        per_device_train_batch_size=config["training"]["batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        warmup_ratio=config["training"]["warmup_ratio"],
        num_train_epochs=config["training"]["epochs"],
        learning_rate=config["training"]["learning_rate"],
        fp16=config["training"]["fp16"],
        logging_steps=10,
        output_dir="outputs",
        optim="adamw_8bit",
        seed=42,
    ),
)

# ===== Train =====
print("\nStarting training...")
print("This will take approximately 4-6 hours on T4 GPU.")
stats = trainer.train()

# ===== Save adapter =====
output_dir = "./model/themis-lora"
Path(output_dir).mkdir(parents=True, exist_ok=True)
print(f"\nSaving adapter to {output_dir}")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)
print(f"Adapter saved to: {output_dir}")
print(f"Files saved:")
for f in Path(output_dir).iterdir():
    print(f"  {f.name} ({f.stat().st_size / 1024:.1f} KB)")
```

Press **Shift+Enter** and WAIT. This will take 4-6 hours. You'll see output like:

```
==((====))==  Unsloth - 2x faster free finetuning | Num GPUs = 1
   \\   /|    Num examples = 1,939 | Num Epochs = 3
O))\ /  |    Batch size per device = 2 | Gradient Accumulation steps = 4
\        /    Total batch size = 8 | Total steps = 726
 "-____-"     Number of trainable parameters = 41,943,040
```

**Don't close the browser tab.** Let it run.

### Step 9: Download the Trained Adapter

When training finishes:

1. On the right sidebar, click **"Output"** (folder icon)
2. You'll see `outputs/` folder — that's checkpoints, not the final adapter
3. Look for `model/themis-lora/` folder
4. Click the three dots **"..."** next to each file and download:
   - `adapter_config.json`
   - `adapter_model.bin`
   - `training_args.bin`
   - `tokenizer.json`
   - `tokenizer.model`
   - `special_tokens_map.json`
5. Put them all in one folder on your computer, e.g.:
   ```
   D:\Vs Code\themis\themis\model\themis-lora\
   ```

---

## PART 2: Push to HuggingFace

### Step 10: Create HuggingFace Repo

1. Go to https://huggingface.co/new
2. Fill in:
   - **Repository name:** `themis-mistral-7b-lora`
   - **Owner:** `deshmukhdaniel` (or your username)
   - **License:** MIT
   - **Check:** "This model is generated by code"
3. Click **"Create repository"**

### Step 11: Get Write Token

1. Go to https://huggingface.co/settings/tokens
2. Click **"Create new token"**
3. Name: `hf-push`
4. Role: **"Write"**
5. Click **"Create token"**
6. **Copy the token.**

### Step 12: Upload via Script

On your local machine, open a terminal and run:

```powershell
# Set your HuggingFace token
$env:HF_TOKEN = "hf_YourWriteTokenHere"

# Navigate to project
cd "D:\Vs Code\themis"

# Run the upload script
.\venv\Scripts\python.exe themis\training\push_to_hub.py
```

If it asks for confirmation, type `y` and press Enter.

You'll see:
```
Uploading adapter from D:\Vs Code\themis\themis\model\themis-lora to danieldeshmukh/themis-mistral-7b-lora...
Successfully uploaded to: https://huggingface.co/deshmukhdaniel/themis-mistral-7b-lora
```

### Step 13: Verify Upload

1. Go to https://huggingface.co/deshmukhdaniel/themis-mistral-7b-lora
2. You should see files:
   - `adapter_config.json`
   - `adapter_model.bin`
   - `tokenizer.json`
   - etc.

### Step 14: Test Inference

Back on your machine:

```powershell
cd "D:\Vs Code\themis"
.\venv\Scripts\python.exe -m themis.cli ask "What is BNS Section 118?"
```

If everything works, you'll see a legal response about punishment for causing hurt by dangerous weapons.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `CUDA out of memory` | Reduce batch_size from 2 to 1 in the training cell |
| `ModuleNotFoundError: unsloth` | Re-run the pip install cell |
| `HF token not valid` | Make sure token starts with `hf_` and has "Write" role |
| Training is slow | Make sure GPU accelerator is selected (right sidebar) |
| `adapter_model.bin` is 0KB | Training didn't complete — re-run the training cell |
| Push fails with 403 | Token needs "Write" role, not "Read" |
