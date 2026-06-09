"""THEMIS fine-tuning script using Unsloth + LoRA.

Designed to run on Kaggle with T4 GPU (16GB VRAM).

Usage (in Kaggle notebook):
    !pip install unsloth
    !python finetune.py
"""

import json
from pathlib import Path

import yaml


def load_config(config_path: str = None) -> dict:
    """Load training config from YAML."""
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train():
    """Run LoRA fine-tuning with Unsloth."""
    try:
        from unsloth import FastLanguageModel
        from trl import SFTTrainer
        from transformers import TrainingArguments
        from datasets import load_dataset
    except ImportError:
        print("Error: Required packages not found.")
        print("Install with: pip install unsloth trl transformers datasets")
        return

    config = load_config()
    lora_config = config["lora"]
    training_config = config["training"]

    print("=" * 60)
    print("THEMIS Fine-tuning with Unsloth + LoRA")
    print("=" * 60)

    # Load base model
    print(f"\nLoading base model: {config['base_model']}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["base_model"],
        max_seq_length=training_config["max_seq_length"],
        dtype=None,
        load_in_4bit=True,
    )

    # Add LoRA adapters
    print("Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_config["r"],
        target_modules=lora_config["target_modules"],
        lora_alpha=lora_config["lora_alpha"],
        lora_dropout=lora_config["lora_dropout"],
        bias=lora_config["bias"],
        use_gradient_checkpointing=training_config.get("unsloth", {}).get(
            "use_gradient_checkpointing", True
        ),
        random_state=training_config.get("unsloth", {}).get("random_state", 42),
    )

    # Load dataset
    dataset_path = Path(__file__).parent.parent / "data" / "dataset.json"
    print(f"\nLoading dataset: {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # Format dataset for SFT
    def format_instruction(sample):
        text = f"### Instruction:\n{sample['instruction']}\n\n"
        if sample.get("input"):
            text += f"### Input:\n{sample['input']}\n\n"
        text += f"### Response:\n{sample['output']}"
        return text

    # Create trainer
    print("Setting up trainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field=None,  # We'll use a custom formatter
        max_seq_length=training_config["max_seq_length"],
        args=TrainingArguments(
            per_device_train_batch_size=training_config["batch_size"],
            gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
            warmup_ratio=training_config["warmup_ratio"],
            num_train_epochs=training_config["epochs"],
            learning_rate=training_config["learning_rate"],
            fp16=training_config.get("fp16", True),
            logging_steps=10,
            output_dir="outputs",
            optim="adamw_8bit",
            seed=training_config.get("unsloth", {}).get("random_state", 42),
        ),
    )

    # Train
    print("\nStarting training...")
    stats = trainer.train()

    # Save adapter
    output_dir = config["output"]["adapter_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"\nSaving adapter to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("\nTraining complete!")
    print(f"Adapter saved to: {output_dir}")
    print(f"Upload to HuggingFace Hub with push_to_hub.py")


if __name__ == "__main__":
    train()
