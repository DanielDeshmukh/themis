"""Data preprocessing pipeline.

Cleans, deduplicates, and merges all data sources into final
Alpaca-format training dataset.
"""

import json
import re
from pathlib import Path

from ..config import config


def clean_text(text: str) -> str:
    """Clean and normalize text content."""
    if not text:
        return ""

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    # Fix common encoding issues
    text = text.replace("\u2019", "'")
    text = text.replace("\u2018", "'")
    text = text.replace("\u201c", '"')
    text = text.replace("\u201d", '"')
    text = text.replace("\u2013", "-")
    text = text.replace("\u2014", "-")

    return text


def validate_section(section: dict) -> bool:
    """Validate that a section has required fields and correct format."""
    if not section.get("section_number"):
        return False
    if not section.get("title"):
        return False
    if not section.get("text") or len(section["text"]) < 20:
        return False
    return True


def load_json_files(directory: Path) -> list[dict]:
    """Load all JSON files from a directory."""
    data = []
    for json_file in directory.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                file_data = json.load(f)
                if isinstance(file_data, list):
                    data.extend(file_data)
                elif isinstance(file_data, dict):
                    data.append(file_data)
        except Exception as e:
            print(f"Warning: Failed to load {json_file}: {e}")
    return data


def sections_to_alpaca(sections: list[dict]) -> list[dict]:
    """Convert raw sections to Alpaca instruction format."""
    pairs = []

    for section in sections:
        if not validate_section(section):
            continue

        act_name = section.get("act_name", "Indian law")
        section_no = section.get("section_number", "")
        title = section.get("title", "")
        text = clean_text(section.get("text", ""))

        if not text:
            continue

        # Create a Q&A pair from the section
        instruction = f"What does Section {section_no} of the {act_name} say about {title.lower().rstrip('.')}?"
        output = (
            f"Section {section_no} of the {act_name} states:\n\n"
            f"{text}\n\n"
            f"DISCLAIMER: This is legal orientation, not legal advice. "
            f"Consult a qualified advocate for your specific situation."
        )

        pairs.append({
            "instruction": instruction,
            "input": "",
            "output": output,
        })

    return pairs


def deduplicate(pairs: list[dict]) -> list[dict]:
    """Remove duplicate Q&A pairs based on instruction similarity."""
    seen = set()
    unique = []

    for pair in pairs:
        # Normalize instruction for comparison
        normalized = pair["instruction"].lower().strip()
        normalized = re.sub(r"[^\w\s]", "", normalized)

        if normalized not in seen:
            seen.add(normalized)
            unique.append(pair)

    return unique


def merge_datasets(raw_dir: Path, synthetic_dir: Path) -> list[dict]:
    """Merge all data sources into single dataset."""
    all_pairs = []

    # Load scraped sections and convert to Alpaca format
    print("Loading scraped sections...")
    raw_data = load_json_files(raw_dir)
    sections = []
    for item in raw_data:
        if "sections" in item:
            for sec in item["sections"]:
                sec["act_name"] = item.get("name", "")
                sections.append(sec)
    section_pairs = sections_to_alpaca(sections)
    print(f"  Converted {len(section_pairs)} sections to Q&A pairs")
    all_pairs.extend(section_pairs)

    # Load synthetic Q&A pairs
    print("Loading synthetic Q&A pairs...")
    synthetic_data = load_json_files(synthetic_dir)
    for item in synthetic_data:
        if "instruction" in item and "output" in item:
            all_pairs.append({
                "instruction": item["instruction"],
                "input": item.get("input", ""),
                "output": item["output"],
            })
    print(f"  Loaded {len(synthetic_data)} synthetic pairs")

    return all_pairs


def create_eval_set(pairs: list[dict], eval_size: int = 50) -> tuple[list[dict], list[dict]]:
    """Split dataset into train and eval sets."""
    # Ensure we have enough pairs
    if len(pairs) < eval_size:
        eval_size = max(10, len(pairs) // 10)

    # Shuffle deterministically
    import random
    random.seed(42)
    shuffled = pairs.copy()
    random.shuffle(shuffled)

    eval_set = shuffled[:eval_size]
    train_set = shuffled[eval_size:]

    return train_set, eval_set


def preprocess_pipeline():
    """Run the full preprocessing pipeline."""
    raw_dir = config.raw_dir
    synthetic_dir = config.synthetic_dir

    print("=" * 60)
    print("THEMIS Data Preprocessing Pipeline")
    print("=" * 60)

    # Merge datasets
    all_pairs = merge_datasets(raw_dir, synthetic_dir)
    print(f"\nTotal pairs before dedup: {len(all_pairs)}")

    # Deduplicate
    unique_pairs = deduplicate(all_pairs)
    print(f"After deduplication: {len(unique_pairs)}")

    if not unique_pairs:
        print("No training data found. Run scrapers first.")
        return

    # Split into train/eval
    train_set, eval_set = create_eval_set(unique_pairs)
    print(f"Train set: {len(train_set)}")
    print(f"Eval set: {len(eval_set)}")

    # Save datasets
    output_dir = config.data_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_file = output_dir / "dataset.json"
    with open(train_file, "w", encoding="utf-8") as f:
        json.dump(train_set, f, indent=2, ensure_ascii=False)
    print(f"Saved training dataset to {train_file}")

    eval_file = config.eval_dir / "eval_set.json"
    config.eval_dir.mkdir(parents=True, exist_ok=True)
    with open(eval_file, "w", encoding="utf-8") as f:
        json.dump(eval_set, f, indent=2, ensure_ascii=False)
    print(f"Saved eval set to {eval_file}")

    # Print stats
    print("\n" + "=" * 60)
    print("Dataset Statistics")
    print("=" * 60)
    print(f"Total unique pairs: {len(unique_pairs)}")
    print(f"Training pairs: {len(train_set)}")
    print(f"Evaluation pairs: {len(eval_set)}")
    print(f"Avg instruction length: {sum(len(p['instruction']) for p in unique_pairs) // len(unique_pairs)} chars")
    print(f"Avg output length: {sum(len(p['output']) for p in unique_pairs) // len(unique_pairs)} chars")


if __name__ == "__main__":
    preprocess_pipeline()
