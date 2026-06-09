"""Synthetic Q&A pair generation using Claude API.

Generates instruction-tuning pairs from parsed Bare Act sections.
Each pair follows the Alpaca format: instruction, input, output.
"""

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import requests

from config import config


@dataclass
class QAPair:
    """A single Q&A training pair."""

    instruction: str
    input: str
    output: str
    source_section: str = ""
    source_act: str = ""


# Prompt template for Claude to generate Q&A pairs
QA_PROMPT_TEMPLATE = """You are generating training data for a legal AI model called THEMIS that helps Indian citizens understand law.

Given this legal section from {act_name}:

Section {section_number}: {section_title}

Full text:
{section_text}

Generate a realistic citizen question that this section answers, and provide a clear answer.

Requirements for the question:
- Should be a question an ordinary Indian citizen might ask
- Should relate specifically to what this section covers
- Use plain English, not legal jargon

Requirements for the answer:
1. Cite the specific section number (e.g., "Section {section_number} of the {act_name}")
2. Explain the legal position in simple, citizen-friendly language
3. Mention any related sections if relevant
4. If the section describes an offence, mention the punishment
5. If the section describes a right, explain how to exercise it
6. End with: "DISCLAIMER: This is legal orientation, not legal advice. Consult a qualified advocate for your specific situation."

Return ONLY a JSON object in this exact format (no markdown, no code blocks):
{{
    "instruction": "the question here",
    "input": "",
    "output": "the answer here"
}}"""


class SyntheticGenerator:
    """Generate synthetic Q&A pairs using Claude API."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Claude API key required. Set ANTHROPIC_API_KEY environment variable."
            )
        self.base_url = "https://api.anthropic.com/v1/messages"
        self.headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

    def generate_qa_pair(self, section: dict) -> QAPair | None:
        """Generate a single Q&A pair from a section."""
        prompt = QA_PROMPT_TEMPLATE.format(
            act_name=section.get("act_name", "Indian law"),
            section_number=section.get("section_number", ""),
            section_title=section.get("title", ""),
            section_text=section.get("text", "")[:2000],  # Limit text length
        )

        payload = {
            "model": "claude-3-haiku-20240307",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": prompt}],
        }

        try:
            resp = requests.post(
                self.base_url, headers=self.headers, json=payload, timeout=60
            )
            resp.raise_for_status()
            data = resp.json()

            # Extract response text
            content = data.get("content", [{}])
            if content:
                text = content[0].get("text", "")
                # Parse JSON from response
                # Remove markdown code blocks if present
                text = re.sub(r"```json\s*", "", text)
                text = re.sub(r"```\s*$", "", text)
                qa_data = json.loads(text.strip())

                return QAPair(
                    instruction=qa_data.get("instruction", ""),
                    input=qa_data.get("input", ""),
                    output=qa_data.get("output", ""),
                    source_section=section.get("section_number", ""),
                    source_act=section.get("act_name", ""),
                )
        except Exception as e:
            print(f"  Warning: Failed to generate Q&A for section {section.get('section_number')}: {e}")
        return None

    def generate_from_sections(self, sections: list[dict], max_pairs: int = None) -> list[QAPair]:
        """Generate Q&A pairs from a list of sections."""
        pairs = []
        limit = max_pairs or len(sections)

        for i, section in enumerate(sections[:limit]):
            if not section.get("text"):
                continue

            print(f"  Generating Q&A {i + 1}/{min(limit, len(sections))}...")
            qa_pair = self.generate_qa_pair(section)
            if qa_pair:
                pairs.append(qa_pair)
                print(f"    Generated: {qa_pair.instruction[:60]}...")

            # Rate limiting
            time.sleep(0.5)

        return pairs


def load_sections_from_raw(raw_dir: Path) -> list[dict]:
    """Load all sections from raw scraped data."""
    sections = []
    for json_file in raw_dir.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "sections" in data:
                    for sec in data["sections"]:
                        sec["act_name"] = data.get("name", "")
                        sections.append(sec)
        except Exception as e:
            print(f"Warning: Failed to load {json_file}: {e}")
    return sections


def generate_training_data(raw_dir: Path = None, output_dir: Path = None):
    """Generate all synthetic Q&A pairs from scraped data."""
    raw_dir = raw_dir or config.raw_dir
    output_dir = output_dir or config.synthetic_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load sections
    sections = load_sections_from_raw(raw_dir)
    print(f"Loaded {len(sections)} sections from raw data")

    if not sections:
        print("No sections found. Run indiacode.py scraper first.")
        return

    # Initialize generator
    try:
        generator = SyntheticGenerator()
    except ValueError as e:
        print(f"Error: {e}")
        print("Set ANTHROPIC_API_KEY environment variable to generate synthetic data.")
        return

    # Generate Q&A pairs
    pairs = generator.generate_from_sections(sections, max_pairs=1500)
    print(f"\nGenerated {len(pairs)} Q&A pairs")

    # Save output
    output_data = [
        {
            "instruction": p.instruction,
            "input": p.input,
            "output": p.output,
            "source_section": p.source_section,
            "source_act": p.source_act,
        }
        for p in pairs
    ]

    output_file = output_dir / "synthetic_qa.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Saved to {output_file}")
    return pairs


if __name__ == "__main__":
    generate_training_data()
