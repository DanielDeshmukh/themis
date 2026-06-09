"""Synthetic Q&A pair generation using free APIs.

Primary: Groq API (free, fast, Mixtral/Llama3)
Fallback: Template-based generation (no API needed)

Generates instruction-tuning pairs from parsed Bare Act sections.
Each pair follows the Alpaca format: instruction, input, output.
"""

import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path

import requests

from ...config import config


@dataclass
class QAPair:
    """A single Q&A training pair."""

    instruction: str
    input: str
    output: str
    source_section: str = ""
    source_act: str = ""


# Prompt template for LLM to generate Q&A pairs
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


# =============================================================================
# Template-based generator (no API needed)
# =============================================================================

# Question templates organized by section title keywords
QUESTION_TEMPLATES = {
    "punishment": [
        "What is the punishment for {title_lower}?",
        "How is {title_lower} punished under {act_name}?",
        "What penalty does the law prescribe for {title_lower}?",
    ],
    "definition": [
        "What does {act_name} mean by '{title}'?",
        "How is '{title}' defined in {act_name}?",
        "What is the legal definition of {title_lower}?",
    ],
    "right": [
        "What are my rights regarding {title_lower}?",
        "Can I {title_lower} under {act_name}?",
        "How do I exercise my right to {title_lower}?",
    ],
    "offence": [
        "Is {title_lower} a crime under {act_name}?",
        "What happens if someone commits {title_lower}?",
        "What are the consequences of {title_lower}?",
    ],
    "procedure": [
        "What is the procedure for {title_lower}?",
        "How do I file for {title_lower}?",
        "What steps should I follow for {title_lower}?",
    ],
    "default": [
        "What does Section {section_number} of {act_name} say?",
        "Explain Section {section_number} of {act_name} in simple terms.",
        "What should I know about {title_lower} under {act_name}?",
    ],
}


def categorize_section(title: str, text: str) -> str:
    """Categorize a section based on its title and content."""
    combined = (title + " " + text).lower()

    if any(kw in combined for kw in ["punish", "imprisonment", "fine", "death", "rigorous"]):
        return "punishment"
    elif any(kw in combined for kw in ["defin", "meaning", "interpretation"]):
        return "definition"
    elif any(kw in combined for kw in ["right", "entitled", "may claim", "shall be entitled"]):
        return "right"
    elif any(kw in combined for kw in ["offence", "crime", "stolen", "fraud", "cheating", "murder", "theft"]):
        return "offence"
    elif any(kw in combined for kw in ["procedure", "file", "complaint", "application", "form"]):
        return "procedure"
    return "default"


def generate_question_template(section: dict) -> QAPair:
    """Generate a Q&A pair using templates (no API needed)."""
    section_no = section.get("section_number", "")
    title = section.get("title", "this provision")
    text = section.get("text", "")
    act_name = section.get("act_name", "Indian law")

    category = categorize_section(title, text)
    templates = QUESTION_TEMPLATES.get(category, QUESTION_TEMPLATES["default"])
    template = random.choice(templates)

    title_lower = title.lower().rstrip(".")
    instruction = template.format(
        title=title,
        title_lower=title_lower,
        section_number=section_no,
        act_name=act_name,
    )

    # Build answer from section text
    clean_text = re.sub(r"\s+", " ", text).strip()
    if len(clean_text) > 500:
        clean_text = clean_text[:500] + "..."

    output = (
        f"Under Section {section_no} of the {act_name} ({title}):\n\n"
        f"{clean_text}\n\n"
        f"This section falls under the {act_name} and provides guidance on {title_lower}.\n\n"
        f"DISCLAIMER: This is legal orientation, not legal advice. "
        f"Consult a qualified advocate for your specific situation."
    )

    return QAPair(
        instruction=instruction,
        input="",
        output=output,
        source_section=section_no,
        source_act=act_name,
    )


# =============================================================================
# Groq API generator (free, fast)
# =============================================================================

class GroqGenerator:
    """Generate synthetic Q&A pairs using Groq API (free tier)."""

    BASE_URL = "https://api.groq.com/openai/v1/chat/completions"
    MODEL = "mixtral-8x7b-32768"  # Free on Groq

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Groq API key required. Get free key at https://console.groq.com\n"
                "Set GROQ_API_KEY environment variable."
            )
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def generate_qa_pair(self, section: dict) -> QAPair | None:
        """Generate a single Q&A pair from a section."""
        prompt = QA_PROMPT_TEMPLATE.format(
            act_name=section.get("act_name", "Indian law"),
            section_number=section.get("section_number", ""),
            section_title=section.get("title", ""),
            section_text=section.get("text", "")[:2000],
        )

        payload = {
            "model": self.MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 1024,
        }

        try:
            resp = requests.post(
                self.BASE_URL, headers=self.headers, json=payload, timeout=60
            )
            resp.raise_for_status()
            data = resp.json()

            text = data["choices"][0]["message"]["content"]
            # Parse JSON from response
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
            print(f"  Warning: Groq generation failed for section {section.get('section_number')}: {e}")
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

            # Rate limiting (Groq is fast, but be polite)
            time.sleep(0.3)

        return pairs


# =============================================================================
# Main generation pipeline
# =============================================================================

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


def generate_training_data(raw_dir: Path = None, output_dir: Path = None, use_api: bool = True):
    """Generate all synthetic Q&A pairs from scraped data.

    Args:
        raw_dir: Directory with scraped JSON files
        output_dir: Directory to save generated Q&A pairs
        use_api: If True, try Groq API first; if False or fails, use templates
    """
    raw_dir = raw_dir or config.raw_dir
    output_dir = output_dir or config.synthetic_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load sections
    sections = load_sections_from_raw(raw_dir)
    print(f"Loaded {len(sections)} sections from raw data")

    if not sections:
        print("No sections found. Run indiacode.py scraper first.")
        return

    pairs = []

    # Try Groq API first
    if use_api:
        try:
            generator = GroqGenerator()
            print("\nUsing Groq API (free tier) for generation...")
            pairs = generator.generate_from_sections(sections, max_pairs=1500)
            print(f"Generated {len(pairs)} pairs via Groq API")
        except ValueError as e:
            print(f"\n{e}")
            print("Falling back to template-based generation...")
        except Exception as e:
            print(f"\nGroq API error: {e}")
            print("Falling back to template-based generation...")

    # Fallback to templates
    if not pairs:
        print("\nUsing template-based generation (no API needed)...")
        random.seed(42)
        for i, section in enumerate(sections):
            if not section.get("text"):
                continue
            qa_pair = generate_question_template(section)
            pairs.append(qa_pair)
            if (i + 1) % 100 == 0:
                print(f"  Generated {i + 1}/{len(sections)} pairs...")
        print(f"Generated {len(pairs)} pairs via templates")

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

    print(f"\nSaved to {output_file}")
    return pairs


if __name__ == "__main__":
    import sys
    use_api = "--no-api" not in sys.argv
    generate_training_data(use_api=use_api)
