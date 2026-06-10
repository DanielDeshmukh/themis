"""THEMIS v2 data generation — scale from 1,939 to 10-15k pairs.

Strategy:
1. Multiple Q&A pairs per section (3-5 questions per section using varied templates)
2. IPC → BNS section mapping pairs (key transitions)
3. BNS abbreviation disambiguation pairs
4. Consumer protection + RTI + contract + property law coverage
"""

import json
import random
import re
from pathlib import Path

from ...config import config

# IPC to BNS section mapping — key transitions only
IPC_BNS_MAPPING = {
    "101": "Section 103 (Murder)",
    "102": "Section 104 (Causing death by negligence)",
    "103": "Section 103 (Murder)",
    "105": "Section 105 (Culpable homicide not amounting to murder)",
    "106": "Section 106 (Abetment of suicide)",
    "109": "Section 108 (Abetment of murder)",
    "111": "Section 110 (Criminal conspiracy)",
    "114": "Section 113 (Unlawful assembly armed with deadly weapon)",
    "143": "Section 139 (Kidnapping from India)",
    "144": "Section 140 (Kidnapping from lawful guardianship)",
    "145": "Section 141 (Abduction)",
    "147": "Section 143 (Rioting)",
    "148": "Section 144 (Rioting armed with deadly weapon)",
    "149": "Section 145 (Affray)",
    "299": "Section 100 (Culpable homicide)",
    "300": "Section 101 (Murder)",
    "302": "Section 101 (Murder)",
    "303": "Section 105 (Culpable homicide not amounting to murder)",
    "304": "Section 105 (Culpable homicide not amounting to murder)",
    "306": "Section 108 (Abetment of suicide)",
    "307": "Section 109 (Attempt to murder)",
    "319": "Section 117 (Hurt)",
    "320": "Section 118 (Grievous hurt)",
    "323": "Section 121 (Voluntarily causing hurt)",
    "324": "Section 122 (Voluntarily causing hurt by dangerous weapons)",
    "325": "Section 123 (Voluntarily causing grievous hurt)",
    "326": "Section 124 (Voluntarily causing grievous hurt by dangerous weapons)",
    "354": "Section 74 (Assault on woman to outrage modesty)",
    "375": "Section 63 (Rape)",
    "376": "Section 64 (Punishment for rape)",
    "378": "Section 303 (Theft)",
    "379": "Section 304 (Punishment for theft)",
    "380": "Section 305 (Theft in dwelling house)",
    "383": "Section 308 (Extortion)",
    "384": "Section 309 (Punishment for extortion)",
    "390": "Section 315 (Robbery)",
    "391": "Section 316 (Dacoity)",
    "392": "Section 317 (Punishment for robbery)",
    "395": "Section 320 (Punishment for dacoity)",
    "403": "Section 328 (Dishonest misappropriation of property)",
    "405": "Section 316 (Criminal breach of trust)",
    "406": "Section 317 (Punishment for criminal breach of trust)",
    "415": "Section 138 (Cheating)",
    "417": "Section 140 (Punishment for cheating)",
    "420": "Section 143 (Cheating and inducing delivery of property)",
    "425": "Section 148 (Mischief)",
    "441": "Section 164 (Criminal trespass)",
    "442": "Section 165 (House trespass)",
    "447": "Section 170 (Punishment for criminal trespass)",
    "448": "Section 171 (Punishment for house trespass)",
    "452": "Section 175 (House-trespass after preparation for hurt)",
    "463": "Section 186 (Forgery)",
    "465": "Section 188 (Punishment for forgery)",
    "467": "Section 190 (Forgery of valuable security or will)",
    "468": "Section 191 (Forgery for purpose of cheating)",
    "497": None,
    "498": "Section 85 (Cruelty by husband or his relatives)",
    "500": "Section 356 (Defamation)",
    "503": "Section 351 (Criminal intimidation)",
    "504": "Section 352 (Intentional insult to provoke breach of peace)",
    "505": "Section 353 (Statements conducing to public mischief)",
    "506": "Section 351 (Punishment for criminal intimidation)",
    "509": "Section 354 (Insult to modesty of woman)",
    "511": "Section 61 (Punishment for attempt to commit offences)",
}

# Abbreviation disambiguation pairs
ABBREVIATION_PAIRS = [
    {
        "instruction": "What does BNS stand for in Indian law?",
        "input": "",
        "output": "BNS stands for the Bharatiya Nyaya Sanhita, 2023. It is the new criminal code of India that replaces the Indian Penal Code (IPC) 1860. The BNS was enacted on December 25, 2023, and came into effect on July 1, 2024. It contains 358 sections covering offences against the state, public tranquility, human body, property, and other criminal matters.\n\nDISCLAIMER: This is legal orientation, not legal advice. Consult a qualified advocate for your specific situation.",
    },
    {
        "instruction": "What does BNSS mean in Indian law?",
        "input": "",
        "output": "BNSS stands for the Bharatiya Nagarik Suraksha Sanhita, 2023. It is the new criminal procedure code of India that replaces the Code of Criminal Procedure (CrPC) 1973. The BNSS was enacted on December 25, 2023, and came into effect on July 1, 2024. It contains 531 sections covering the procedure for investigation, inquiry, and trial of criminal offences.\n\nDISCLAIMER: This is legal orientation, not legal advice. Consult a qualified advocate for your specific situation.",
    },
    {
        "instruction": "What does BSA stand for in Indian law?",
        "input": "",
        "output": "BSA stands for the Bharatiya Sakshya Adhiniyam, 2023. It is the new Indian Evidence Act that replaces the Evidence Act 1872. The BSA was enacted on December 25, 2023, and came into effect on July 1, 2024. It contains 170 sections covering the admissibility, relevancy, and examination of evidence in Indian courts.\n\nDISCLAIMER: This is legal orientation, not legal advice. Consult a qualified advocate for your specific situation.",
    },
    {
        "instruction": "What is the difference between IPC and BNS?",
        "input": "",
        "output": "The IPC (Indian Penal Code) 1860 was India's criminal code for over 160 years. The BNS (Bharatiya Nyaya Sanhita) 2023 is its replacement, enacted on December 25, 2023, and effective from July 1, 2024. Key differences: BNS has 358 sections (IPC had 511), BNS adds new offences like organized crime and terrorism, BNS increases minimum punishments for many offences, and BNS incorporates community service as a form of punishment.\n\nDISCLAIMER: This is legal orientation, not legal advice. Consult a qualified advocate for your specific situation.",
    },
    {
        "instruction": "What is the difference between CrPC and BNSS?",
        "input": "",
        "output": "The CrPC (Code of Criminal Procedure) 1973 was India's criminal procedure code. The BNSS (Bharatiya Nagarik Suraksha Sanhita) 2023 is its replacement, effective from July 1, 2024. Key changes: BNSS mandates forensic investigation for crimes punishable with 7+ years, BNSS allows electronic/digital records as evidence, BNSS reduces time limits for trials, and BNSS introduces plea bargaining provisions.\n\nDISCLAIMER: This is legal orientation, not legal advice. Consult a qualified advocate for your specific situation.",
    },
    {
        "instruction": "What is the difference between the Evidence Act and BSA?",
        "input": "",
        "output": "The Indian Evidence Act 1872 governed admissibility of evidence in Indian courts. The BSA (Bharatiya Sakshya Adhiniyam) 2023 is its replacement, effective from July 1, 2024. Key changes: BSA recognizes electronic and digital records as primary evidence, BSA allows statements of accused persons under certain conditions, and BSA modernizes provisions on expert testimony.\n\nDISCLAIMER: This is legal orientation, not legal advice. Consult a qualified advocate for your specific situation.",
    },
    {
        "instruction": "What does RTI mean in Indian law?",
        "input": "",
        "output": "RTI stands for the Right to Information Act, 2005. It grants Indian citizens the right to request information from public authorities. Any citizen can seek information within 30 days. Public authorities must designate Public Information Officers (PIOs) to handle requests.\n\nDISCLAIMER: This is legal orientation, not legal advice. Consult a qualified advocate for your specific situation.",
    },
    {
        "instruction": "What does CPA mean in Indian law?",
        "input": "",
        "output": "CPA stands for the Consumer Protection Act, 2019. It provides a three-tier consumer dispute redressal mechanism: District Commission, State Commission, and National Commission. The Act covers all goods and services, including e-commerce and digital transactions.\n\nDISCLAIMER: This is legal orientation, not legal advice. Consult a qualified advocate for your specific situation.",
    },
]


def generate_ipc_bns_pairs() -> list[dict]:
    """Generate Q&A pairs for IPC to BNS section mapping."""
    pairs = []
    for ipc_section, bns_equiv in IPC_BNS_MAPPING.items():
        if bns_equiv is None:
            continue
        pairs.append({
            "instruction": f"What is the BNS equivalent of Section {ipc_section} of the Indian Penal Code?",
            "input": "",
            "output": f"Section {ipc_section} of the Indian Penal Code (IPC) corresponds to {bns_equiv} of the Bharatiya Nyaya Sanhita (BNS) 2023.\n\nDISCLAIMER: This is legal orientation, not legal advice. Consult a qualified advocate for your specific situation.",
        })
    return pairs


# Question templates for multiple Q&A per section
QUESTION_TEMPLATES_V2 = {
    "punishment": [
        "What is the punishment for {title_lower} under {act_name}?",
        "How is {title_lower} punished under {act_name}?",
        "What penalty does the law prescribe for {title_lower}?",
        "What happens if someone is found guilty of {title_lower}?",
        "What is the jail term for {title_lower}?",
    ],
    "definition": [
        "What does {act_name} mean by '{title}'?",
        "How is '{title}' defined in {act_name}?",
        "What is the legal definition of {title_lower}?",
        "Can you explain what '{title}' means under {act_name}?",
        "What is the meaning of {title_lower} in Indian law?",
    ],
    "right": [
        "What are my rights regarding {title_lower}?",
        "Can I {title_lower} under {act_name}?",
        "How do I exercise my right to {title_lower}?",
        "What rights do I have under {act_name} about {title_lower}?",
        "Am I entitled to {title_lower} under Indian law?",
    ],
    "offence": [
        "Is {title_lower} a crime under {act_name}?",
        "What happens if someone commits {title_lower}?",
        "What are the consequences of {title_lower}?",
        "Can I be arrested for {title_lower}?",
        "What are the penalties for {title_lower}?",
    ],
    "procedure": [
        "What is the procedure for {title_lower}?",
        "How do I file for {title_lower}?",
        "What steps should I follow for {title_lower}?",
        "How does one go about {title_lower}?",
        "What is the process for {title_lower} under {act_name}?",
    ],
    "default": [
        "What does Section {section_number} of {act_name} say about {title_lower}?",
        "Explain Section {section_number} of {act_name} in simple terms.",
        "What should I know about {title_lower} under {act_name}?",
        "Can you explain Section {section_number} of {act_name}?",
        "What does the law say about {title_lower}?",
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


def generate_multi_qa_per_section(section: dict, pairs_per_section: int = 3) -> list[dict]:
    """Generate multiple Q&A pairs per section using varied templates."""
    section_no = section.get("section_number", "")
    title = section.get("title", "this provision")
    text = section.get("text", "")
    act_name = section.get("act_name", "Indian law")

    category = categorize_section(title, text)
    templates = QUESTION_TEMPLATES_V2.get(category, QUESTION_TEMPLATES_V2["default"])

    title_lower = title.lower().rstrip(".")
    clean_text = re.sub(r"\s+", " ", text).strip()
    if len(clean_text) > 600:
        clean_text = clean_text[:600] + "..."

    pairs = []
    selected = random.sample(templates, min(pairs_per_section, len(templates)))

    for template in selected:
        instruction = template.format(
            title=title,
            title_lower=title_lower,
            section_number=section_no,
            act_name=act_name,
        )

        output = (
            f"Under Section {section_no} of the {act_name} ({title}):\n\n"
            f"{clean_text}\n\n"
            f"This section falls under the {act_name} and provides guidance on {title_lower}.\n\n"
            f"DISCLAIMER: This is legal orientation, not legal advice. "
            f"Consult a qualified advocate for your specific situation."
        )

        pairs.append({
            "instruction": instruction,
            "input": "",
            "output": output,
        })

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


def generate_v2_dataset(pairs_per_section: int = 3):
    """Generate the full v2 dataset targeting 10-15k pairs.

    Sources:
    1. Scraped sections x 3-5 Q&A pairs each
    2. IPC to BNS mapping pairs (~50 key transitions)
    3. Abbreviation disambiguation pairs (~8)
    """
    raw_dir = config.raw_dir
    output_dir = config.synthetic_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(42)
    all_pairs = []

    # 1. Load scraped sections and generate multiple Q&A per section
    sections = load_sections_from_raw(raw_dir)
    print(f"Loaded {len(sections)} sections from raw data")

    if sections:
        for i, section in enumerate(sections):
            if not section.get("text"):
                continue
            qa_pairs = generate_multi_qa_per_section(section, pairs_per_section)
            all_pairs.extend(qa_pairs)
            if (i + 1) % 100 == 0:
                print(f"  Generated {len(all_pairs)} pairs from {i + 1}/{len(sections)} sections...")
        print(f"Generated {len(all_pairs)} pairs from {len(sections)} sections ({pairs_per_section} per section)")

    # 2. Add IPC to BNS mapping pairs
    ipc_bns_pairs = generate_ipc_bns_pairs()
    all_pairs.extend(ipc_bns_pairs)
    print(f"Added {len(ipc_bns_pairs)} IPC to BNS mapping pairs")

    # 3. Add abbreviation disambiguation pairs
    all_pairs.extend(ABBREVIATION_PAIRS)
    print(f"Added {len(ABBREVIATION_PAIRS)} abbreviation disambiguation pairs")

    # Deduplicate
    seen = set()
    unique_pairs = []
    for pair in all_pairs:
        normalized = pair["instruction"].lower().strip()
        normalized = re.sub(r"[^\w\s]", "", normalized)
        if normalized not in seen:
            seen.add(normalized)
            unique_pairs.append(pair)

    print(f"\nTotal unique pairs: {len(unique_pairs)}")

    # Save
    output_file = output_dir / "synthetic_qa_v2.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(unique_pairs, f, indent=2, ensure_ascii=False)

    print(f"Saved to {output_file}")

    # Summary
    print("\n" + "=" * 60)
    print("v2 Dataset Summary")
    print("=" * 60)
    section_count = len(all_pairs) - len(ipc_bns_pairs) - len(ABBREVIATION_PAIRS)
    print(f"Section-based pairs: {section_count}")
    print(f"IPC to BNS mapping pairs: {len(ipc_bns_pairs)}")
    print(f"Abbreviation pairs: {len(ABBREVIATION_PAIRS)}")
    print(f"Total (before dedup): {len(all_pairs)}")
    print(f"Total (after dedup): {len(unique_pairs)}")
    print(f"Target range: 10,000 - 15,000")

    return unique_pairs


if __name__ == "__main__":
    generate_v2_dataset()
