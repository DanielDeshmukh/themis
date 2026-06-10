"""THEMIS v2 data generation — expanded template engine for 15-20k pairs.

Strategy:
1. 15+ question template categories with 75+ unique templates
2. 8-12 Q&A pairs per section (vs 3 in v1)
3. IPC → BNS section mapping pairs (200+ transitions)
4. Abbreviation disambiguation pairs (20+)
5. Cross-section comparison pairs
6. Scenario-based practical pairs
7. All template-only, no API needed
"""

import json
import random
import re
from pathlib import Path

from ...config import config

# =============================================================================
# IPC to BNS section mapping — comprehensive
# =============================================================================

IPC_BNS_MAPPING = {
    "34": "Section 9", "101": "Section 103", "102": "Section 104", "103": "Section 103",
    "105": "Section 105", "106": "Section 106", "109": "Section 108", "111": "Section 110",
    "114": "Section 113", "120A": "Section 110", "120B": "Section 110", "141": "Section 142",
    "143": "Section 139", "144": "Section 140", "145": "Section 141", "146": "Section 143",
    "147": "Section 143", "148": "Section 144", "149": "Section 145", "160": "Section 146",
    "224": "Section 232", "225": "Section 233", "299": "Section 100", "300": "Section 101",
    "302": "Section 101", "303": "Section 105", "304": "Section 105", "304A": "Section 106",
    "305": "Section 106", "306": "Section 108", "307": "Section 109", "308": "Section 110",
    "309": "Section 47", "311": "Section 114", "312": "Section 117", "313": "Section 117",
    "314": "Section 117", "315": "Section 117", "316": "Section 117", "317": "Section 117",
    "318": "Section 117", "319": "Section 117", "320": "Section 118", "321": "Section 118",
    "322": "Section 118", "323": "Section 121", "324": "Section 122", "325": "Section 123",
    "326": "Section 124", "327": "Section 125", "328": "Section 126", "329": "Section 127",
    "330": "Section 128", "331": "Section 129", "332": "Section 130", "333": "Section 131",
    "334": "Section 132", "335": "Section 133", "336": "Section 134", "337": "Section 135",
    "338": "Section 136", "341": "Section 164", "342": "Section 165", "343": "Section 166",
    "344": "Section 167", "345": "Section 168", "346": "Section 169", "347": "Section 170",
    "348": "Section 171", "352": "Section 352", "354": "Section 74", "355": "Section 355",
    "356": "Section 356", "357": "Section 357", "358": "Section 358", "359": "Section 363",
    "360": "Section 364", "361": "Section 365", "362": "Section 366", "363A": "Section 367",
    "364": "Section 368", "365": "Section 369", "366": "Section 370", "366A": "Section 370",
    "366B": "Section 370", "367": "Section 371", "368": "Section 372", "369": "Section 373",
    "370": "Section 370", "371": "Section 371", "372": "Section 372", "373": "Section 373",
    "374": "Section 374", "375": "Section 63", "376": "Section 64", "376A": "Section 64",
    "376B": "Section 64", "376C": "Section 64", "376D": "Section 64", "377": "Section 77",
    "378": "Section 303", "379": "Section 304", "380": "Section 305", "383": "Section 308",
    "384": "Section 309", "385": "Section 310", "386": "Section 311", "387": "Section 312",
    "388": "Section 313", "389": "Section 314", "390": "Section 315", "391": "Section 316",
    "392": "Section 317", "393": "Section 318", "394": "Section 319", "395": "Section 320",
    "396": "Section 321", "397": "Section 322", "398": "Section 323", "399": "Section 324",
    "400": "Section 325", "401": "Section 326", "402": "Section 327", "403": "Section 328",
    "404": "Section 329", "405": "Section 316", "406": "Section 317", "407": "Section 318",
    "408": "Section 319", "409": "Section 320", "411": "Section 330", "412": "Section 331",
    "413": "Section 332", "414": "Section 333", "415": "Section 138", "416": "Section 139",
    "417": "Section 140", "418": "Section 141", "419": "Section 142", "420": "Section 143",
    "421": "Section 148", "422": "Section 149", "423": "Section 150", "424": "Section 151",
    "425": "Section 148", "426": "Section 149", "427": "Section 150", "428": "Section 151",
    "429": "Section 152", "430": "Section 153", "431": "Section 154", "432": "Section 155",
    "433": "Section 156", "434": "Section 157", "435": "Section 158", "436": "Section 159",
    "437": "Section 160", "438": "Section 161", "439": "Section 162", "440": "Section 163",
    "441": "Section 164", "442": "Section 165", "443": "Section 166", "444": "Section 167",
    "445": "Section 168", "446": "Section 169", "447": "Section 170", "448": "Section 171",
    "449": "Section 172", "450": "Section 173", "451": "Section 174", "452": "Section 175",
    "453": "Section 176", "454": "Section 177", "455": "Section 178", "456": "Section 179",
    "457": "Section 180", "458": "Section 181", "459": "Section 182", "460": "Section 183",
    "461": "Section 184", "462": "Section 185", "463": "Section 186", "464": "Section 187",
    "465": "Section 188", "466": "Section 189", "467": "Section 190", "468": "Section 191",
    "469": "Section 192", "470": "Section 193", "471": "Section 194", "472": "Section 195",
    "473": "Section 196", "474": "Section 197", "475": "Section 198", "476": "Section 199",
    "477": "Section 200", "477A": "Section 200", "478": "Section 201", "479": "Section 202",
    "480": "Section 203", "481": "Section 204", "482": "Section 205", "483": "Section 206",
    "484": "Section 207", "485": "Section 208", "486": "Section 209", "487": "Section 210",
    "488": "Section 211", "489": "Section 212", "489A": "Section 212", "489B": "Section 213",
    "489C": "Section 214", "489D": "Section 215", "489E": "Section 216", "490": "Section 217",
    "491": "Section 218", "492": "Section 219", "493": "Section 220", "494": "Section 221",
    "495": "Section 222", "496": "Section 223", "498": "Section 85", "498A": "Section 85",
    "500": "Section 356", "501": "Section 357", "502": "Section 358", "503": "Section 351",
    "504": "Section 352", "505": "Section 353", "506": "Section 351", "507": "Section 352",
    "508": "Section 354", "509": "Section 354", "510": "Section 355", "511": "Section 61",
}

# =============================================================================
# Abbreviation disambiguation pairs — comprehensive
# =============================================================================

ABBREVIATION_PAIRS = [
    {"instruction": "What does BNS stand for in Indian law?", "input": "", "output": "BNS stands for the Bharatiya Nyaya Sanhita, 2023. It is India's new criminal code replacing the Indian Penal Code (IPC) 1860. Enacted December 25, 2023, effective July 1, 2024, it contains 358 sections.\n\nDISCLAIMER: This is legal orientation, not legal advice. Consult a qualified advocate for your specific situation."},
    {"instruction": "What does BNSS stand for in Indian law?", "input": "", "output": "BNSS stands for the Bharatiya Nagarik Suraksha Sanhita, 2023. It is India's new criminal procedure code replacing CrPC 1973. Effective July 1, 2024, it contains 531 sections governing investigation, inquiry, and trial.\n\nDISCLAIMER: This is legal orientation, not legal advice. Consult a qualified advocate for your specific situation."},
    {"instruction": "What does BSA stand for in Indian law?", "input": "", "output": "BSA stands for the Bharatiya Sakshya Adhiniyam, 2023. It replaces the Indian Evidence Act 1872. Effective July 1, 2024, it contains 170 sections on evidence admissibility and examination.\n\nDISCLAIMER: This is legal orientation, not legal advice. Consult a qualified advocate for your specific situation."},
    {"instruction": "What is the difference between IPC and BNS?", "input": "", "output": "IPC (1860) had 511 sections; BNS (2023) has 358. BNS adds organized crime and terrorism offences, increases minimum punishments, and includes community service as punishment.\n\nDISCLAIMER: This is legal orientation, not legal advice. Consult a qualified advocate for your specific situation."},
    {"instruction": "What is the difference between CrPC and BNSS?", "input": "", "output": "CrPC (1973) was replaced by BNSS (2023). BNSS mandates forensic investigation for crimes with 7+ years punishment, allows electronic evidence, reduces trial timelines, and introduces plea bargaining.\n\nDISCLAIMER: This is legal orientation, not legal advice. Consult a qualified advocate for your specific situation."},
    {"instruction": "What is the difference between Evidence Act and BSA?", "input": "", "output": "The Evidence Act (1872) was replaced by BSA (2023). BSA recognizes electronic records as primary evidence, allows accused statements under certain conditions, and modernizes expert testimony provisions.\n\nDISCLAIMER: This is legal orientation, not legal advice. Consult a qualified advocate for your specific situation."},
    {"instruction": "What does RTI stand for?", "input": "", "output": "RTI stands for Right to Information Act, 2005. It grants citizens the right to request information from public authorities within 30 days. Public Information Officers (PIOs) handle requests.\n\nDISCLAIMER: This is legal orientation, not legal advice. Consult a qualified advocate for your specific situation."},
    {"instruction": "What does CPA stand for?", "input": "", "output": "CPA stands for Consumer Protection Act, 2019. It provides three-tier dispute redressal: District Commission, State Commission, and National Commission. Covers all goods and services including e-commerce.\n\nDISCLAIMER: This is legal orientation, not legal advice. Consult a qualified advocate for your specific situation."},
    {"instruction": "What is the difference between BNS and BNSS?", "input": "", "output": "BNS defines substantive criminal law (what is an offence and its punishment). BNSS defines criminal procedure (how offences are investigated and tried). BNS: 'murder is punishable by death.' BNSS: 'how a murder case is tried.'\n\nDISCLAIMER: This is legal orientation, not legal advice. Consult a qualified advocate for your specific situation."},
    {"instruction": "What is the difference between BNS and BSA?", "input": "", "output": "BNS defines offences and punishments. BSA defines evidence law — what is admissible and how to prove facts in court. BNS: 'theft is a crime.' BSA: 'how you prove theft happened.'\n\nDISCLAIMER: This is legal orientation, not legal advice. Consult a qualified advocate for your specific situation."},
    {"instruction": "What is FIR in Indian law?", "input": "", "output": "FIR stands for First Information Report. Under Section 173 BNSS (formerly Section 154 CrPC), it is the first report to police about a cognizable offence. Any person can file it; police must register it.\n\nDISCLAIMER: This is legal orientation, not legal advice. Consult a qualified advocate for your specific situation."},
    {"instruction": "What is a Cognizable Offence under BNSS?", "input": "", "output": "A cognizable offence allows police to arrest without warrant and investigate without Magistrate's permission. Under BNSS, these are offences punishable with 3+ years or listed in the First Schedule.\n\nDISCLAIMER: This is legal orientation, not legal advice. Consult a qualified advocate for your specific situation."},
    {"instruction": "What is a Non-Cognizable Offence under BNSS?", "input": "", "output": "A non-cognizable offence requires police to get a warrant for arrest and Magistrate's permission for investigation. Not listed in the First Schedule. Complainant must approach Magistrate directly.\n\nDISCLAIMER: This is legal orientation, not legal advice. Consult a qualified advocate for your specific situation."},
    {"instruction": "What is Anticipatory Bail under BNSS?", "input": "", "output": "Anticipatory bail under Section 482 BNSS (formerly Section 438 CrPC) allows a person to seek bail in advance if they apprehend arrest. Applied to Sessions Court or High Court.\n\nDISCLAIMER: This is legal orientation, not legal advice. Consult a qualified advocate for your specific situation."},
    {"instruction": "What is a Charge Sheet under BNSS?", "input": "", "output": "A charge sheet (now 'police report') is filed under Section 193 BNSS after investigation. Contains offence details, evidence, and accused names. Magistrate takes cognizance based on it.\n\nDISCLAIMER: This is legal orientation, not legal advice. Consult a qualified advocate for your specific situation."},
    {"instruction": "What is Summary Trial under BNSS?", "input": "", "output": "Summary trial under BNSS applies to offences punishable with 2 years or less. Faster procedure: Magistrate records brief evidence outline, trial completes in 1-2 hearings.\n\nDISCLAIMER: This is legal orientation, not legal advice. Consult a qualified advocate for your specific situation."},
    {"instruction": "What is a Bailable Offence under BNSS?", "input": "", "output": "A bailable offence gives the accused a right to bail. Generally offences punishable with 3 years or less. Police or Magistrate must release on bail if conditions are met.\n\nDISCLAIMER: This is legal orientation, not legal advice. Consult a qualified advocate for your specific situation."},
    {"instruction": "What is a Non-Bailable Offence under BNSS?", "input": "", "output": "A non-bailable offence means bail is discretionary, not a right. Serious offences like murder, rape, dacoity. Accused must apply to court; granted based on offence seriousness and flight risk.\n\nDISCLAIMER: This is legal orientation, not legal advice. Consult a qualified advocate for your specific situation."},
    {"instruction": "What is a Compoundable Offence under BNS?", "input": "", "output": "A compoundable offence can be settled amicably between victim and accused, dropping the case. Includes minor assault, defamation, certain property offences. Murder and rape are non-compoundable.\n\nDISCLAIMER: This is legal orientation, not legal advice. Consult a qualified advocate for your specific situation."},
    {"instruction": "What does Cognizance mean under BNSS?", "input": "", "output": "Taking cognizance means the Magistrate formally acknowledges an offence and proceeds. Can be on charge sheet from police, complaint from victim, or information from any person.\n\nDISCLAIMER: This is legal orientation, not legal advice. Consult a qualified advocate for your specific situation."},
]

# =============================================================================
# Expanded question templates — 15 categories, 75+ templates
# =============================================================================

QUESTION_TEMPLATES = {
    "definition": [
        "What does {act_name} mean by '{title}'?",
        "How is '{title}' defined in {act_name}?",
        "What is the legal definition of {title_lower}?",
        "Can you explain what '{title}' means under {act_name}?",
        "What is the meaning of {title_lower} in Indian law?",
        "Define {title_lower} as per {act_name}.",
        "What are the essential elements of {title_lower} under {act_name}?",
    ],
    "punishment": [
        "What is the punishment for {title_lower} under {act_name}?",
        "How is {title_lower} punished under {act_name}?",
        "What penalty does the law prescribe for {title_lower}?",
        "What happens if someone is found guilty of {title_lower}?",
        "What is the jail term for {title_lower}?",
        "What is the maximum punishment for {title_lower}?",
        "Is there a minimum punishment for {title_lower}?",
    ],
    "procedure": [
        "What is the procedure for {title_lower}?",
        "How do I file for {title_lower}?",
        "What steps should I follow for {title_lower}?",
        "How does one go about {title_lower}?",
        "What is the process for {title_lower} under {act_name}?",
        "What is the timeline for {title_lower}?",
        "What documents are needed for {title_lower}?",
    ],
    "right": [
        "What are my rights regarding {title_lower}?",
        "Can I {title_lower} under {act_name}?",
        "How do I exercise my right to {title_lower}?",
        "What rights do I have under {act_name} about {title_lower}?",
        "Am I entitled to {title_lower} under Indian law?",
        "What remedies are available if my right to {title_lower} is violated?",
    ],
    "offence": [
        "Is {title_lower} a crime under {act_name}?",
        "What happens if someone commits {title_lower}?",
        "What are the consequences of {title_lower}?",
        "Can I be arrested for {title_lower}?",
        "What are the penalties for {title_lower}?",
        "Is {title_lower} a cognizable or non-cognizable offence?",
        "Is {title_lower} a bailable offence?",
    ],
    "exception": [
        "What are the exceptions to {title_lower} under {act_name}?",
        "Are there any defences available for {title_lower}?",
        "When does {title_lower} not apply?",
        "What situations are excluded from {title_lower}?",
        "Are there any mitigating circumstances for {title_lower}?",
    ],
    "comparison": [
        "How does {title_lower} differ from similar offences?",
        "What makes {title_lower} different from related crimes?",
        "How is {title_lower} classified under {act_name}?",
        "What distinguishes {title_lower} from lesser offences?",
    ],
    "scenario": [
        "If someone commits {title_lower}, what legal action can be taken?",
        "A person is accused of {title_lower}. What happens next?",
        "What legal options does a victim of {title_lower} have?",
        "How would a court handle a case of {title_lower}?",
        "What evidence is needed to prove {title_lower}?",
    ],
    "section_reference": [
        "What does Section {section_number} of {act_name} say about {title_lower}?",
        "Explain Section {section_number} of {act_name} in simple terms.",
        "What should I know about {title_lower} under {act_name}?",
        "Can you explain Section {section_number} of {act_name}?",
        "What does the law say about {title_lower}?",
        "Break down Section {section_number} of {act_name} for me.",
        "What are the key points of Section {section_number} of {act_name}?",
    ],
    "jurisdiction": [
        "Which court has jurisdiction over {title_lower}?",
        "Where should a case of {title_lower} be filed?",
        "Which Magistrate can try {title_lower}?",
        "Is {title_lower} triable by a Sessions Court?",
    ],
    "limitation": [
        "What is the time limit for filing a case related to {title_lower}?",
        "How long do I have to report {title_lower}?",
        "Is there a statute of limitations for {title_lower}?",
    ],
    "burden_of_proof": [
        "Who has the burden of proof in a {title_lower} case?",
        "What must the prosecution prove for {title_lower}?",
        "What is the standard of proof for {title_lower}?",
    ],
    "cross_law": [
        "How does {section_number} of {act_name} relate to criminal procedure?",
        "What role does {act_name} play in investigating {title_lower}?",
        "How does evidence law apply to {title_lower} cases?",
    ],
    "practical": [
        "How can I protect myself from {title_lower}?",
        "What should I do if I am a victim of {title_lower}?",
        "Can {title_lower} be settled out of court?",
        "What are the consequences of a false {title_lower} complaint?",
    ],
    "appeal": [
        "Can a {title_lower} conviction be appealed?",
        "Where should I file an appeal against {title_lower}?",
        "What is the time limit for appealing a {title_lower} case?",
    ],
}


# =============================================================================
# Categorization
# =============================================================================

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
    elif any(kw in combined for kw in ["except", "defence", "not apply", "exclusion"]):
        return "exception"
    elif any(kw in combined for kw in ["jurisdiction", "court", "magistrate", "tribunal"]):
        return "jurisdiction"
    elif any(kw in combined for kw in ["limit", "time", "period", "deadline"]):
        return "limitation"
    elif any(kw in combined for kw in ["burden", "proof", "evidence", "presumption"]):
        return "burden_of_proof"
    elif any(kw in combined for kw in ["appeal", "revision", "review"]):
        return "appeal"
    return "section_reference"


# =============================================================================
# Pair generation
# =============================================================================

def generate_multi_qa_per_section(section: dict, pairs_per_section: int = 10) -> list[dict]:
    """Generate multiple Q&A pairs per section using varied templates."""
    section_no = section.get("section_number", "")
    title = section.get("title", "this provision")
    text = section.get("text", "")
    act_name = section.get("act_name", "Indian law")

    category = categorize_section(title, text)
    templates = QUESTION_TEMPLATES.get(category, QUESTION_TEMPLATES["section_reference"])

    # Always include section_reference templates for diversity
    all_templates = templates + QUESTION_TEMPLATES["section_reference"]
    # Remove duplicates while preserving order
    seen = set()
    unique_templates = []
    for t in all_templates:
        if t not in seen:
            seen.add(t)
            unique_templates.append(t)

    title_lower = title.lower().rstrip(".")
    clean_text = re.sub(r"\s+", " ", text).strip()
    if len(clean_text) > 600:
        clean_text = clean_text[:600] + "..."

    pairs = []
    selected = random.sample(unique_templates, min(pairs_per_section, len(unique_templates)))

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


def generate_scenario_pairs(section: dict, num_scenarios: int = 2) -> list[dict]:
    """Generate scenario-based Q&A pairs from section text."""
    section_no = section.get("section_number", "")
    title = section.get("title", "this offence")
    text = section.get("text", "")
    act_name = section.get("act_name", "Indian law")

    if not text or len(text) < 50:
        return []

    title_lower = title.lower().rstrip(".")
    clean_text = re.sub(r"\s+", " ", text).strip()
    if len(clean_text) > 500:
        clean_text = clean_text[:500] + "..."

    scenarios = [
        f"A person is accused of {title.lower()} under {act_name}. What are the elements the prosecution must prove?",
        f"If someone commits {title.lower()}, what legal remedies are available to the victim?",
        f"Under what circumstances can {title.lower()} be justified as an exception?",
        f"What investigation steps must the police follow when {title_lower} is reported?",
        f"How would a court determine the severity of {title.lower()} in a specific case?",
    ]

    pairs = []
    selected = random.sample(scenarios, min(num_scenarios, len(scenarios)))

    for scenario in selected:
        output = (
            f"Regarding {title} under Section {section_no} of {act_name}:\n\n"
            f"{clean_text}\n\n"
            f"This section provides the legal framework for addressing {title_lower}. "
            f"The specific application depends on the facts and circumstances of each case.\n\n"
            f"DISCLAIMER: This is legal orientation, not legal advice. "
            f"Consult a qualified advocate for your specific situation."
        )
        pairs.append({
            "instruction": scenario,
            "input": "",
            "output": output,
        })

    return pairs


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


# =============================================================================
# Data loading
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


# =============================================================================
# Main generation pipeline
# =============================================================================

def generate_v2_dataset(pairs_per_section: int = 12):
    """Generate the full v2 dataset targeting 15-20k pairs.

    Sources:
    1. Scraped sections x 8-12 Q&A pairs each (templates + scenarios)
    2. Expanded IPC → BNS mapping pairs (~200 transitions)
    3. Abbreviation disambiguation pairs (~20)
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
            # Template-based Q&A
            qa_pairs = generate_multi_qa_per_section(section, pairs_per_section)
            all_pairs.extend(qa_pairs)
            # Scenario-based Q&A
            scenario_pairs = generate_scenario_pairs(section, 2)
            all_pairs.extend(scenario_pairs)
            if (i + 1) % 100 == 0:
                print(f"  Generated {len(all_pairs)} pairs from {i + 1}/{len(sections)} sections...")
        print(f"Generated {len(all_pairs)} pairs from {len(sections)} sections ({pairs_per_section} + 2 scenarios per section)")

    # 2. Add expanded IPC to BNS mapping pairs
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
    print(f"Target range: 15,000 - 20,000")

    return unique_pairs


if __name__ == "__main__":
    generate_v2_dataset()
