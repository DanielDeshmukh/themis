# THEMIS — Data Pipeline

## Overview

THEMIS requires ~2,800 instruction-tuning pairs in Alpaca format. The data comes from three sources: India Code (Bare Acts), Indian Kanoon (judgment summaries), and synthetic Q&A pairs generated via Groq API (free) or template-based fallback.

---

## Data Sources

### 1. India Code (indiacode.nic.in)

**Type:** Official Bare Acts text
**Content:** Full statutory text of Indian laws
**Target Laws:**
- Bharatiya Nyaya Sanhita (BNS) 2023
- Bharatiya Nagarik Suraksha Sanhita (BNSS) 2023
- Bharatiya Sakshya Adhiniyam (BSA) 2023
- Consumer Protection Act 2019
- Right to Information Act 2005
- Payment of Wages Act 1936
- Industrial Disputes Act 1947

**Volume:** ~800 sections parsed
**Format:** Structured sections with section numbers, titles, and text

### 2. Indian Kanoon (indiankanoon.org)

**Type:** Judgment summaries
**Content:** Selected judgment summaries across domains
**Target Domains:**
- Criminal law (BNS/IPC)
- Civil disputes
- Consumer rights
- Labour law

**Volume:** ~500 judgments
**Format:** Case summaries with holdings and relevant sections cited

### 3. Synthetic Q&A (Groq API / Template Fallback)

**Type:** Instruction-tuning pairs
**Content:** Q&A pairs generated from parsed Bare Act sections
**Generation Methods:**
1. **Groq API** (primary) — Free tier, Mixtral-8x7b, fast inference
2. **Template-based** (fallback) — No API needed, deterministic generation

**Volume:** ~1,500 pairs
**Format:** Alpaca instruction format
**Cost:** $0 (Groq free tier or template fallback)

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    RAW SOURCES                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐          │
│  │India Code│  │Indian    │  │Groq API /    │          │
│  │(Bare Acts)│  │Kanoon    │  │Templates     │          │
│  └────┬─────┘  └────┬─────┘  └──────┬───────┘          │
└───────┼──────────────┼───────────────┼──────────────────┘
        │              │               │
        ▼              ▼               ▼
┌───────────────┐ ┌───────────┐ ┌───────────────┐
│  indiacode.py │ │ kanoon.py │ │  generate.py  │
│  (Scraper)    │ │ (Scraper) │ │  (Generator)  │
└───────┬───────┘ └─────┬─────┘ └───────┬───────┘
        │               │               │
        ▼               ▼               ▼
┌─────────────────────────────────────────────────────────┐
│                  RAW DATA FILES                          │
│  india_code_sections.json    (~800 sections)            │
│  kanoon_judgments.json       (~500 judgments)           │
│  synthetic_qa.json           (~1500 Q&A pairs)         │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
                ┌───────────────┐
                │ preprocess.py │
                │ (Clean+Merge) │
                └───────┬───────┘
                        │
                        ▼
                ┌───────────────┐
                │ dataset.json  │
                │ (~2800 pairs) │
                └───────────────┘
```

---

## Scraping Scripts

### indiacode.py

**Purpose:** Parse Bare Act sections from India Code website

**Strategy:**
1. Navigate to specific law pages on indiacode.nic.in
2. Extract section structure (number, title, text)
3. Handle HTML parsing with BeautifulSoup
4. Store as structured JSON

**Output Format:**
```json
{
  "source": "india_code",
  "law": "BNS 2023",
  "section_number": "118",
  "section_title": "Cause grievous hurt by dangerous weapons",
  "text": "Whoever causes grievous hurt by means of any instrument...",
  "replaces": "IPC 326",
  "scraped_at": "2026-06-09T12:00:00Z"
}
```

**Challenges:**
- Site may have anti-scraping measures
- HTML structure may change
- Need to handle pagination for large acts
- Rate limiting to be respectful

### kanoon.py

**Purpose:** Scrape judgment summaries from Indian Kanoon

**Strategy:**
1. Search for judgments by domain (criminal, civil, consumer)
2. Extract case name, court, date, holdings
3. Identify cited sections (BNS/IPC/BNSS)
4. Store as structured JSON

**Output Format:**
```json
{
  "source": "indian_kanoon",
  "case_name": "State of Maharashtra v. XYZ",
  "court": "Supreme Court of India",
  "date": "2024-01-15",
  "cited_sections": ["BNS 302", "IPC 302"],
  "summary": "The court held that...",
  "domain": "criminal",
  "scraped_at": "2026-06-09T12:00:00Z"
}
```

---

## Synthetic Generation

### generate.py

**Purpose:** Generate Q&A pairs from parsed Bare Act sections

**Methods (in order of preference):**

1. **Groq API** (primary) — Free tier, Mixtral-8x7b
2. **Template-based** (fallback) — No API needed

**Groq API Method:**
1. Load parsed sections from scraped JSON files
2. For each section, call Groq API with structured prompt
3. Parse JSON response to extract Q&A pair
4. Include citation in answer
5. Store in Alpaca format

**Template Fallback Method:**
1. Categorize section by title keywords (punishment, definition, right, offence, procedure)
2. Select appropriate question template
3. Generate answer from section text with proper citation
4. No API key required

**Rate Limiting:**
- Groq: 30 requests/minute (free tier)
- Templates: No rate limiting

**Quality Controls:**
- Verify generated section numbers exist in source data
- Check answer length (minimum 50 words)
- Ensure disclaimer is present
- Deduplicate by instruction similarity

---

## Preprocessing

### preprocess.py

**Purpose:** Clean, deduplicate, and merge all data sources into final dataset

**Steps:**

1. **Load** all raw JSON files
2. **Clean** text content:
   - Remove HTML tags
   - Normalize whitespace
   - Fix encoding issues
   - Remove non-legal content
3. **Validate** data quality:
   - Check required fields exist
   - Verify section numbers are valid
   - Ensure answers are substantive (>50 words)
4. **Deduplicate**:
   - Remove exact duplicate questions
   - Remove near-duplicates (cosine similarity > 0.9)
5. **Merge** all sources into single Alpaca format
6. **Split** into train/eval (90/10 split)
7. **Export** final `dataset.json`

**Output Format (Alpaca):**
```json
[
  {
    "instruction": "What is the punishment for murder under BNS?",
    "input": "",
    "output": "Under Section 101 of the Bharatiya Nyaya Sanhita (BNS) 2023, whoever commits murder shall be punished with death, or imprisonment for life, and shall also be liable to fine. This replaces Section 302 of the Indian Penal Code (IPC) 1860.\n\n⚠ DISCLAIMER: This is legal orientation, not legal advice. Consult a qualified advocate for your specific situation."
  }
]
```

---

## Data Quality Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Total pairs | ~2,800 | Count of valid entries |
| Citation accuracy | 100% | Every cited section exists in source |
| Disclaimer present | 100% | Every answer includes disclaimer |
| Answer length | >50 words average | Word count per answer |
| Deduplication | 0 duplicates | Cosine similarity < 0.9 |
| Source distribution | Balanced | ~30% India Code, ~20% Kanoon, ~50% Synthetic |

---

## Dataset Statistics (Planned)

| Source | Pairs | Percentage |
|--------|-------|------------|
| India Code (parsed sections) | ~800 | 28% |
| Indian Kanoon (judgment Q&A) | ~500 | 18% |
| Synthetic (Groq/template) | ~1,500 | 54% |
| **Total** | **~2,800** | **100%** |

---

## File Locations

```
data/
├── scraper/
│   ├── indiacode.py           # India Code scraper
│   ├── kanoon.py              # Indian Kanoon scraper
│   └── raw/
│       ├── india_code_sections.json   # Raw scraped sections
│       └── kanoon_judgments.json      # Raw scraped judgments
├── synthetic/
│   ├── generate.py            # Groq/template Q&A generator
│   ├── prompts.py             # Prompt templates
│   └── synthetic_qa.json      # Generated Q&A pairs
├── preprocess.py              # Cleaning and merging
├── dataset.json               # Final training dataset
└── eval_set.json              # 50-question evaluation set
```

---

## Execution Order

1. **Scrape India Code** → `india_code_sections.json`
2. **Scrape Indian Kanoon** → `kanoon_judgments.json`
3. **Generate synthetic Q&A** → `synthetic_qa.json`
4. **Preprocess and merge** → `dataset.json`
5. **Create eval set** → `eval_set.json` (manually curated)
6. **Validate** → Run quality checks on final dataset

---

## Cost Estimates

### Synthetic Generation
- **Groq API (primary):** $0 — Free tier (30 req/min, 14K tokens/day)
- **Template fallback:** $0 — No API needed
- Total: ~1,500 Q&A pairs at zero cost

### Scraping
- Free (public data from India Code + Indian Kanoon)
- Time: ~2-4 hours total

### Storage
- Raw data: ~50MB
- Final dataset: ~10MB
- Minimal cost

### Total Project Cost: $0
All components use free tiers or open-source tools.

---

## Future Enhancements

| Enhancement | Version | Description |
|-------------|---------|-------------|
| Auto-refresh | v1.1 | Re-scrape periodically for legal updates |
| Hindi data | v1.2 | Add Hindi Bare Acts and Q&A pairs |
| More laws | v1.1+ | Add Labour Act, Family Law, etc. |
| Quality scoring | v1.1 | Automated quality scoring for generated pairs |
| Active learning | v2.0 | Use model predictions to identify data gaps |
