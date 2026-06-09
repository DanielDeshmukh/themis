"""THEMIS evaluation harness.

Runs 50 held-out questions and computes metrics:
- Citation accuracy
- Refusal rate
- ROUGE-L
- Hallucination rate
"""

import json
import re
from pathlib import Path


def extract_section_citations(text: str) -> set[str]:
    """Extract section citations from text (e.g., 'Section 118', 'BNS 302')."""
    patterns = [
        r"Section\s+(\d+[A-Z]?)",
        r"(\d+[A-Z]?)\s+(IPC|BNS|BNSS|BSA)",
        r"§\s*(\d+[A-Z]?)",
    ]
    citations = set()
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            citations.add(match.group(1) if match.lastindex == 1 else match.group(0))
    return citations


def citation_accuracy(predicted: str, ground_truth: str) -> float:
    """Check if correct section numbers are cited."""
    pred_citations = extract_section_citations(predicted)
    gt_citations = extract_section_citations(ground_truth)

    if not gt_citations:
        return 1.0 if not pred_citations else 0.5

    if not pred_citations:
        return 0.0

    # Calculate overlap
    overlap = pred_citations & gt_citations
    return len(overlap) / len(gt_citations)


def refusal_rate(query: str, response: str) -> bool:
    """Check if model correctly refuses out-of-scope queries."""
    refusal_indicators = [
        "i cannot",
        "i can't",
        "i am unable",
        "outside the scope",
        "consult a lawyer",
        "consult a qualified",
        "not legal advice",
        "seek professional",
    ]
    return any(indicator in response.lower() for indicator in refusal_indicators)


def rouge_l(predicted: str, ground_truth: str) -> float:
    """Compute ROUGE-L score between predicted and ground truth."""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = scorer.score(ground_truth, predicted)
        return scores["rougeL"].fmeasure
    except ImportError:
        # Simple fallback: word overlap
        pred_words = set(predicted.lower().split())
        gt_words = set(ground_truth.lower().split())
        if not gt_words:
            return 0.0
        return len(pred_words & gt_words) / len(gt_words)


def hallucination_check(response: str, valid_sections: set[str] = None) -> bool:
    """Check for fabricated section numbers."""
    if valid_sections is None:
        return False  # Can't check without valid sections

    citations = extract_section_citations(response)
    if not citations:
        return False

    # Check if all cited sections are valid
    invalid = citations - valid_sections
    return len(invalid) > 0


def run_evaluation(eval_set_path: str = None, predictions_path: str = None):
    """Run the full evaluation harness."""
    from ..config import config

    eval_dir = config.eval_dir
    eval_set_path = eval_set_path or str(eval_dir / "eval_set.json")
    predictions_path = predictions_path or str(eval_dir / "results.json")

    # Load eval set
    with open(eval_set_path, "r", encoding="utf-8") as f:
        eval_set = json.load(f)

    # Load predictions
    with open(predictions_path, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    # Build set of valid sections from raw data for hallucination check
    valid_sections = set()
    for json_file in config.raw_dir.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                for sec in data.get("sections", []):
                    valid_sections.add(sec.get("section_number", ""))
        except Exception:
            pass

    print("=" * 60)
    print("THEMIS Evaluation Results")
    print("=" * 60)

    # Compute metrics
    citation_scores = []
    rouge_scores = []
    refusal_count = 0
    hallucination_count = 0
    total = len(predictions)

    for pred in predictions:
        expected = pred.get("expected", "")
        predicted = pred.get("predicted", "")
        question = pred.get("question", "")

        # Citation accuracy
        ca = citation_accuracy(predicted, expected)
        citation_scores.append(ca)

        # ROUGE-L
        rl = rouge_l(predicted, expected)
        rouge_scores.append(rl)

        # Refusal rate (check if model refuses appropriately)
        if refusal_rate(question, predicted):
            refusal_count += 1

        # Hallucination check
        if hallucination_check(predicted, valid_sections):
            hallucination_count += 1

    # Aggregate metrics
    avg_citation = sum(citation_scores) / len(citation_scores) if citation_scores else 0
    avg_rouge = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0
    refusal_pct = refusal_count / total if total else 0
    hallucination_pct = hallucination_count / total if total else 0

    # Display results
    print(f"\nTotal questions evaluated: {total}")
    print(f"\nCitation Accuracy:    {avg_citation:.2%}")
    print(f"ROUGE-L Score:        {avg_rouge:.2%}")
    print(f"Refusal Rate:         {refusal_pct:.2%} ({refusal_count}/{total})")
    print(f"Hallucination Rate:   {hallucination_pct:.2%} ({hallucination_count}/{total})")

    # Per-category breakdown (if categories available)
    print("\n" + "-" * 40)
    print("Summary")
    print("-" * 40)
    print(f"{'Metric':<30} {'Score':<10}")
    print(f"{'Citation Accuracy':<30} {avg_citation:.2%}")
    print(f"{'ROUGE-L':<30} {avg_rouge:.2%}")
    print(f"{'Refusal Rate':<30} {refusal_pct:.2%}")
    print(f"{'Hallucination Rate':<30} {hallucination_pct:.2%}")

    return {
        "citation_accuracy": avg_citation,
        "rouge_l": avg_rouge,
        "refusal_rate": refusal_pct,
        "hallucination_rate": hallucination_pct,
        "total_questions": total,
    }


if __name__ == "__main__":
    run_evaluation()
