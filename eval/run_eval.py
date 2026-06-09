"""THEMIS evaluation runner.

Loads the eval set, runs each question through the model,
and computes metrics.
"""

import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_eval(verbose: bool = False):
    """Run the evaluation harness."""
    from eval.metrics import citation_accuracy, rouge_l, run_evaluation

    eval_dir = Path(__file__).parent
    eval_set_path = eval_dir / "eval_set.json"
    results_path = eval_dir / "results.json"

    if not eval_set_path.exists():
        print(f"Error: Eval set not found: {eval_set_path}")
        print("Run preprocess.py to generate the eval set.")
        return

    # Load eval set
    with open(eval_set_path, "r", encoding="utf-8") as f:
        eval_set = json.load(f)

    print(f"Loaded {len(eval_set)} evaluation questions")

    # Try to load model and generate predictions
    try:
        from infer import load_model, get_inference
        load_model()
        inference = get_inference()

        results = []
        for i, item in enumerate(eval_set):
            question = item.get("instruction", "")
            print(f"Q{i+1}/{len(eval_set)}: {question[:60]}...")

            try:
                result = inference.generate(question)
                results.append({
                    "question": question,
                    "expected": item.get("output", ""),
                    "predicted": result.response,
                })
                if verbose:
                    print(f"  A: {result.response[:80]}...\n")
            except Exception as e:
                print(f"  Error: {e}\n")

        # Save results
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\nResults saved to {results_path}")

    except Exception as e:
        print(f"Could not run model inference: {e}")
        print("Running metrics on existing results...")

    # Compute metrics
    if results_path.exists():
        metrics = run_evaluation(str(eval_set_path), str(results_path))
        return metrics


if __name__ == "__main__":
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    run_eval(verbose)
