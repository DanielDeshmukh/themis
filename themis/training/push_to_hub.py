"""Push trained adapter to HuggingFace Hub."""

import sys
from pathlib import Path


def push_to_hub(adapter_dir: str = None, repo_name: str = None):
    """Upload LoRA adapter to HuggingFace Hub."""
    try:
        from huggingface_hub import HfApi, login
    except ImportError:
        print("Error: huggingface_hub not installed.")
        print("Install with: pip install huggingface_hub")
        return

    # Check for HF token
    import os
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("Error: HuggingFace token not found.")
        print("Set HF_TOKEN environment variable or run: huggingface-cli login")
        return

    login(token=token)

    adapter_dir = adapter_dir or str(Path(__file__).parent.parent / "model" / "themis-lora")
    repo_name = repo_name or "Daniel2503/themis-mistral-7b-lora"

    if not Path(adapter_dir).exists():
        print(f"Error: Adapter directory not found: {adapter_dir}")
        return

    print(f"Uploading adapter from {adapter_dir} to {repo_name}...")

    api = HfApi()
    api.upload_folder(
        folder_path=adapter_dir,
        repo_id=repo_name,
        repo_type="model",
    )

    print(f"Successfully uploaded to: https://huggingface.co/{repo_name}")


if __name__ == "__main__":
    adapter_dir = sys.argv[1] if len(sys.argv) > 1 else None
    push_to_hub(adapter_dir)
