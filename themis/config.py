"""THEMIS project configuration."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    """Project-wide configuration for THEMIS."""

    # Paths
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    data_dir: Path = field(default_factory=lambda: Path(__file__).parent / "data")
    raw_dir: Path = field(default_factory=lambda: Path(__file__).parent / "data" / "scraper" / "raw")
    synthetic_dir: Path = field(default_factory=lambda: Path(__file__).parent / "data" / "synthetic")
    model_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "model")
    eval_dir: Path = field(default_factory=lambda: Path(__file__).parent / "eval")

    # Model
    base_model: str = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
    lora_adapter: str = "Daniel2503/themis-mistral-7b-lora"
    model_cache_dir: Path = field(default_factory=lambda: Path(__file__).parent / "model")

    # Generation
    temperature: float = 0.3
    top_p: float = 0.9
    max_new_tokens: int = 512
    repetition_penalty: float = 1.1

    # Device
    device: str = "auto"
    load_in_4bit: bool = True

    # Training
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    epochs: int = 3
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    max_seq_length: int = 512

    # Data
    target_laws: list[str] = field(default_factory=lambda: [
        "Bharatiya Nyaya Sanhita, 2023",
        "Bharatiya Nagarik Suraksha Sanhita, 2023",
        "Bharatiya Sakshya Adhiniyam, 2023",
        "Consumer Protection Act, 2019",
        "Right to Information Act, 2005",
        "Payment of Wages Act, 1936",
        "Industrial Disputes Act, 1947",
    ])

    # Scraper
    india_code_base: str = "https://www.indiacode.nic.in"
    request_delay: float = 1.0  # seconds between requests
    max_retries: int = 3

    # Synthetic generation (Groq free tier)
    groq_api_url: str = "https://api.groq.com/openai/v1/chat/completions"
    groq_model: str = "mixtral-8x7b-32768"  # Free on Groq
    use_template_fallback: bool = True  # Use templates if API unavailable

    # System prompt
    system_prompt: str = (
        "You are THEMIS, a legal intelligence engine for Indian law. "
        "Answer questions about Indian statutory law based on the Bharatiya Nyaya Sanhita (BNS), "
        "Bharatiya Nagarik Suraksha Sanhita (BNSS), Indian Penal Code (IPC), "
        "Consumer Protection Act, RTI Act, and related statutes. "
        "Always cite specific section numbers in your responses. "
        "Provide plain-language explanations suitable for citizens. "
        "If a question is outside the scope of statutory law, refuse and recommend consulting a lawyer. "
        "Always include a disclaimer that this is legal orientation, not legal advice."
    )

    disclaimer: str = (
        "DISCLAIMER: This is legal orientation, not legal advice. "
        "Consult a qualified advocate for your specific situation."
    )


config = Config()
