"""THEMIS inference engine.

Handles model loading, LoRA adapter attachment, and response generation.
"""

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from .config import config


@dataclass
class GenerationResult:
    """Result from model generation."""

    response: str
    input_tokens: int
    output_tokens: int
    device: str


class ThemisInference:
    """THEMIS model inference engine."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = self._resolve_device()

    def _resolve_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def load_model(self):
        """Load the base model in 4-bit and attach the LoRA adapter from HuggingFace Hub."""
        base_model_id = config.base_model
        lora_adapter_id = config.lora_adapter

        print(f"Loading base model: {base_model_id}")
        print(f"Device: {self.device}")

        # 4-bit quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model in 4-bit
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            cache_dir=str(config.model_cache_dir),
            trust_remote_code=True,
        )

        # Attach LoRA adapter — downloads from HuggingFace Hub on first run
        print(f"Loading LoRA adapter: {lora_adapter_id}")
        self.model = PeftModel.from_pretrained(
            self.model,
            lora_adapter_id,
            cache_dir=str(config.model_cache_dir),
        )

        self.model.eval()
        print("Model loaded successfully.")

    def format_prompt(self, question: str, history: list[dict] = None) -> str:
        """Format the user question as an Alpaca-style instruction prompt."""
        prompt = f"### Instruction:\n{question}\n\n### Response:\n"
        return prompt

    def generate(
        self,
        question: str,
        history: list[dict] = None,
        temperature: float = None,
        max_new_tokens: int = None,
    ) -> GenerationResult:
        """Generate a response to a legal question."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        prompt = self.format_prompt(question, history)

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_tokens = inputs["input_ids"].shape[1]

        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                temperature=temperature or config.temperature,
                top_p=config.top_p,
                max_new_tokens=max_new_tokens or config.max_new_tokens,
                repetition_penalty=config.repetition_penalty,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only new tokens
        new_tokens = outputs[0][input_tokens:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Strip the prompt prefix if echoed back
        if response.startswith(prompt):
            response = response[len(prompt):]

        response = response.strip()

        # Add disclaimer if not present
        if "disclaimer" not in response.lower():
            response = f"{response}\n\n{config.disclaimer}"

        return GenerationResult(
            response=response,
            input_tokens=input_tokens,
            output_tokens=len(new_tokens),
            device=self.device,
        )


# Singleton instance
_inference = None


def get_inference() -> ThemisInference:
    """Get or create the singleton inference instance."""
    global _inference
    if _inference is None:
        _inference = ThemisInference()
    return _inference


def load_model():
    """Load the model (convenience function)."""
    inference = get_inference()
    inference.load_model()
    return inference


def generate_response(question: str) -> str:
    """Generate a response (convenience function)."""
    inference = get_inference()
    result = inference.generate(question)
    return result.response
