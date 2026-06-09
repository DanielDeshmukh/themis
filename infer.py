"""THEMIS inference engine.

Handles model loading, LoRA adapter attachment, and response generation.
"""

from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from config import config


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
        """Determine the best available device."""
        if config.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return config.device

    def load_model(self, model_path: str = None, lora_path: str = None):
        """Load the base model and attach LoRA adapter."""
        model_path = model_path or config.base_model
        lora_path = lora_path or str(config.local_model_path)

        print(f"Loading base model: {model_path}")
        print(f"Device: {self.device}")

        # Quantization config for 4-bit loading
        bnb_config = None
        if config.load_in_4bit and self.device == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto" if self.device == "cuda" else None,
            torch_dtype=torch.float16 if self.device != "cpu" else None,
            trust_remote_code=True,
        )

        # Attach LoRA adapter if available
        lora_path_obj = Path(lora_path)
        if lora_path_obj.exists():
            print(f"Attaching LoRA adapter: {lora_path}")
            self.model = PeftModel.from_pretrained(self.model, lora_path)
        else:
            print(f"Warning: LoRA adapter not found at {lora_path}")
            print("Running with base model only.")

        self.model.eval()
        print("Model loaded successfully.")

    def format_prompt(self, query: str, history: list[dict] = None) -> str:
        """Format the user query with system prompt and history."""
        messages = [{"role": "system", "content": config.system_prompt}]

        if history:
            messages.extend(history)

        messages.append({"role": "user", "content": query})

        # Use Mistral chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return prompt

    def generate(
        self,
        query: str,
        history: list[dict] = None,
        temperature: float = None,
        max_new_tokens: int = None,
    ) -> GenerationResult:
        """Generate a response to a legal query."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        prompt = self.format_prompt(query, history)

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

        # Add disclaimer if not present
        if "DISCLAIMER" not in response and "disclaimer" not in response.lower():
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


def load_model(model_path: str = None, lora_path: str = None):
    """Load the model (convenience function)."""
    inference = get_inference()
    inference.load_model(model_path, lora_path)
    return inference


def generate_response(query: str, history: list[dict] = None) -> str:
    """Generate a response (convenience function)."""
    inference = get_inference()
    result = inference.generate(query, history)
    return result.response
