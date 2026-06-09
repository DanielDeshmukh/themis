"""Unit tests for THEMIS inference pipeline."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestConfig:
    """Tests for configuration."""

    def test_config_loads(self):
        from config import config
        assert config.base_model == "mistralai/Mistral-7B-Instruct-v0.3"
        assert config.temperature == 0.3
        assert config.max_new_tokens == 1024

    def test_config_paths(self):
        from config import config
        assert config.project_root.exists()
        assert config.data_dir.name == "data"
        assert config.model_dir.name == "model"


class TestInference:
    """Tests for inference engine."""

    def test_device_resolution(self):
        from infer import ThemisInference
        engine = ThemisInference()
        device = engine._resolve_device()
        assert device in ("cuda", "mps", "cpu")

    def test_format_prompt(self):
        from infer import ThemisInference
        engine = ThemisInference()
        # Mock tokenizer
        engine.tokenizer = MagicMock()
        engine.tokenizer.apply_chat_template.return_value = "[INST] test [/INST]"
        engine.tokenizer.eos_token = "</s>"

        prompt = engine.format_prompt("What is BNS Section 118?")
        assert prompt is not None


class TestMetrics:
    """Tests for evaluation metrics."""

    def test_citation_accuracy_perfect(self):
        from eval.metrics import citation_accuracy
        predicted = "Section 118 of BNS states..."
        ground_truth = "Section 118 of BNS states..."
        score = citation_accuracy(predicted, ground_truth)
        assert score == 1.0

    def test_citation_accuracy_partial(self):
        from eval.metrics import citation_accuracy
        predicted = "Section 118 of BNS and Section 302 IPC"
        ground_truth = "Section 118 of BNS"
        score = citation_accuracy(predicted, ground_truth)
        assert 0.0 < score <= 1.0

    def test_citation_accuracy_none(self):
        from eval.metrics import citation_accuracy
        predicted = "This is a general answer."
        ground_truth = "Section 118 of BNS"
        score = citation_accuracy(predicted, ground_truth)
        assert score == 0.0

    def test_rouge_l_identical(self):
        from eval.metrics import rouge_l
        text = "Section 118 of BNS states the punishment."
        score = rouge_l(text, text)
        assert score == 1.0

    def test_extract_section_citations(self):
        from eval.metrics import extract_section_citations
        text = "Section 118 of BNS and Section 302 IPC apply here."
        citations = extract_section_citations(text)
        assert "118" in citations


class TestPreprocess:
    """Tests for preprocessing pipeline."""

    def test_clean_text(self):
        from data.preprocess import clean_text
        text = "  Hello   World  "
        cleaned = clean_text(text)
        assert cleaned == "Hello World"

    def test_clean_text_html(self):
        from data.preprocess import clean_text
        text = "<p>Hello</p> <b>World</b>"
        cleaned = clean_text(text)
        assert "<p>" not in cleaned
        assert "Hello" in cleaned

    def test_validate_section_valid(self):
        from data.preprocess import validate_section
        section = {
            "section_number": "118",
            "title": "Punishment for causing hurt",
            "text": "Whoever causes grievous hurt shall be punished.",
        }
        assert validate_section(section) is True

    def test_validate_section_invalid(self):
        from data.preprocess import validate_section
        section = {
            "section_number": "",
            "title": "Title",
            "text": "Short",
        }
        assert validate_section(section) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
