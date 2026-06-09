# THEMIS — Security Audit

## Overview

THEMIS is an offline CLI tool. The attack surface is minimal compared to web-deployed systems. However, several security considerations apply to the training pipeline, model deployment, and user interaction.

---

## Threat Model

### In Scope
1. **Data Pipeline Security** — Scraper integrity, API key management, data sanitization
2. **Training Security** — Model weight integrity, training data poisoning
3. **Inference Security** — Prompt injection, output safety, disclaimers
4. **Distribution Security** — Package integrity, dependency vulnerabilities

### Out of Scope (v1.0)
- Multi-user authentication (CLI is single-user)
- Network security (offline-first design)
- Encryption at rest (model weights are public)

---

## Data Pipeline Security

### API Key Management

**Risk:** API key exposure during synthetic generation.

| Control | Implementation |
|---------|----------------|
| Environment variables | Store `GROQ_API_KEY` in `.env`, never commit |
| `.gitignore` | Ensure `.env` is in `.gitignore` |
| No hardcoded keys | Scan codebase for leaked credentials |
| Least privilege | Use API key with minimal necessary permissions |

**Status:** Not implemented yet — implement before Phase 1.

### Scraper Integrity

**Risk:** Malicious content injection through scraped legal texts.

| Control | Implementation |
|---------|----------------|
| Source validation | Only scrape from official sources (indiacode.nic.in, indiankanoon.org) |
| Content sanitization | Strip HTML, scripts, non-legal content |
| Deduplication | Remove duplicate sections to prevent training bias |
| Manual review | Spot-check scraped sections against official Bare Acts |

### Training Data Poisoning

**Risk:** Adversarial Q&A pairs that embed harmful legal advice.

| Control | Implementation |
|---------|----------------|
| API prompt engineering | Instruct model to cite only official sections |
| Output validation | Verify cited section numbers exist in parsed data |
| Human review | Review sample of generated Q&A pairs |
| Disclaimer injection | Every training pair includes disclaimer language |

---

## Model Security

### Weight Integrity

**Risk:** Tampered LoRA adapter weights.

| Control | Implementation |
|---------|----------------|
| HuggingFace Hub | Weights hosted on trusted platform |
| Checksum verification | Verify adapter SHA256 hash on download |
| Reproducible builds | Training script + config enables independent verification |

### Prompt Injection

**Risk:** User input designed to override system prompt.

| Control | Implementation |
|---------|----------------|
| System prompt isolation | System prompt prepended, not user-controlled |
| Input sanitization | Strip special tokens from user input |
| Output filtering | Post-process to ensure legal-only responses |
| Refusal training | Model trained to decline non-legal queries |

**Residual Risk:** Low — prompt injection on local CLI has limited impact.

---

## Inference Security

### Output Safety

**Risk:** Model generates harmful or incorrect legal guidance.

| Control | Implementation |
|---------|----------------|
| Disclaimer | Every response includes "not legal advice" disclaimer |
| Scope limitation | Model refuses queries outside statutory law |
| Citation requirement | Model must cite specific sections, reducing speculation |
| Refusal training | Model trained to say "consult a lawyer" when uncertain |

### User Safety

**Risk:** User relies on model output as actual legal advice.

| Control | Implementation |
|---------|----------------|
| Prominent disclaimer | Displayed in CLI output for every response |
| Scope warnings | Clear documentation of what THEMIS does NOT do |
| No liability | MIT license, no warranty clause |

---

## Distribution Security

### Package Integrity

**Risk:** Compromised package on PyPI.

| Control | Implementation |
|---------|----------------|
| Signed releases | Sign releases with GPG key (if PyPI supports) |
| Version pinning | Pin exact dependency versions in pyproject.toml |
| Dependency audit | Run `pip-audit` before release |
| Source transparency | Open source, auditable codebase |

### Dependency Risks

| Dependency | Risk | Mitigation |
|------------|------|------------|
| transformers | Large, complex library | Pin version, monitor CVEs |
| peft | LoRA implementation | Pin version |
| torch | CUDA compatibility | Pin version matching Kaggle environment |
| typer | CLI framework | Low risk, well-maintained |
| rich | Terminal formatting | Low risk, well-maintained |

**Action:** Run `pip-audit` before each release.

---

## Legal Disclaimer Compliance

### Required Disclaimer Text

```
⚠ DISCLAIMER: This is legal orientation, not legal advice. 
Consult a qualified advocate for your specific situation.
```

### Implementation Requirements

1. **CLI Output:** Disclaimer displayed in every response panel
2. **Training Data:** Every Q&A pair includes disclaimer in output
3. **System Prompt:** Model instructed to always include disclaimer
4. **README:** Clear documentation that THEMIS is not a lawyer substitute

---

## Security Checklist

### Pre-Release (v1.0)

- [ ] No hardcoded API keys in codebase
- [ ] `.env` in `.gitignore`
- [ ] All training data sanitized
- [ ] Disclaimer in every training Q&A pair
- [ ] Disclaimer in CLI output
- [ ] System prompt includes scope limitations
- [ ] Model refuses out-of-scope queries
- [ ] Dependencies audited with `pip-audit`
- [ ] No secrets in model weights
- [ ] README documents limitations clearly

### Ongoing

- [ ] Monitor dependency CVEs
- [ ] Review scraped data sources periodically
- [ ] Update legal disclaimers as regulations change
- [ ] Audit training data quality periodically

---

## Incident Response

If a security issue is discovered:

1. **Severity Assessment** — Is it a data leak, model tampering, or output safety issue?
2. **Immediate Mitigation** — Disable affected functionality
3. **Root Cause Analysis** — Identify source of vulnerability
4. **Patch and Release** — Fix issue, bump version, publish update
5. **Disclosure** — Document in GitHub Security Advisories if applicable

---

## Notes

- v1.0 has minimal attack surface (local CLI, no network)
- v2.0 (web UI) will require additional security review
- v2.0 will need authentication, rate limiting, input validation
- Consider OWASP Top 10 when building web interface
