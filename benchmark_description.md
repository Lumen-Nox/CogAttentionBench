# CogAttentionBench — Benchmark Description

*For pasting into the Kaggle benchmark description field (Markdown)*

---

## What is CogAttentionBench?

CogAttentionBench measures whether AI models exhibit cognitive attention patterns studied in experimental psychology. Unlike existing benchmarks that test knowledge retrieval, CogAttentionBench tests **attention allocation** — the ability to filter, sustain, shift, and prioritize information under interference.

## Why does this matter?

Transformer "attention heads" are well-understood computationally, but we lack benchmarks testing whether AI systems exhibit the *cognitive* attention patterns found in human psychology: selective filtering, sustained vigilance, flexible task-switching, anomaly detection, and saliency-driven prioritization.

Current attention tests (e.g., needle-in-haystack) use explicit markers that LLMs trivially parse. CogAttentionBench creates **real interference** — competing signals where the correct response requires suppressing stronger but incorrect signals.

## Five Tasks

| Task | Paradigm | What it tests |
|------|----------|---------------|
| **Selective Attention** | Stroop (1935); Eriksen flanker (1974) | Extracting targets when statistically stronger distractors compete |
| **Attention Shifting** | Rogers & Monsell (1995) | Switching between confusable rules with irregular patterns |
| **Sustained Attention** | CPT (Rosvold et al., 1956) | Finding unmarked targets buried in increasing filler context |
| **Inattentional Blindness** | Simons & Chabris (1999) | Noticing anomalies during a primary task, then recalling them |
| **Saliency Awareness** | Treisman & Gelade (1980); Itti & Koch (2001) | Ranking competing salient elements by perceptual hierarchy |

## Key Design Principles

- **LLM-specific interference**: Frequency traps, context priming, expectation violation — exploiting how language models process information
- **No explicit markers**: Targets embedded naturally in flowing text, not flagged with `[TASK]` tags
- **Grounded in psychology**: Each task maps to established experimental paradigms with known human baselines
- **Differential predictions**: Where AI attention diverges from human attention reveals what "attention" means in each system

## Connection to MFA Theory

This benchmark is informed by the *Magnetic Field of Attention* (MFA) framework (Zeng, 2026), which models attentional processes as field-theoretic gradients. CogAttentionBench operationalizes MFA's key predictions about attention resource competition, gradient decay, and load-dependent processing.

## Citation

```bibtex
@misc{zeng2026cogattentionbench,
  author = {Zeng, Zihan},
  title = {CogAttentionBench: Probing Cognitive Attention Mechanisms in Frontier AI Models},
  year = {2026},
  url = {https://www.kaggle.com/benchmarks/corazeng/cogattentionbench}
}
```
