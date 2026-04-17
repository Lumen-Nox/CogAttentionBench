# CogAttentionBench: Measuring Cognitive Attention in Frontier AI Models

### Zihan Zeng (cora zeng)

### Problem Statement

Current AI evaluations test what models *know* but not how they *attend*. Transformer "attention heads" are well-studied computationally, but we lack benchmarks testing whether AI systems exhibit the *cognitive* attention patterns studied in experimental psychology: selective filtering, sustained vigilance, flexible task-switching, anomaly detection, and saliency-driven prioritization.

Existing attention tests (e.g., needle-in-a-haystack) use explicit markers that LLMs trivially parse. CogAttentionBench creates **real interference** — competing signals where the correct response requires suppressing stronger but incorrect signals — grounded in decades of experimental psychology research.

This benchmark is motivated by the **Magnetic Field of Attention (MFA)** framework (Zeng, 2026), which models attentional processes as field-theoretic gradients following F = S/r², unifying five classical attention theories as special cases.

### Task & Benchmark Construction

CogAttentionBench contains **5 tasks** mapped to established experimental paradigms:

| Task | Paradigm | What It Tests |
|------|----------|---------------|
| **Selective Attention** | Stroop (1935); Eriksen flanker (1974) | Extracting targets when statistically stronger distractors compete |
| **Attention Shifting** | Rogers & Monsell (1995) | Switching between confusable rules with irregular patterns |
| **Sustained Attention** | CPT (Rosvold et al., 1956) | Finding unmarked targets buried in increasing filler context |
| **Inattentional Blindness** | Simons & Chabris (1999) | Noticing anomalies during a primary task, then recalling them |
| **Saliency Awareness** | Treisman & Gelade (1980); Itti & Koch (2001) | Ranking competing salient elements by perceptual hierarchy |

Each task uses LLM-specific interference mechanisms:
- **Frequency traps**: High-frequency tokens that prime incorrect responses
- **Context priming**: Surrounding text that biases toward wrong answers
- **Expectation violation**: Patterns that establish then break regularity
- **No explicit markers**: Targets embedded naturally in flowing text

### Dataset

The benchmark contains **38 items** across five tasks (14 + 5 + 8 + 5 + 6), each with verified ground-truth answers. Most items are hand-crafted to create precise interference patterns; sustained_attention items are procedurally generated via seeded randomization to vary filler-passage length. The dataset is self-contained within the benchmark notebooks — no external data dependencies.

Key design choices:
- Items target specific interference mechanisms rather than maximizing item count
- Each item has a single unambiguous correct answer
- Scoring uses exact-match or ranked-list comparison (no subjective judgment)
- All prompts are in English with standardized formatting

### Related Work

**Needle-in-a-Haystack Benchmarks.** The most common approach to testing LLM "attention" embeds target information in long contexts. However, these benchmarks test *retrieval* — finding explicitly marked information — not *selective attention* under interference. CogAttentionBench differs by embedding targets among competing signals that are *more salient* than the target, requiring suppression rather than search.

**Attention Heads vs. Cognitive Attention.** Wu et al. (2025, NeurIPS) characterized transformer attention heads by their "retrieval score" — the frequency with which a head assigns highest attention to the relevant token. This mechanistic analysis of *computational* attention is complementary to our behavioral approach: we test whether *system-level behavior* exhibits cognitive attention patterns regardless of the underlying mechanism.

**Gap.** No existing benchmark tests the full spectrum of cognitive attention (selective, sustained, shifting, inattentional blindness, saliency) using stimuli designed to exploit LLM-specific interference patterns. CogAttentionBench fills this gap.

### Technical Details

The benchmark is built on the **kaggle-benchmarks SDK** with 5 task notebooks:
- `selective_attention_v2` (v4): Stroop-inspired flanker interference with frequency-biased distractors
- `attention_shifting` (v1): Rule-switching between confusable categorization schemes
- `sustained_attention` (v4): Target detection in progressively longer filler passages
- `inattentional_blindness` (v1): Dual-task anomaly detection during primary counting task
- `saliency_awareness` (v1): Multi-element saliency ranking with competing feature dimensions

Each task constructs interference at the prompt level: selective_attention embeds targets among high-frequency distractor tokens that prime incorrect completions; attention_shifting alternates categorization rules at irregular intervals so models cannot rely on pattern repetition; sustained_attention buries unmarked targets in progressively longer filler passages; inattentional_blindness requires noticing anomalies during a primary counting task; and saliency_awareness presents competing feature dimensions (color, size, motion, position) that must be ranked by perceptual hierarchy.

Scoring for each task yields a value in [0, 1] using exact-match or ranked-list comparison (no subjective judgment). The benchmark-level score is the unweighted mean across all 5 tasks.

### Results, Insights, and Conclusions

**28 models evaluated** across the frontier model landscape. While aggregate scores show a ceiling effect (22/28 models achieve 1.00), individual task scores reveal meaningful stratification across cognitive dimensions.

| Model | Selective | Shifting | Sustained | Inattentional | Saliency | **Overall** |
|-------|-----------|----------|-----------|---------------|----------|-------------|
| GPT-5.4 | 1.00 | 1.00 | 1.00 | 1.00 | 0.86 | **0.97** |
| Claude Sonnet 4.6 | 1.00 | 1.00 | 1.00 | 1.00 | 0.72 | **0.94** |
| Gemma 3 4B | 1.00 | 0.75 | 1.00 | 1.00 | 0.63 | **0.88** |
| DeepSeek V3.1 | 0.00 | 1.00 | 1.00 | 1.00 | 0.71 | **0.74** |
| Gemma 3 1B | 0.00 | 0.13 | 0.00 | 0.00 | 0.00 | **0.06** |

*Table: Representative models showing diagnostic stratification. 22 of 28 models score 1.00 overall.*

**Key findings:**

1. **Saliency awareness is the hardest discriminator.** saliency_awareness reveals meaningful stratification even among frontier models:
   - GPT-5.4: 0.86
   - Claude Sonnet 4.6: 0.72
   - DeepSeek V3.1: 0.71
   - This suggests frontier models struggle most with *perceptual hierarchy reasoning* — ranking competing salient elements rather than simply detecting them.

2. **Attention shifting distinguishes small from large models.** Gemma 3 4B scores 0.75 while Gemma 3 1B scores only 0.13, revealing that cognitive flexibility (rule-switching) scales with model size in ways that selective attention does not.

3. **DeepSeek V3.1 shows a specific selective attention deficit.** It scores 0.00 on selective_attention despite strong performance on other tasks, suggesting a systematic vulnerability to frequency-based interference — a pattern reminiscent of Stroop interference in human cognition.

4. **Inattentional blindness reveals a frontier/non-frontier boundary.** All frontier models score 1.00, but Gemma 3 1B scores 0.00, suggesting that detecting anomalies during a primary task requires a minimum model capacity threshold.

**What this benchmark reveals that we couldn't see before:**
- Cognitive attention is *not* a single capability — models can excel at sustained attention while failing at saliency ranking
- The pattern of failures maps onto known human attention phenomena, suggesting that LLM "attention" shares structural properties with cognitive attention despite different mechanisms
- Small models fail in task-specific ways loosely analogous to clinical attention profiles (e.g., selective deficits in attention_shifting scores resembling inattentive patterns)

### Limitations and Future Work

**Current limitations:**

1. **Ceiling effect:** 22 of 28 models score 1.00 on the aggregate benchmark, indicating that the current task difficulty is calibrated below the capability of frontier models for most dimensions. This limits discriminative power at the top of the leaderboard. Future iterations will introduce parametrically harder variants (e.g., deeper embedding, multi-step interference, longer sustained-attention passages).

2. **English-only stimuli:** All tasks use English-language prompts, which may conflate linguistic processing with cognitive attention measurement for non-English-primary models. Cross-lingual versions would strengthen construct validity.

3. **Static item sets:** The current item pool is fixed per task version. Adaptive item generation — where difficulty scales based on model performance — would yield more precise measurements across the capability spectrum.

4. **Single-trial measurement:** Each model receives one pass per task. Human cognitive testing typically uses multiple trials with averaged scores. Multi-trial protocols with variance analysis would improve reliability estimates.

5. **Construct validity:** The mapping from human cognitive attention constructs to LLM behavior is an analogy grounded in functional equivalence, not mechanistic identity. LLMs process text through transformer attention heads, not biological neural circuits. The benchmark measures whether LLM behavior *patterns* resemble cognitive attention phenomena, not whether the underlying mechanisms are the same. This distinction is important when interpreting cross-species (human-to-AI) comparisons.

**What the ceiling effect reveals:** Paradoxically, the high ceiling validates a key finding — basic cognitive attention subtasks (filtering, vigilance, flexibility) are *solved problems* for frontier models. The benchmark's value lies precisely in the dimensions where models diverge: **saliency hierarchy reasoning** and **small-model cognitive fragility**. Future work will selectively harden the high-ceiling tasks while preserving the diagnostic dimensions that already differentiate.

### Organizational Affiliations

Independent high school researcher. Bick International Bilingual School (BIBS), Chengdu, China.

### References & Citations

- Zeng, Z. (2026). The Magnetic Field of Attention: A Unified Framework for Selective Attention. *Under review, Nature Communications.*
- Stroop, J. R. (1935). Studies of interference in serial verbal reactions. *Journal of Experimental Psychology*, 18(6), 643–662.
- Eriksen, B. A., & Eriksen, C. W. (1974). Effects of noise letters upon identification. *Perception & Psychophysics*, 16, 143–149.
- Rogers, R. D., & Monsell, S. (1995). Costs of a predictable switch between simple cognitive tasks. *Journal of Experimental Psychology: General*, 124(2), 207–231.
- Rosvold, H. E., et al. (1956). A continuous performance test of brain damage. *Journal of Consulting Psychology*, 20(5), 343–350.
- Simons, D. J., & Chabris, C. F. (1999). Gorillas in our midst. *Perception*, 28(9), 1059–1074.
- Treisman, A. M., & Gelade, G. (1980). A feature-integration theory of attention. *Cognitive Psychology*, 12(1), 97–136.
- Itti, L., & Koch, C. (2001). Computational modelling of visual attention. *Nature Reviews Neuroscience*, 2(3), 194–203.
- DeepMind (2026). Measuring progress toward AGI: A cognitive framework. Google DeepMind Technical Report.
