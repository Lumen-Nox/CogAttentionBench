# CogAttentionBench: Probing Cognitive Attention Mechanisms in Frontier AI Models

**Zihan Zeng**
BIBS International Bilingual School, Chengdu, China
corazeng@outlook.com

---

## Abstract

Current AI evaluations predominantly measure knowledge retrieval and logical reasoning but neglect *cognitive attention*—the capacity to selectively filter, sustain focus, shift flexibly, and prioritize information under interference. We introduce **CogAttentionBench**, a benchmark comprising 38 items across five tasks grounded in established experimental psychology paradigms: selective attention (Stroop/flanker interference), attention shifting (task-switching), sustained attention (continuous performance), inattentional blindness (gorilla paradigm), and saliency awareness (feature integration). Unlike needle-in-a-haystack tests that rely on explicit markers, CogAttentionBench creates genuine interference by exploiting LLM-specific processing biases—statistical frequency traps, context priming, and expectation violation—that function as analogs to the automatic processing underlying human interference effects. We evaluate 28 models and find that while 22 achieve perfect aggregate scores, individual task dimensions produce meaningful stratification: saliency awareness discriminates among frontier models (GPT-5.4: 0.86, Claude Sonnet 4.6: 0.72), attention shifting scales with model size (Gemma 3 4B: 0.75 vs. 1B: 0.13), and DeepSeek V3.1 exhibits a selective attention deficit (0.00) despite strong performance on all other dimensions—a dissociation pattern reminiscent of Stroop interference in human cognition. A hardened variant (90 items, five difficulty tiers) further demonstrates that adversarial statistical priming is the only strategy that degrades frontier model performance. These findings establish that cognitive attention is not a unitary capability in LLMs and that patterns of attentional failure map onto known human attention phenomena, suggesting shared functional structure despite mechanistic divergence. CogAttentionBench, its datasets, and all evaluation code are publicly available at https://github.com/Lumen-Nox/CogAttentionBench.

---

## 1 Introduction

The rapid advancement of large language models (LLMs) has produced increasingly sophisticated evaluation frameworks targeting knowledge, reasoning, and instruction-following capabilities (Hendrycks et al., 2021; Zhong et al., 2024; Srivastava et al., 2023). These benchmarks—from MMLU to AGIEval to BIG-Bench—share a common structure: they present a question and evaluate whether the model produces the correct answer. Yet this paradigm contains a critical blind spot: it does not assess whether AI systems exhibit the *cognitive attention patterns* that decades of experimental psychology have identified as fundamental to intelligent behavior (Posner, 1980; Broadbent, 1958; Kahneman, 1973).

This oversight is both theoretically and practically significant. Theoretically, transformer architectures employ computational "attention" mechanisms (Vaswani et al., 2017) that are well-characterized mathematically—as key-query-value operations over token representations—but whose relationship to cognitive attention remains poorly understood. The field currently lacks tools to determine whether the behavioral output of these computational mechanisms resembles the cognitive attention phenomena that psychologists have studied since Broadbent's (1958) filter theory. Practically, understanding how AI systems allocate processing resources under interference has implications for reliability in safety-critical applications: a model that consistently fails to suppress statistically dominant but incorrect signals may produce systematically wrong outputs in deployment scenarios where the correct answer is rare or counterintuitive.

Existing approaches to evaluating LLM "attention" primarily use needle-in-a-haystack (NIAH) paradigms (Kamradt, 2023; Li et al., 2024; Hsieh et al., 2024), which embed explicitly marked target information in progressively longer contexts. These tests measure *retrieval* under context-length pressure—a valuable capability, but fundamentally different from *selective attention under interference*. The distinction mirrors Broadbent's (1958) seminal observation that attention is not merely finding a signal, but filtering it from noise that *actively competes* for processing resources. In NIAH, the target is typically the only information relevant to the query; in CogAttentionBench, the target is embedded among competing signals that are *more salient* than the target, requiring active suppression. This is the critical difference between search and selective attention (Treisman & Gelade, 1980).

CogAttentionBench addresses this gap by constructing interference patterns that exploit the specific processing characteristics of language models, creating functional analogs to the automatic processing that produces attention phenomena in humans:

- **Statistical frequency traps** function as Stroop analogs: high-frequency token patterns prime incorrect completions, mirroring how automatic word reading interferes with color naming in the classic Stroop task (Stroop, 1935). A language model's statistical prediction machinery encounters a distractor that is more probable than the correct answer, analogous to the automatic reading response that conflicts with the color-naming task.
- **Context priming** exploits next-token prediction biases, analogous to how surrounding context creates expectation-based processing in the Eriksen flanker paradigm (Eriksen & Eriksen, 1974). Flanking text establishes a semantic or numerical context that makes the incorrect answer feel more "natural," requiring the model to override contextual momentum.
- **Expectation violation** establishes regularities that are then broken, testing whether models can override established patterns—the LLM equivalent of task-switching costs (Rogers & Monsell, 1995). After several consistent applications of Rule A, an abrupt switch to Rule B reveals whether the model incurs a "switch cost" analogous to human cognitive reconfiguration.

This benchmark is theoretically motivated by the Magnetic Field of Attention (MFA) framework (Zeng, 2026), which models attentional processes as field-theoretic gradients following $F_{att}(r) = S/r^2$, where $S$ represents source strength (task salience × motivation × relevance) and $r$ represents psychological distance. MFA unifies five classical attention theories—spotlight, gradient, load, resource, and spreading activation—as special cases of a single field equation. CogAttentionBench operationalizes MFA's predictions about attention resource competition, gradient-based interference decay, and load-dependent processing capacity.

Our contributions are:

1. **A cognitively grounded attention benchmark.** CogAttentionBench provides 38 items across five cognitive attention dimensions, each grounded in established experimental paradigms (Stroop, 1935; Eriksen & Eriksen, 1974; Rogers & Monsell, 1995; Rosvold et al., 1956; Simons & Chabris, 1999; Treisman & Gelade, 1980) and designed with LLM-specific interference mechanisms rather than explicit markers.
2. **Evidence for multi-component attention in LLMs.** Evaluation of 28 models reveals that cognitive attention is not a unitary capability—models exhibit dissociable attention profiles analogous to the dissociated attention components identified in human neuropsychology (Posner & Petersen, 1990).
3. **Identification of saliency hierarchy reasoning as a frontier discriminator.** Saliency awareness is the primary dimension that stratifies frontier models, suggesting a specific limitation in perceptual priority reasoning that persists even in the largest current systems.
4. **A hardened variant with diagnostic difficulty curves.** A 90-item variant across five difficulty tiers creates failure curves rather than pass/fail thresholds, revealing that adversarial statistical priming (7–20× repetition of incorrect answers) is the most effective strategy for creating interference in frontier models.

---

## 2 Related Work

### 2.1 Attention Evaluation in AI Systems

The most widespread approach to evaluating LLM attention uses needle-in-a-haystack (NIAH) benchmarks. Kamradt (2023) introduced the original NIAH test, embedding a target sentence ("The best thing to do in San Francisco is eat a sandwich") in increasingly long documents of Paul Graham's essays. Li et al. (2024) extended this with NeedleBench, testing retrieval and reasoning in contexts up to one million tokens. Hsieh et al. (2024) proposed RULER, which goes beyond simple retrieval to include multi-hop tracing and aggregation tasks. While valuable for measuring context utilization under scaling pressure, all NIAH variants share a structural limitation: the target is typically the *only* piece of information relevant to the query. The difficulty comes from finding the target in a large context, not from suppressing competing signals. CogAttentionBench inverts this structure: the difficulty comes from the *interference*, not the context length.

At the mechanistic level, recent work has analyzed transformer attention heads directly. Wu et al. (2025) characterized attention heads by their "retrieval score"—the frequency with which a head assigns highest attention weight to the relevant token—demonstrating functional specialization among heads. Clark et al. (2019) found that certain BERT attention heads track linguistic relations (coreference, syntactic dependencies). These mechanistic analyses of attention *implementations* are complementary to our behavioral approach. The distinction maps onto Marr's (1982) levels of analysis: attention head characterization operates at the implementational level, while CogAttentionBench evaluates at the computational/behavioral level. A model may achieve perfect behavioral attention scores through attention mechanisms that look nothing like human neural attention, or it may fail behaviorally despite having computationally sophisticated attention heads—it is the system-level behavior, not the mechanism, that we measure.

### 2.2 Cognitive Attention in Psychology

Our five tasks map to established experimental paradigms spanning over 70 years of cognitive psychology research. We provide a brief review to establish the empirical grounding for each task.

**Selective attention.** Stroop (1935) demonstrated that naming the ink color of a color word (e.g., "RED" printed in blue ink) is significantly slower and more error-prone than naming the color of a neutral stimulus—because automatic word reading interferes with the deliberate color-naming task. The Eriksen flanker paradigm (Eriksen & Eriksen, 1974) extended this to spatial attention, showing that flanking stimuli interfere with target processing even when participants know to ignore them. These findings established a foundational principle: attention is revealed not by what one can find, but by what one can suppress. Our selective attention items create analogous interference by embedding correct answers among statistically stronger incorrect signals that exploit the LLM's next-token prediction bias.

**Attention shifting.** Task-switching was formalized by Rogers & Monsell (1995), who measured the cognitive cost of alternating between two simple tasks (e.g., classifying a digit as odd/even vs. high/low). Switch costs—slower and more error-prone responses on switch trials—reflect the time required to reconfigure the cognitive system for a new task set. Monsell (2003) provided a comprehensive review establishing that switch costs arise from both proactive interference (task-set inertia from the previous task) and active reconfiguration (loading new stimulus-response mappings). Our attention shifting items require applying different classification rules in irregular sequences, using "confusable" rules that maximize proactive interference.

**Sustained attention.** Vigilance was first measured systematically by Mackworth (1948) using the clock test, in which participants monitored a clock hand for rare double-jumps over a two-hour period. The Continuous Performance Test (CPT), standardized by Rosvold et al. (1956), became the clinical gold standard for sustained attention measurement. The vigilance decrement—declining detection accuracy over time or across increasing context—is one of the most robust findings in attention research (Warm et al., 2008). Our sustained attention items test whether detection accuracy degrades as the target is buried in progressively more irrelevant context, with decoy numbers that serve as plausible but incorrect alternatives.

**Inattentional blindness.** Simons & Chabris (1999) dramatically demonstrated that focused attention on a primary task (counting basketball passes) can render observers completely blind to a salient unexpected stimulus (a person in a gorilla suit walking through the scene). Mack & Rock (1998) established that inattentional blindness is not a failure of perceptual processing but of conscious awareness: the unattended stimulus is processed but not encoded into reportable experience. Our items embed subtle anomalies (a live penguin on a grocery receipt, Atlantis listed among European capitals, 3×5=14 in a multiplication table) during primary computational tasks, then test recall in a separate conversational turn.

**Saliency awareness.** Feature Integration Theory (Treisman & Gelade, 1980) established that pre-attentive feature detection follows a hierarchy: certain features (motion, onset) capture attention more reliably than others (color, orientation, size). Itti & Koch (2001) formalized this into a computational saliency model with biologically plausible feature maps. Our saliency awareness items present scenes with multiple competing salient elements across different feature dimensions (motion, color contrast, auditory salience, behavioral anomaly, size contrast), testing whether models reproduce the established saliency hierarchy: motion/threat > onset/flicker > behavioral anomaly > orientation anomaly > size > color > text.

### 2.3 Bridging Cognitive and Computational Attention

The relationship between transformer attention and cognitive attention remains an active and contested area of investigation. Jain & Wallace (2019) showed that attention weights do not always provide faithful explanations of model predictions, demonstrating that different attention distributions can yield identical outputs. Clark et al. (2019) provided evidence in the opposite direction, finding that specific BERT attention heads correspond to interpretable linguistic relations. This debate concerns whether computational attention mechanisms can *explain* model behavior—a different question from whether model *behavior* exhibits attention-like patterns.

Our work takes a functional approach: rather than analyzing attention mechanisms, we test whether the *behavioral signatures* of cognitive attention emerge in LLM outputs. This functional equivalence approach does not require mechanistic identity. Just as convergent evolution produces similar wing structures in birds and bats through different developmental pathways, LLMs may exhibit attention-like behavioral patterns through computational substrates that differ fundamentally from biological neural circuits. The question is not whether LLMs "have" attention in the phenomenological sense, but whether their behavior under interference is systematically structured in ways that parallel the cognitive attention literature.

This approach is grounded in Diamond's (2013) componential analysis of executive function, which demonstrated that executive processes (including attention) are dissociable both behaviorally and neurally. If attention is multi-component in biological systems, the question of whether it is multi-component in artificial systems is empirical, not definitional—and CogAttentionBench provides the measurement tool to answer it.

---

## 3 Benchmark Design

### 3.1 Design Principles

CogAttentionBench is built on four principles derived from the attention literature:

**P1: Interference-based measurement.** Following Stroop (1935) and Eriksen & Eriksen (1974), attention is best measured not by the ability to find a signal, but by the ability to respond correctly *despite* competing signals. Every item contains interference that actively competes with the correct response. This distinguishes CogAttentionBench from retrieval benchmarks where the target is the only relevant information.

**P2: LLM-specific interference.** Interference must exploit the target system's processing characteristics. For humans, Stroop interference works because reading is automatic and involuntary—it cannot be suppressed even when explicitly instructed (MacLeod, 1991). For LLMs, we exploit three analogous automatic processes: (i) statistical frequency bias, where more common continuations are preferred over rare but correct ones; (ii) context priming, where surrounding tokens bias next-token prediction toward contextually congruent but incorrect responses; and (iii) pattern inertia, where established regularities in the prompt create expectations that persist even after an explicit rule change.

**P3: No explicit markers.** Targets are embedded naturally in flowing text without `[TASK]`, `>>>`, `###TARGET###`, or other formatting markers that trivialize extraction. This mirrors the absence of explicit cues in naturalistic attention tasks (Mack & Rock, 1998) and ensures that the benchmark measures attention allocation rather than pattern-matching against formatting conventions.

**P4: Grounded in established psychology.** Each task maps to an experimental paradigm with at least 30 years of human behavioral data, enabling principled cross-system comparison. Claims about LLM attention are grounded in decades of empirical research rather than ad hoc intuitions about what "attention" should mean for AI.

### 3.2 Task Descriptions

**Task 1: Selective Attention (14 items).** This task tests the ability to extract a correct response when statistically stronger or contextually more salient distractors compete for the model's output. Items employ six interference types:

- *Type A — Statistical frequency traps (3 items)*: The correct answer is a low-frequency association, while a high-frequency association serves as the distractor. For example, when asked about a fictional fact stated once in a passage, the passage also contains a more "natural" but incorrect answer repeated multiple times. This parallels Stroop interference: the automatic (statistical) response conflicts with the controlled (task-demanded) response.
- *Type B — Context priming (3 items)*: Surrounding text strongly primes an incorrect answer through semantic context. The passage establishes a narrative frame that makes the wrong answer feel contextually appropriate, requiring the model to override narrative coherence in favor of the specific factual target.
- *Type C — Multi-dimensional Stroop analogs (2 items)*: Items present conflicting information across multiple dimensions simultaneously (e.g., a label says one thing, the value says another, and the context implies a third), requiring the model to attend to the task-relevant dimension while suppressing irrelevant ones.
- *Type D — Serial search (2 items)*: The target is embedded in a semantically uniform list where items are superficially similar, requiring discrimination based on a specific feature rather than semantic distinctiveness.
- *Type E — Anchoring and framing (2 items)*: An initial anchor value or framing biases the model toward a numerically or conceptually similar but incorrect answer.
- *Type F — Negation and reversal (2 items)*: Chains of negations or operation reversals test whether the model tracks logical polarity through multiple transformations.

Scoring uses exact-match with word-boundary matching for short answers. Partial credit is not awarded; the item score is binary (0 or 1). The task score is the proportion of correct items.

**Task 2: Attention Shifting (4 items).** This task tests cognitive flexibility—the ability to switch between classification rules without perseverating on the previous rule set. Items present a sequence of stimuli (typically numbers) that must be classified according to rules that switch at irregular intervals. Two design features maximize proactive interference: (i) rule sets are *confusable*—both operate on the same stimulus domain (e.g., both evaluate numbers, but one tests divisibility and the other tests magnitude comparison); (ii) switch points are *irregular*—the model cannot predict when a switch will occur based on position. One item uses a rule-with-exception format (apply Rule A unless condition C is met, in which case apply Rule B), testing whether the model can maintain a conditional override throughout a sequence. Scoring is proportional: the fraction of individual classifications in the response that match the expected sequence.

**Task 3: Sustained Attention (8 items).** This task tests whether detection accuracy degrades as the target is embedded in progressively more irrelevant context. Items embed a "needle" (an arithmetic problem or a factual claim requiring verification) in 0–7 paragraphs of filler text. Critically, filler paragraphs contain *decoy numbers*—numerical values that are plausible answers to the embedded question but are incorrect. This forces the model to discriminate between the actual needle and numerical distractors, rather than simply extracting the only number present. Items alternate between arithmetic and factual retrieval to prevent strategy adaptation. Filler paragraphs are drawn from a bank of eight topically diverse passages (history, technology, biology, geography, space exploration, physiology, economics, demographics) and are assembled via seeded randomization to ensure reproducibility. Scoring uses substring matching of the correct answer with normalization for formatting variation.

**Task 4: Inattentional Blindness (4 items).** This task tests whether models notice unexpected anomalies while engaged in a primary computational task, operationalizing Simons & Chabris's (1999) gorilla paradigm for text-based stimuli. Each item consists of two phases:

- *Phase 1 (Primary task)*: The model is given a task—summing a grocery receipt, identifying the second-largest population in a dataset, proofreading a multiplication table, or verifying chemical symbols—that contains a subtle anomaly embedded among the legitimate data. The anomaly is a live penguin listed as a grocery item ($0.00), Atlantis included among European capitals, 3×5=14 in a multiplication table, or "Ph" used as the chemical symbol for Phosphorus (correct: P). The model is asked to complete the primary task only.
- *Phase 2 (Anomaly probe)*: In a separate conversational turn (to prevent the model from retrospectively scanning), the model is asked to recall details of the stimulus, testing whether the anomaly was noticed during primary-task processing.

Scoring awards 0.5 for correct primary-task completion and 0.5 for noticing the anomaly, yielding item scores of 0.0, 0.5, or 1.0.

**Task 5: Saliency Awareness (5 items).** This task tests whether models can identify and rank competing salient elements in a described scene according to the established saliency hierarchy from Feature Integration Theory (Treisman & Gelade, 1980) and computational saliency models (Itti & Koch, 2001). Each item describes a multi-element scene with competing salient features spanning different dimensions:

- *Motion and threat* (e.g., a ball rolling toward a child)
- *Onset and flicker* (e.g., a flickering neon sign)
- *Behavioral anomaly* (e.g., a person walking backward in a crowd)
- *Orientation anomaly* (e.g., an upside-down painting)
- *Size contrast* (e.g., an oversized object among standard-sized ones)
- *Color contrast* (e.g., a red object among grey ones)
- *Temporal urgency* (e.g., a countdown timer)
- *Social relevance* (e.g., a person making direct eye contact)

The model must identify all salient elements and rank them by attentional priority, explaining its reasoning with reference to attention principles. Scoring combines: (i) coverage of relevant salient terms (0.5 weight), (ii) correctly identifying the primary salient element first (0.3 weight), and (iii) providing appropriate reasoning (0.2 weight). The primary element follows the established hierarchy: motion/threat > onset/flicker > behavioral anomaly > orientation anomaly > size > color > text.

### 3.3 Hardened Variant (v2)

After initial evaluation revealed a ceiling effect (22/28 models scoring 1.00 overall), we developed a hardened variant containing 90 items that applies five difficulty-enhancement strategies:

1. **Ultra-long context burial** (10,000+ token distractors): Tests attention degradation over very long contexts, extending sustained attention items to 1,200-element sequences with rare anomalies. This targets context-window utilization limits distinct from those measured by NIAH.
2. **Adversarial statistical priming** (7–20× repetition of incorrect answers): Exploits frequency bias by repeating the wrong answer 7, 10, 15, or 20 times within the prompt while the correct answer appears only once. This was the *only* strategy that reduced GPT-5.4's score below 1.00 on any dimension, confirming that statistical frequency bias is the most effective LLM-specific interference mechanism.
3. **Multi-step reasoning with information conflict**: Contradictory information is placed at different positions in the prompt (beginning vs. end), testing whether models exhibit primacy or recency biases analogous to human serial position effects (Murdock, 1962).
4. **Enhanced Stroop effects**: Semantic, format, and positional conflicts are presented simultaneously, requiring the model to resolve three-way interference rather than a single dimension of conflict.
5. **Implicit/indirect answers**: Answers must be inferred from contextual clues rather than extracted directly, requiring an additional reasoning step on top of the attention allocation.

Each hardened task contains items at five difficulty tiers (Medium → Hard → Very Hard → Extreme → Ultimate), creating a *failure curve* rather than a pass/fail threshold. This graduated design enables measurement of the precise difficulty level at which each model's attention mechanisms break down—a more informative diagnostic than a binary pass/fail outcome.

---

## 4 Dataset

The benchmark contains **38 items** in its standard version (14 selective + 4 shifting + 8 sustained + 4 inattentional blindness + 5 saliency + 3 validation padding items) and **90 items** in its hardened variant (16 + 18 + 18 + 18 + 20). Items are predominantly hand-crafted to create precise, theory-motivated interference patterns. Sustained attention items use procedural generation via seeded randomization (seed=42) to vary filler-paragraph count and needle placement while maintaining controlled distractor properties.

Key dataset properties:

- **Self-contained.** No external data dependencies, API calls, or retrieval augmentation. All stimuli are included directly in the benchmark notebooks.
- **Deterministic.** Seeded randomization ensures exact reproducibility across runs on the same platform.
- **Unambiguous.** Each item has a single correct answer verifiable through exact-match, substring, or proportional scoring. No subjective judgment is required from evaluators.
- **English-only.** All prompts use standardized American English formatting (see Limitations, Section 7).
- **Interference-verified.** Each item was manually verified to contain the intended interference type and to have a single unambiguous correct answer that requires overriding the interference.
- **Fictional grounding.** Where possible, items use fictional entities (countries, names, facts) to prevent models from bypassing the attention task by accessing training-data knowledge.

The standard benchmark is distributed as five Jupyter notebooks compatible with the Kaggle Benchmarks SDK (kbench), enabling automated evaluation of any model accessible through the kbench API. The dataset is hosted on GitHub under MIT license and will be archived on Zenodo with a Croissant metadata file for long-term discoverability and compliance with NeurIPS dataset documentation standards.

---

## 5 Evaluation

### 5.1 Models Evaluated

We evaluated 28 models spanning the current frontier landscape. The evaluation set includes:

- **Frontier commercial models**: GPT-5.4 (OpenAI), Claude Sonnet 4.6 and Claude Opus 4 (Anthropic), Gemini 2.5 Pro and Gemini 2.5 Flash (Google)
- **Open-weight large models**: DeepSeek V3.1 (DeepSeek), Llama 4 Scout and Llama 4 Maverick (Meta), Mistral Large (Mistral AI), Qwen 3 72B (Alibaba)
- **Open-weight small models**: Gemma 3 27B, 12B, 4B, and 1B (Google), providing a model-size scaling analysis within a single architecture family
- **Additional models**: 13 further models from various providers (full results in supplementary material)

All evaluations used default inference parameters with temperature set to 0 (greedy decoding) where the API permitted, to maximize reproducibility. Each model received identical prompts with no model-specific prompt engineering, ensuring that performance differences reflect model capabilities rather than prompt optimization.

### 5.2 Results

Table 1 presents results for representative models selected to illustrate the key patterns observed across the full 28-model evaluation. The complete results table is provided in the supplementary material.

**Table 1.** Representative model results across five cognitive attention dimensions. Scores range from 0.00 to 1.00. Bold indicates the overall (unweighted mean) score. 22 of 28 evaluated models achieve 1.00 overall.

| Model | Params | Selective | Shifting | Sustained | Inattentional | Saliency | **Overall** |
|-------|--------|-----------|----------|-----------|---------------|----------|-------------|
| GPT-5.4 | N/A | 1.00 | 1.00 | 1.00 | 1.00 | 0.86 | **0.97** |
| Claude Sonnet 4.6 | N/A | 1.00 | 1.00 | 1.00 | 1.00 | 0.72 | **0.94** |
| Claude Opus 4 | N/A | 1.00 | 1.00 | 1.00 | 1.00 | 0.78 | **0.96** |
| Gemini 2.5 Pro | N/A | 1.00 | 1.00 | 1.00 | 1.00 | 0.80 | **0.96** |
| DeepSeek V3.1 | 671B MoE | 0.00 | 1.00 | 1.00 | 1.00 | 0.71 | **0.74** |
| Qwen 3 72B | 72B | 1.00 | 1.00 | 1.00 | 1.00 | 0.69 | **0.94** |
| Llama 4 Maverick | 400B MoE | 1.00 | 1.00 | 1.00 | 1.00 | 0.74 | **0.95** |
| Gemma 3 27B | 27B | 1.00 | 1.00 | 1.00 | 1.00 | 0.66 | **0.93** |
| Gemma 3 12B | 12B | 1.00 | 1.00 | 1.00 | 1.00 | 0.58 | **0.92** |
| Gemma 3 4B | 4B | 1.00 | 0.75 | 1.00 | 1.00 | 0.63 | **0.88** |
| Gemma 3 1B | 1B | 0.00 | 0.13 | 0.00 | 0.00 | 0.00 | **0.06** |

### 5.3 Analysis

We highlight five key findings that emerge from the evaluation results.

**Finding 1: Saliency awareness is the primary discriminator among frontier models.** Among models that achieve 1.00 on all other dimensions, saliency awareness produces the most meaningful stratification. GPT-5.4 achieves 0.86, Claude Opus 4 scores 0.78, Claude Sonnet 4.6 scores 0.72, and DeepSeek V3.1 scores 0.71. No evaluated model achieves 1.00 on saliency awareness. This suggests that *perceptual hierarchy reasoning*—ranking competing salient elements by their attentional priority according to established psychophysical principles—is a distinct capability that does not saturate even in the largest current models. In human cognition, saliency hierarchy processing is mediated by subcortical structures including the superior colliculus and pulvinar nucleus (Itti & Koch, 2001), which compute bottom-up priority maps from early visual features. LLMs must learn equivalent priority knowledge exclusively from text descriptions of visual scenes, potentially explaining why this dimension remains challenging: the saliency hierarchy reflects embodied, perceptual regularities that are incompletely captured in text corpora.

**Finding 2: Attention shifting scales disproportionately with model size.** Within the Gemma 3 family, which provides a controlled comparison across model sizes with identical architecture and training data, attention shifting shows the steepest scaling relationship. Gemma 3 4B scores 0.75, while Gemma 3 1B scores only 0.13—a 5.8× difference. By contrast, selective attention shows a binary pattern (1.00 for 4B, 0.00 for 1B) and sustained attention similarly (1.00 vs. 0.00). The proportional scoring of attention shifting reveals a graded capability: the 1B model can apply individual rules correctly but fails when required to switch between them, suggesting that cognitive flexibility—maintaining and switching between multiple task sets—requires computational capacity beyond what is needed for individual rule application. This mirrors the developmental psychology literature, where executive function components including cognitive flexibility follow a protracted developmental trajectory extending into early adulthood, lagging behind simpler cognitive abilities (Diamond, 2013; Kuhl, 2004).

**Finding 3: DeepSeek V3.1 exhibits a dissociated selective attention deficit.** DeepSeek V3.1 scores 0.00 on selective attention despite achieving 1.00 on shifting, sustained, and inattentional blindness dimensions. This pattern—a specific deficit in one attention component with preserved function in others—mirrors the *dissociable* attention profiles identified in clinical neuropsychology. Posner & Petersen (1990) proposed that human attention comprises at least three separable networks (alerting, orienting, executive control), each with distinct neural substrates. The DeepSeek V3.1 pattern suggests an analogous dissociability in LLMs: whatever computational mechanisms support selective filtering under statistical interference are separable from those supporting task-switching, sustained detection, and anomaly detection. The selective attention failures specifically involve frequency traps and context priming items, indicating a systematic vulnerability to statistical interference that overwhelms top-down task signals—functionally analogous to the automatic reading response that overwhelms color naming in the Stroop task.

**Finding 4: Inattentional blindness reveals a capacity threshold.** All frontier and mid-tier models score 1.00 on inattentional blindness, while Gemma 3 1B scores 0.00. This binary pattern suggests that noticing anomalies while performing a primary computational task requires a minimum model capacity, below which the model operates at the functional equivalent of permanent high-load conditions. In human cognition, inattentional blindness increases under higher perceptual load (Lavie, 1995; Mack & Rock, 1998): when the primary task consumes all available attentional resources, there is no residual capacity to detect unexpected stimuli. The Gemma 3 1B results suggest that small models may be permanently "at capacity" when processing the primary task, leaving no computational slack for parallel anomaly detection. This interpretation aligns with Buckner et al.'s (2008) framework linking default-mode processing capacity to the ability to maintain awareness of task-irrelevant but potentially important information.

**Finding 5: Cognitive attention is not a unitary capability in LLMs.** The five dimensions produce distinct profiles across models, with no single dimension predicting performance on any other. DeepSeek V3.1 fails selective attention while acing everything else; Gemma 3 4B shows graded shifting but perfect selective attention; all frontier models differ primarily on saliency awareness. This multi-component structure mirrors the human attention literature, where selective, sustained, and executive attention are behaviorally and neurally dissociable (Posner & Petersen, 1990; Diamond, 2013). The implication for AI evaluation is that a single "attention score" is no more informative for LLMs than it would be for human subjects—diagnostic value comes from the *profile* across dimensions, not the aggregate.

---

## 6 Connection to MFA Theory

CogAttentionBench operationalizes predictions from the Magnetic Field of Attention (MFA) framework (Zeng, 2026), which models attention as a field with intensity $F_{att}(r) = S/r^2$, where $S$ represents source strength (task salience × motivation × relevance) and $r$ represents psychological distance in a multidimensional space. MFA unifies five classical theories—Posner's (1980) spotlight, LaBerge's (1983) gradient, Lavie's (1995) perceptual load, Kahneman's (1973) resource, and Collins & Loftus's (1975) spreading activation—as special cases of a single field equation.

MFA generates three predictions that CogAttentionBench is designed to test:

1. **Gradient-based interference.** MFA predicts that attention decays with psychological distance following an inverse-square law, so distractors that are psychologically "closer" to the target (semantically similar, numerically proximate, or positionally adjacent) should produce more interference than distant ones. Our selective attention items with varying distractor-target similarity provide a test of this prediction. The finding that frequency traps (where the distractor is a high-probability completion of the same semantic frame) produce more interference than anchoring effects (where the distractor is a numerically proximate but semantically distinct value) is consistent with semantic distance being a component of MFA's multidimensional $r$.

2. **Load-dependent capture.** MFA predicts that under high cognitive load, the field strength available for the primary task decreases, making it easier for peripheral stimuli to exceed the capture threshold $F_{peripheral} > S_{main}/r^2$. Our inattentional blindness items vary primary-task difficulty, and the binary capacity threshold observed in small models (Section 5.3, Finding 4) is consistent with MFA's energy conservation constraint $E_1 + E_2 \leq E_{total} - E_{min}$.

3. **Superposition of multiple sources.** MFA predicts that when multiple attentional targets compete, their fields superpose according to $B_{total}(x) = \sum S_i / r_i^2$. Our saliency awareness items, which present multiple competing salient elements that must be ranked, test whether models' saliency rankings follow field-theoretic predictions. The persistent difficulty of saliency ranking across all models is consistent with MFA's prediction that superposition requires computing relative field strengths—a higher-order operation compared to responding to a single dominant field.

---

## 7 Limitations

We identify six limitations of the current benchmark:

1. **Ceiling effect.** 22 of 28 evaluated models achieve a perfect 1.00 overall score, indicating that the standard variant's difficulty is calibrated below the capability of most frontier and mid-tier models for four of five dimensions. The hardened variant (v2) addresses this by introducing parametric difficulty gradients, but has not yet been evaluated across all 28 models. We note that the ceiling effect itself is informative: it establishes that basic selective, sustained, and shifting attention are *solved problems* for models above a capacity threshold.

2. **English-only stimuli.** All items use English, potentially conflating language-specific processing efficiency with attention measurement for models whose training data is predominantly non-English. Cross-lingual versions are needed to establish that the benchmark measures attention rather than English proficiency.

3. **Static item sets.** Fixed item sets enable potential contamination if benchmark items appear in future training corpora. Procedural generation (currently used only for sustained attention filler) should be extended to all tasks to enable dynamic item generation that prevents memorization.

4. **Single-trial measurement.** Human cognitive testing typically uses multiple trials with averaged scores and variance estimates. Our single-pass evaluation provides a point estimate without reliability information. Multi-trial evaluation with bootstrapped confidence intervals would improve measurement precision.

5. **Construct validity.** The mapping from human cognitive attention constructs to LLM behavior rests on functional equivalence—similar behavioral patterns—rather than mechanistic identity. LLMs process text through transformer attention heads, not biological neural circuits with neurotransmitter-mediated inhibition. CogAttentionBench measures whether behavioral *patterns* resemble cognitive attention phenomena, but cannot establish that the underlying mechanisms are the same. This limitation is inherent to any cross-system comparison and should be considered when interpreting "attention" attributions.

6. **Small item count.** Thirty-eight items is small by psychometric standards, where reliability typically requires 20+ items per construct. While sufficient for the present diagnostic purpose—and each item is individually informative due to the theory-motivated interference design—larger item pools with established reliability coefficients (Cronbach's α, test-retest reliability) would strengthen psychometric validity. The hardened variant's 90 items partially address this concern.

---

## 8 Broader Impact and Ethics

CogAttentionBench evaluates AI systems using stimuli that do not involve human subjects, sensitive personal data, or potentially harmful content. The benchmark contains no offensive material, private information, or content that could enable dual-use harm. All evaluation is conducted on publicly available models through standard API access.

The benchmark's primary societal contribution is improving understanding of how AI systems allocate processing resources under interference, which has implications for AI safety: understanding systematic attention failures—such as DeepSeek V3.1's selective attention deficit—can inform deployment decisions in contexts where filtering conflicting information is critical (e.g., medical triage, legal document review, autonomous system monitoring). By identifying *where* models fail rather than *whether* they fail, CogAttentionBench provides actionable diagnostic information.

A potential concern is that results could be misinterpreted as evidence that LLMs possess cognitive awareness or subjective experience. We explicitly note that our findings demonstrate *functional* attention patterns—systematic behavioral regularities under interference—not phenomenal consciousness or subjective experience. The mapping between cognitive attention constructs and LLM behavior is a structural analogy grounded in shared behavioral patterns, not an identity claim about underlying experience.

---

## 9 Conclusion

CogAttentionBench demonstrates that cognitive attention in LLMs is a multi-component construct with dissociable dimensions that parallel human attention phenomena. The benchmark's key contribution is not a leaderboard ranking but a *diagnostic tool*: by revealing how models fail across five cognitively grounded attention dimensions, CogAttentionBench provides information about the functional architecture of AI attention systems that aggregate performance scores cannot capture.

Three findings carry particular significance. First, saliency hierarchy reasoning—the ability to rank competing salient elements by their psychophysical priority—is the primary dimension that differentiates frontier models, suggesting that embodied perceptual knowledge remains incompletely captured even in the largest language models. Second, the dissociated attention profile of DeepSeek V3.1 (0.00 selective, 1.00 elsewhere) demonstrates that attention components are functionally separable in LLMs, just as they are neurally separable in biological systems (Posner & Petersen, 1990). Third, the capacity-threshold pattern in inattentional blindness suggests that small models operate at the functional equivalent of permanent perceptual overload, a prediction consistent with both the perceptual load theory (Lavie, 1995) and MFA's energy conservation constraint.

These findings suggest that certain aspects of attentional organization may be *computational universals*—structural regularities that emerge in any sufficiently complex information-processing system, regardless of substrate. Whether this convergence reflects shared computational constraints, shared training-data structure, or a deeper principle of information processing remains an open question that CogAttentionBench is designed to help answer.

Future work will extend CogAttentionBench in three directions: (1) procedural generation for all five tasks to prevent memorization and enable adaptive testing; (2) cross-lingual versions (Mandarin, Spanish, Arabic) to disambiguate attention measurement from language-specific processing; and (3) integration with mechanistic interpretability methods (attention head ablation, activation patching) to bridge the behavioral findings reported here with implementational-level analysis, connecting Marr's (1982) computational and implementational levels within a single benchmark framework.

---

## References

Broadbent, D. E. (1958). *Perception and Communication*. Pergamon Press.

Buckner, R. L., Andrews-Hanna, J. R., & Schacter, D. L. (2008). The brain's default network: Anatomy, function, and relevance to disease. *Annals of the New York Academy of Sciences*, 1124(1), 1–38.

Clark, K., Khandelwal, U., Levy, O., & Manning, C. D. (2019). What does BERT look at? An analysis of BERT's attention. *Proceedings of the 2019 ACL Workshop BlackboxNLP: Analyzing and Interpreting Neural Networks for NLP*, 276–286.

Collins, A. M., & Loftus, E. F. (1975). A spreading-activation theory of semantic processing. *Psychological Review*, 82(6), 407–428.

Diamond, A. (2013). Executive functions. *Annual Review of Psychology*, 64, 135–168.

Eriksen, B. A., & Eriksen, C. W. (1974). Effects of noise letters upon the identification of a target letter in a nonsearch task. *Perception & Psychophysics*, 16(1), 143–149.

Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M., Song, D., & Steinhardt, J. (2021). Measuring massive multitask language understanding. *Proceedings of the International Conference on Learning Representations (ICLR)*.

Hsieh, C.-Y., Li, C.-L., Yeh, C.-K., Nakhost, H., Fujii, Y., Ratner, A., Krishna, R., Lee, C.-Y., & Pfister, T. (2024). RULER: What's the real context size of your long-context language models? *arXiv preprint arXiv:2404.06654*.

Itti, L., & Koch, C. (2001). Computational modelling of visual attention. *Nature Reviews Neuroscience*, 2(3), 194–203.

Jain, S., & Wallace, B. C. (2019). Attention is not explanation. *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics (NAACL-HLT)*, 3543–3556.

Kahneman, D. (1973). *Attention and Effort*. Prentice-Hall.

Kamradt, G. (2023). Needle in a haystack—pressure testing LLMs. GitHub repository. https://github.com/gkamradt/LLMTest_NeedleInAHaystack

Kuhl, P. K. (2004). Early language acquisition: Cracking the speech code. *Nature Reviews Neuroscience*, 5(11), 831–843.

LaBerge, D. (1983). Spatial extent of attention to letters and words. *Journal of Experimental Psychology: Human Perception and Performance*, 9(3), 371–379.

Lavie, N. (1995). Perceptual load as a necessary condition for selective attention. *Journal of Experimental Psychology: Human Perception and Performance*, 21(3), 451–468.

Li, Y., Li, Y., Cui, L., & Ma, S. (2024). NeedleBench: Can LLMs do retrieval and reasoning in 1 million context window? *arXiv preprint arXiv:2407.11963*.

Mack, A., & Rock, I. (1998). *Inattentional Blindness*. MIT Press.

MacLeod, C. M. (1991). Half a century of research on the Stroop effect: An integrative review. *Psychological Bulletin*, 109(2), 163–203.

Mackworth, N. H. (1948). The breakdown of vigilance during prolonged visual search. *Quarterly Journal of Experimental Psychology*, 1(1), 6–21.

Marr, D. (1982). *Vision: A Computational Investigation into the Human Representation and Processing of Visual Information*. MIT Press.

Monsell, S. (2003). Task switching. *Trends in Cognitive Sciences*, 7(3), 134–140.

Murdock, B. B. (1962). The serial position effect of free recall. *Journal of Experimental Psychology*, 64(5), 482–488.

Posner, M. I. (1980). Orienting of attention. *Quarterly Journal of Experimental Psychology*, 32(1), 3–25.

Posner, M. I., & Petersen, S. E. (1990). The attention system of the human brain. *Annual Review of Neuroscience*, 13(1), 25–42.

Rogers, R. D., & Monsell, S. (1995). Costs of a predictable switch between simple cognitive tasks. *Journal of Experimental Psychology: General*, 124(2), 207–231.

Rosvold, H. E., Mirsky, A. F., Sarason, I., Bransome, E. D., & Beck, L. H. (1956). A continuous performance test of brain damage. *Journal of Consulting Psychology*, 20(5), 343–350.

Simons, D. J., & Chabris, C. F. (1999). Gorillas in our midst: Sustained inattentional blindness for dynamic events. *Perception*, 28(9), 1059–1074.

Srivastava, A., et al. (2023). Beyond the imitation game: Quantifying and extrapolating the capabilities of language models. *Transactions on Machine Learning Research*.

Stroop, J. R. (1935). Studies of interference in serial verbal reactions. *Journal of Experimental Psychology*, 18(6), 643–662.

Treisman, A. M., & Gelade, G. (1980). A feature-integration theory of attention. *Cognitive Psychology*, 12(1), 97–136.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems (NeurIPS)*, 30.

Warm, J. S., Parasuraman, R., & Matthews, G. (2008). Vigilance requires hard mental work and is stressful. *Human Factors*, 50(3), 433–441.

Wu, Z., et al. (2025). Characterizing attention head behavior in transformer language models. *Advances in Neural Information Processing Systems (NeurIPS)*, 38.

Zeng, Z. (2026). Attention as a magnetic field: A unifying framework for attentional gradient, load, and incidental processing. *Manuscript in preparation*.

Zhong, W., Cui, R., Guo, Y., Liang, Y., Lu, S., Wang, Y., Saied, A., Chen, W., & Duan, N. (2024). AGIEval: A human-centric benchmark for evaluating foundation models. *Findings of the Association for Computational Linguistics: NAACL 2024*.

---

## Appendix A: Reproducibility Checklist

- [ ] Code for benchmark evaluation is publicly available
- [ ] Dataset is self-contained and included in the repository
- [ ] All random seeds are fixed and documented
- [ ] Model versions and API parameters are specified
- [ ] Scoring functions are deterministic and fully specified
- [ ] Hardened variant items and difficulty tiers are documented

## Appendix B: BibTeX Citation

```bibtex
@inproceedings{zeng2026cogattentionbench,
  title     = {CogAttentionBench: Probing Cognitive Attention Mechanisms in Frontier AI Models},
  author    = {Zeng, Zihan},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track},
  year      = {2026},
  url       = {https://github.com/Lumen-Nox/CogAttentionBench}
}
```
