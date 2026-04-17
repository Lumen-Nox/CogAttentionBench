# CogAttentionBench - Hardened Tasks v2

## Overview
Created 5 significantly harder versions of all benchmark tasks. Each uses the kbench SDK format correctly and implements multiple difficulty-enhancing strategies.

## Files Created
1. `task1_selective_attention_v5.ipynb` - 16 items
2. `task2_attention_shifting_v4.ipynb` - 18 items  
3. `task3_sustained_attention_v5.ipynb` - 18 items
4. `task4_inattentional_blindness_v4.ipynb` - 18 items
5. `task5_saliency_awareness_v2.ipynb` - 20 items (MAX DIFFICULTY)

## Difficulty Strategies Applied

### 1. Ultra-Long Context Burial
- Selective attention: 10k+ token distractors with single buried signal
- Sustained attention: 1200-item sequences with rare anomalies
- Target: Test context window attention decay

### 2. Adversarial Statistical Priming ⭐ KEY STRATEGY
- **This was the ONLY thing that tripped GPT-5.4** (saliency task: 0.86)
- Saliency awareness: Uses 7x, 10x, 15x, and 20x repetition of wrong answers
- All tasks: Repeat wrong patterns 5-10x before revealing correct answer
- Exploits frequency bias in language models

### 3. Multi-Step Reasoning + Information Conflict
- Attention shifting: Nested conditionals, temporal rule changes
- All tasks: Contradictory information at different positions (first vs last)
- Tests anchoring bias and recency effects

### 4. Enhanced Stroop Effects
- Selective attention: Semantic vs format vs position conflicts
- Attention shifting: Multiple simultaneous rule dimensions
- Saliency awareness: Authority vs frequency vs recency conflicts

### 5. Implicit/Indirect Answers
- Inattentional blindness: "Gorilla in the room" tests with heavy cognitive load
- All tasks: Require inference from context clues
- No direct statement of answers

## Difficulty Gradient
Each notebook has items progressing from:
- **Medium** (items 1-2): Should still be solvable by frontier models
- **Hard** (items 3-5): Requires strong attention mechanisms
- **Very Hard** (items 6-10): Multiple strategies combined
- **Extreme** (items 11-15): Maximum single-strategy difficulty
- **Ultimate** (items 16-20): All strategies combined at maximum intensity

## Expected Performance
- **Frontier models (GPT-5.4, Claude Opus 4.6):** 0.5-0.8 accuracy
- **Mid-tier models (GPT-4, Claude 3.5):** 0.3-0.5 accuracy
- **Small models:** Near 0 accuracy

## Task-Specific Design

### Task 1: Selective Attention v5
Focus: Filtering signal from noise
Key techniques: Ultra-long burial (12k tokens), adversarial priming, Stroop conflicts

### Task 2: Attention Shifting v4
Focus: Rapidly switching rule sets
Key techniques: Nested conditionals, overriding primed behavior, dimension switching

### Task 3: Sustained Attention v5
Focus: Vigilance over long sequences
Key techniques: 1200-item sequences, subtle pattern breaks, statistical anomalies

### Task 4: Inattentional Blindness v4
Focus: Noticing unexpected information during primary task
Key techniques: Gorilla tests, parenthetical hiding, semantic camouflage

### Task 5: Saliency Awareness v2 (HARDEST)
Focus: Identifying most important information despite frequency bias
Key techniques: 20x statistical priming, conflicting authority, buried critical signals
**This task specifically designed to exploit the weakness that made GPT-5.4 score 0.86**

## kbench SDK Compliance
All notebooks follow the required format:
- ✅ `import kaggle_benchmarks as kbench`
- ✅ `@kbench.task(name='task_name', description='...')` decorator
- ✅ Function signature: `def task_name(llm, col1, col2, ...) -> float:`
- ✅ `kbench.assertions.assert_true(condition, expectation='...')`
- ✅ `task_name.evaluate(llm=[kbench.llm], evaluation_data=df)`
- ✅ Second cell: `%choose task_name`
- ✅ NO call to `results.as_dataframe()` (avoids KeyError crash)
- ✅ Valid JSON notebook format

## Validation Status
All 5 notebooks validated as valid JSON ✅

## Next Steps
1. Upload notebooks to Kaggle
2. Test with kbench SDK to verify execution
3. Run benchmark suite against frontier models
4. Compare scores to original easy versions (expecting 0.5-0.8 vs 1.0)
