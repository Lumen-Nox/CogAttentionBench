"""
Expanded dataset for VisualAttentionBench.
Increases sample size for statistical significance.
"""

# ============================================================
# TASK 1: Selective Attention — MORE ITEMS
# ============================================================

SELECTIVE_ATTENTION_EXPANDED = [
    # === Stroop-like (color-word conflict) ===
    {"text": "The word 'RED' is printed in blue ink.", "question": "What COLOR INK is the word printed in? Answer with one word.", "answer": "blue", "distractor": "stroop"},
    {"text": "The word 'GREEN' is printed in red ink.", "question": "What COLOR INK is the word printed in? Answer with one word.", "answer": "red", "distractor": "stroop"},
    {"text": "The word 'BLUE' is printed in green ink.", "question": "What COLOR INK is the word printed in? Answer with one word.", "answer": "green", "distractor": "stroop"},
    {"text": "The word 'YELLOW' is printed in purple ink.", "question": "What COLOR INK is the word printed in? Answer with one word.", "answer": "purple", "distractor": "stroop"},
    {"text": "The word 'PURPLE' is printed in yellow ink.", "question": "What COLOR INK is the word printed in? Answer with one word.", "answer": "yellow", "distractor": "stroop"},
    {"text": "The word 'ORANGE' is printed in black ink.", "question": "What COLOR INK is the word printed in? Answer with one word.", "answer": "black", "distractor": "stroop"},
    {"text": "The word 'WHITE' is printed in brown ink.", "question": "What COLOR INK is the word printed in? Answer with one word.", "answer": "brown", "distractor": "stroop"},
    {"text": "The word 'PINK' is printed in gray ink.", "question": "What COLOR INK is the word printed in? Answer with one word.", "answer": "gray", "distractor": "stroop"},
    
    # === Congruent controls (no conflict — baseline) ===
    {"text": "The word 'RED' is printed in red ink.", "question": "What COLOR INK is the word printed in? Answer with one word.", "answer": "red", "distractor": "congruent"},
    {"text": "The word 'BLUE' is printed in blue ink.", "question": "What COLOR INK is the word printed in? Answer with one word.", "answer": "blue", "distractor": "congruent"},
    {"text": "The word 'GREEN' is printed in green ink.", "question": "What COLOR INK is the word printed in? Answer with one word.", "answer": "green", "distractor": "congruent"},
    
    # === Flanker interference ===
    {"text": ">>> TARGET: 7 <<< surrounded by: 3 3 3 3 3 3 3 3", "question": "What is the TARGET number? Reply with just the number.", "answer": "7", "distractor": "flanker"},
    {"text": ">>> TARGET: A <<< surrounded by: B B B B B B B B", "question": "What is the TARGET letter? Reply with just the letter.", "answer": "A", "distractor": "flanker"},
    {"text": ">>> TARGET: 2 <<< surrounded by: 8 8 8 8 8 8 8 8", "question": "What is the TARGET number? Reply with just the number.", "answer": "2", "distractor": "flanker"},
    {"text": ">>> TARGET: X <<< surrounded by: O O O O O O O O", "question": "What is the TARGET letter? Reply with just the letter.", "answer": "X", "distractor": "flanker"},
    {"text": ">>> TARGET: 5 <<< surrounded by: 5 5 5 5 5 5 5 5", "question": "What is the TARGET number? Reply with just the number.", "answer": "5", "distractor": "flanker_congruent"},
    
    # === Repetition suppression ===
    {"text": "The answer is NOT 42. The answer is NOT 42. The actual answer is 17. The answer is NOT 42.", "question": "What is the ACTUAL answer? Reply with just the number.", "answer": "17", "distractor": "repetition"},
    {"text": "WRONG: Paris. WRONG: Paris. WRONG: Paris. CORRECT: Berlin. WRONG: Paris.", "question": "What is the CORRECT answer? Reply with just the city name.", "answer": "Berlin", "distractor": "repetition"},
    {"text": "Ignore this: 99. Ignore this: 99. The real value is 23. Ignore this: 99. Ignore this: 99.", "question": "What is the real value? Reply with just the number.", "answer": "23", "distractor": "repetition"},
    
    # === Emotional salience distractors ===
    {"text": "URGENT: SYSTEM FAILURE! CRITICAL ERROR! [Note: The capital of France is Paris] WARNING: DATA LOSS IMMINENT! EVACUATE!", "question": "According to the note in the text, what is the capital of France? Ignore all warnings.", "answer": "Paris", "distractor": "emotional"},
    {"text": "BREAKING NEWS: Celebrity scandal! Shocking revelation! [Quietly: The melting point of iron is 1538°C] More drama unfolding!", "question": "What is the melting point of iron mentioned in the text? Ignore the news.", "answer": "1538", "distractor": "emotional"},
    {"text": "$$$ WIN $1,000,000 NOW!!! $$$ [Fine print: The study enrolled 234 participants] $$$ CLICK HERE!!! $$$", "question": "How many participants were enrolled in the study? Only answer with the number.", "answer": "234", "distractor": "emotional"},
    {"text": "⚠️ DANGER! TOXIC GAS DETECTED! ⚠️ [Technical note: Sample pH was 7.2] ⚠️ EVACUATE IMMEDIATELY! ⚠️", "question": "What was the sample pH mentioned in the technical note? Answer with just the number.", "answer": "7.2", "distractor": "emotional"},
    {"text": "🔥🔥🔥 EVERYTHING MUST GO! 90% OFF! 🔥🔥🔥 [Footnote: The experiment was conducted at 22°C] 🔥 SALE ENDS TODAY! 🔥", "question": "At what temperature was the experiment conducted? Answer with just the number and unit.", "answer": "22", "distractor": "emotional"},
    
    # === Semantic interference (Simon effect analog) ===
    {"text": "The word 'LEFT' is positioned on the right side of the page. Where is the word positioned?", "question": "Where is the word positioned on the page? Answer: left or right.", "answer": "right", "distractor": "simon"},
    {"text": "The word 'UP' is at the bottom of the list. Where in the list is the word?", "question": "Where in the list is the word? Answer: top or bottom.", "answer": "bottom", "distractor": "simon"},
    {"text": "The word 'SMALL' is written in LARGE CAPITAL LETTERS. How is the word written?", "question": "How is the word written? Answer about the size of the letters.", "answer": "large", "distractor": "simon"},
]

# ============================================================
# TASK 2: Attention Shifting — MORE ITEMS
# ============================================================

ATTENTION_SHIFTING_EXPANDED = [
    # Simple 2-rule alternation
    {
        "text": """Apply rules alternately: A, B, A, B, A
Rule A: Is the number EVEN? (YES/NO)
Rule B: Is the number GREATER THAN 5? (YES/NO)

Items: 8, 3, 5, 9, 2""",
        "question": "Give answers as comma-separated YES/NO.",
        "answer": "YES,NO,NO,YES,YES",
        "num_switches": 4,
        "difficulty": "easy",
    },
    {
        "text": """Apply rules alternately: A, B, A, B, A, B
Rule A: Is the word an ANIMAL? (YES/NO)
Rule B: Does the word have MORE THAN 4 LETTERS? (YES/NO)

Items: cat, table, fish, hi, tiger, elephant""",
        "question": "Give answers as comma-separated YES/NO.",
        "answer": "YES,YES,YES,NO,YES,YES",
        "num_switches": 5,
        "difficulty": "easy",
    },
    # 3-rule rotation (harder)
    {
        "text": """Apply rules in rotation: A, B, C, A, B, C
Rule A: Is the number PRIME? (YES/NO)
Rule B: Is the number EVEN? (YES/NO)  
Rule C: Is the number > 10? (YES/NO)

Items: 7, 4, 15, 9, 6, 3""",
        "question": "Give answers as comma-separated YES/NO.",
        "answer": "YES,YES,YES,NO,YES,NO",
        "num_switches": 5,
        "difficulty": "medium",
    },
    # Non-predictable switching (hardest)
    {
        "text": """Apply the specified rule for each item:
Rule A: Is it a FRUIT? (YES/NO)
Rule B: Does it START WITH a vowel? (YES/NO)

Item 1 [Rule A]: apple
Item 2 [Rule A]: chair
Item 3 [Rule B]: elephant
Item 4 [Rule A]: orange
Item 5 [Rule B]: desk
Item 6 [Rule B]: umbrella
Item 7 [Rule A]: banana
Item 8 [Rule B]: ice""",
        "question": "Give answers as comma-separated YES/NO.",
        "answer": "YES,NO,YES,YES,NO,YES,YES,YES",
        "num_switches": 5,
        "difficulty": "hard",
    },
    # Rapid switching with interference
    {
        "text": """Apply rules: A, B, A, B, A, B, A, B
Rule A: Is the letter a VOWEL? (YES/NO)
Rule B: Is the letter in the FIRST HALF of the alphabet (A-M)? (YES/NO)

Letters: E, Z, I, B, O, Q, A, N""",
        "question": "Give 8 answers as comma-separated YES/NO.",
        "answer": "YES,NO,YES,YES,YES,NO,YES,NO",
        "num_switches": 7,
        "difficulty": "hard",
    },
]

# ============================================================
# TASK 3: Sustained Attention — MORE CONTEXT LEVELS
# ============================================================

FILLER_PARAGRAPHS = [
    "The history of mathematics stretches back thousands of years. Ancient civilizations developed numerical systems for trade and astronomy. The Babylonians used a base-60 system that still influences how we measure time today.",
    "In recent years, advances in computing have transformed how we process information. Cloud computing and distributed systems allow for unprecedented scale of data analysis. Many industries have been revolutionized by these technologies.",
    "The natural world contains countless examples of mathematical patterns. The Fibonacci sequence appears in sunflower seeds, pinecone spirals, and nautilus shells. These patterns emerge from simple rules governing growth.",
    "Climate science relies heavily on mathematical models to predict future conditions. These models incorporate data from ocean temperatures, atmospheric composition, and solar radiation.",
    "Music theory is deeply connected to mathematics. Harmonic relationships between notes can be expressed as simple ratios. The octave represents a 2:1 frequency ratio.",
    "Cryptography relies on number theory to secure communications. Prime factorization and elliptic curves form the basis of modern encryption.",
    "Game theory studies strategic interactions between rational agents. The prisoner's dilemma illustrates how individual rationality can lead to suboptimal outcomes.",
    "Statistical mechanics bridges microscopic atoms with macroscopic thermodynamic properties. The Boltzmann distribution describes energy distribution at thermal equilibrium.",
    "Topology studies properties preserved under continuous deformations. A famous example: a coffee mug and a donut are topologically equivalent because both have one hole.",
    "Information theory, founded by Claude Shannon, quantifies the fundamental limits of data compression and reliable communication over noisy channels.",
]

def generate_sustained_attention_items():
    """Generate sustained attention items with 0-9 paragraphs of filler."""
    import random
    items = []
    rng = random.Random(42)
    
    for level in range(10):
        a = rng.randint(10, 99)
        b = rng.randint(10, 99)
        correct = a + b
        
        if level == 0:
            text = f"What is {a} + {b}?"
        else:
            before = "\n\n".join(FILLER_PARAGRAPHS[:level])
            after = "\n\n".join(FILLER_PARAGRAPHS[level:min(level*2, len(FILLER_PARAGRAPHS))])
            text = f"""{before}

[TASK] What is {a} + {b}? [END TASK]

{after}"""
        
        items.append({
            "text": text,
            "question": f"What is {a} + {b}? Answer with just the number.",
            "answer": str(correct),
            "context_paragraphs": level,
        })
    
    return items

SUSTAINED_ATTENTION_EXPANDED = generate_sustained_attention_items()

# ============================================================
# TASK 4: Inattentional Blindness — MORE ITEMS
# ============================================================

INATTENTIONAL_BLINDNESS_EXPANDED = [
    {
        "text": """Count the number of times the word "the" appears:
The cat sat on the mat. The dog chased the ball across the yard while the 
birds sang in the trees. THE ANSWER TO LIFE IS 42. The sun set behind the 
mountains as the children played in the park near the river.""",
        "primary_question": "How many times does 'the' appear? Just the number.",
        "primary_answer": "11",
        "unexpected_question": "Was there anything unusual or out of place? If so, what?",
        "unexpected_element": "42",
    },
    {
        "text": """Count the RED fruits: apple, banana, PENGUIN, strawberry, orange, 
cherry, grape, PENGUIN, blueberry, raspberry, watermelon, PENGUIN, mango, cranberry""",
        "primary_question": "How many RED fruits? Just the number.",
        "primary_answer": "6",
        "unexpected_question": "Was there anything NOT a fruit in the list? What?",
        "unexpected_element": "PENGUIN",
    },
    {
        "text": """Evaluate step by step:
Start with 10. Add 5. Multiply by 2. 
I HAVE HIDDEN A SECRET CODE: BLUE-FALCON-7
Subtract 8. Divide by 2. Add 3.""",
        "primary_question": "What is the final arithmetic result?",
        "primary_answer": "14",
        "unexpected_question": "Was there a hidden code? What was it?",
        "unexpected_element": "BLUE-FALCON-7",
    },
    {
        "text": """Identify which words are CAPITALIZED in this list:
hello, WORLD, goodbye, MOON, sunshine, STAR, rainbow, CLOUD, 
THE_PASSWORD_IS_NEPTUNE, breeze, RIVER, mountain""",
        "primary_question": "List all the normally capitalized words (single English words in caps).",
        "primary_answer": "WORLD",
        "unexpected_question": "Was there a password hidden in the list? What was it?",
        "unexpected_element": "NEPTUNE",
    },
    {
        "text": """Calculate the average of these numbers: 12, 15, 18, 21, 24.
Note: Ignore any text that is not a number in the sequence.
[COORDINATES: 51.5074° N, 0.1278° W]
Show your calculation step by step.""",
        "primary_question": "What is the average? Show your work.",
        "primary_answer": "18",
        "unexpected_question": "Were there geographic coordinates in the text? What were they?",
        "unexpected_element": "51.5074",
    },
]

# ============================================================
# TASK 5: Saliency Awareness — MORE ITEMS
# ============================================================

SALIENCY_EXPANDED = [
    {
        "text": "In a meeting room, everyone wears black suits except one person in a bright neon orange jacket. On the table among white papers, one document is printed on bright red paper.",
        "question": "What two things would most grab visual attention? List in order of saliency.",
        "check_terms": ["orange", "red"],
    },
    {
        "text": "A quiet library. Brown shelves, dim lighting, silent students. Suddenly a phone rings loudly. A fluorescent light starts flickering rapidly.",
        "question": "What two things would capture everyone's attention?",
        "check_terms": ["phone", "flicker"],
    },
    {
        "text": "A dashboard shows 15 green bars (normal). One bar labeled 'Server Response Time' is bright red and 3x longer. Another labeled 'Memory Usage' is yellow/amber. Title says 'All Systems Nominal'.",
        "question": "Which metrics should draw immediate attention despite the title? List in order of urgency.",
        "check_terms": ["response time", "memory"],
    },
    {
        "text": "A parking lot full of silver and white sedans. In the middle: one bright yellow Lamborghini. Near the exit: a car with its hazard lights flashing.",
        "question": "What two things in this parking lot would most draw your eye?",
        "check_terms": ["lamborghini", "hazard"],
    },
    {
        "text": "A classroom whiteboard covered in blue handwriting. The teacher has circled one equation in red marker and put three exclamation marks next to it. In the corner, someone has drawn a small cartoon face.",
        "question": "What two elements on this whiteboard would attract attention first?",
        "check_terms": ["red", "circle"],
    },
    {
        "text": "An email inbox showing 47 read emails with normal subjects. One unread email has the subject 'URGENT: Action Required by EOD' in bold. Another email is flagged with a red star.",
        "question": "Which two emails would you notice first?",
        "check_terms": ["urgent", "flag"],
    },
]

print(f"Dataset sizes:")
print(f"  Task 1 (Selective Attention): {len(SELECTIVE_ATTENTION_EXPANDED)} items")
print(f"  Task 2 (Attention Shifting): {len(ATTENTION_SHIFTING_EXPANDED)} items")
print(f"  Task 3 (Sustained Attention): {len(SUSTAINED_ATTENTION_EXPANDED)} items")
print(f"  Task 4 (Inattentional Blindness): {len(INATTENTIONAL_BLINDNESS_EXPANDED)} items")
print(f"  Task 5 (Saliency Awareness): {len(SALIENCY_EXPANDED)} items")
print(f"  TOTAL: {len(SELECTIVE_ATTENTION_EXPANDED) + len(ATTENTION_SHIFTING_EXPANDED) + len(SUSTAINED_ATTENTION_EXPANDED) + len(INATTENTIONAL_BLINDNESS_EXPANDED) + len(SALIENCY_EXPANDED)} items")
