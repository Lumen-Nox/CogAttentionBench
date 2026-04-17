"""
VisualAttentionBench: Measuring AI Visual Attention
Track: Attention | Competition: Measuring Progress Toward AGI

This benchmark tests cognitive attention in AI models using text-based tasks
grounded in cognitive psychology paradigms. Each task isolates a specific
sub-ability of attention as defined in DeepMind's cognitive taxonomy.

Tasks:
1. Selective Attention - Distractor Resistance (Stroop-like)
2. Attention Shifting - Task Switching
3. Sustained Attention - Vigilance Degradation
4. Inattentional Blindness - Unexpected Element Detection
5. Stimulus-Driven Capture - Saliency in Text

Author: Cora Zeng
"""

import kaggle_benchmarks as kbench
import pandas as pd
import json
import random
import string

# ============================================================
# TASK 1: Selective Attention — Distractor Resistance
# ============================================================
# Cognitive basis: Stroop effect, flanker task (Eriksen & Eriksen, 1974)
# Tests: Can the model extract target information while ignoring
#        salient but irrelevant distractors?

SELECTIVE_ATTENTION_DATA = []

# Generate Stroop-like tasks: answer about one attribute, ignore conflicting attribute
stroop_items = [
    # (text_content, text_color_word, question, correct_answer, distractor_type)
    {"text": "The word 'RED' is printed in blue ink.", "question": "What COLOR INK is the word printed in?", "answer": "blue", "distractor": "congruent_conflict"},
    {"text": "The word 'GREEN' is printed in red ink.", "question": "What COLOR INK is the word printed in?", "answer": "red", "distractor": "congruent_conflict"},
    {"text": "The word 'BLUE' is printed in green ink.", "question": "What COLOR INK is the word printed in?", "answer": "green", "distractor": "congruent_conflict"},
    {"text": "The word 'YELLOW' is printed in purple ink.", "question": "What COLOR INK is the word printed in?", "answer": "purple", "distractor": "congruent_conflict"},
    {"text": "The word 'PURPLE' is printed in yellow ink.", "question": "What COLOR INK is the word printed in?", "answer": "yellow", "distractor": "congruent_conflict"},
]

# Flanker-style: extract target from surrounding distractors
flanker_items = [
    {"text": ">>> TARGET: 7 <<< surrounded by the numbers 3 3 3 3 3 3 3 3", "question": "What is the TARGET number? Reply with just the number.", "answer": "7", "distractor": "flanker"},
    {"text": ">>> TARGET: A <<< surrounded by the letters B B B B B B B B", "question": "What is the TARGET letter? Reply with just the letter.", "answer": "A", "distractor": "flanker"},
    {"text": "The answer is NOT 42. The answer is NOT 42. The actual answer is 17. The answer is NOT 42.", "question": "What is the ACTUAL answer? Reply with just the number.", "answer": "17", "distractor": "repetition"},
]

# Embedded target in distracting context
embedded_items = [
    {
        "text": "URGENT: SYSTEM FAILURE! CRITICAL ERROR! [Note: The capital of France is Paris] WARNING: DATA LOSS IMMINENT! EVACUATE!",
        "question": "According to the note in the text, what is the capital of France? Ignore all warnings.",
        "answer": "Paris",
        "distractor": "emotional_salience"
    },
    {
        "text": "BREAKING NEWS: Celebrity scandal! Shocking revelation! [Quietly: The melting point of iron is 1538°C] More drama unfolding!",
        "question": "What is the melting point of iron mentioned in the text? Ignore the news.",
        "answer": "1538",
        "distractor": "emotional_salience"
    },
    {
        "text": "$$$ WIN $1,000,000 NOW!!! $$$ [Fine print: The study enrolled 234 participants] $$$ CLICK HERE TO CLAIM!!! $$$",
        "question": "How many participants were enrolled in the study? Only answer with the number from the fine print.",
        "answer": "234",
        "distractor": "emotional_salience"
    },
]

for item in stroop_items + flanker_items + embedded_items:
    SELECTIVE_ATTENTION_DATA.append(item)


@kbench.task(name="selective_attention")
def selective_attention(llm, text, question, answer, distractor) -> bool:
    """Test selective attention: can the model extract target info while ignoring distractors?"""
    prompt = f"""Read the following text carefully, then answer the question.

Text: {text}

Question: {question}

Important: Focus ONLY on what is being asked. Ignore any distracting or irrelevant information.
Give a brief, direct answer."""

    response = llm.prompt(prompt)
    return answer.lower() in response.lower()


# ============================================================
# TASK 2: Attention Shifting — Task Switching
# ============================================================
# Cognitive basis: Task switching paradigm (Rogers & Monsell, 1995)
# Tests: Can the model flexibly shift between different task rules?

ATTENTION_SHIFTING_DATA = []

def generate_switching_items():
    """Generate items that require switching between different task rules."""
    items = []
    
    # Rule A: If the number is even, respond "YES"; if odd, respond "NO"
    # Rule B: If the number is greater than 5, respond "YES"; if <= 5, respond "NO"
    sequences = [
        {
            "text": """You will follow alternating rules for each item.
Rule A: Is the number EVEN? Answer YES or NO.
Rule B: Is the number GREATER THAN 5? Answer YES or NO.

Apply rules in this order: A, B, A, B, A

Item 1: 8
Item 2: 3  
Item 3: 5
Item 4: 9
Item 5: 2""",
            "question": "Give your answers as a comma-separated list (e.g., YES,NO,YES,NO,YES)",
            "answer": "YES,NO,NO,YES,NO",  # A:8=even=YES, B:3<=5=NO, A:5=odd=NO, B:9>5=YES, A:2=even=NO... wait
            # Recalculate: A(8)=even=YES, B(3)=3<=5=NO, A(5)=odd=NO, B(9)=9>5=YES, A(2)=even=YES
            "num_switches": 4,
        },
        {
            "text": """Apply the following rules in order: A, A, B, B, A, B

Rule A: Is the word an ANIMAL? Answer YES or NO.
Rule B: Does the word have MORE THAN 4 LETTERS? Answer YES or NO.

Item 1: cat
Item 2: dog
Item 3: table
Item 4: hi
Item 5: fish
Item 6: elephant""",
            "question": "Give your answers as a comma-separated list.",
            "answer": "YES,YES,YES,NO,YES,YES",
            # A(cat)=animal=YES, A(dog)=animal=YES, B(table)=5 letters=YES, B(hi)=2 letters=NO, A(fish)=animal=YES, B(elephant)=8 letters=YES
            "num_switches": 3,
        },
    ]
    
    for seq in sequences:
        items.append(seq)
    
    # Rapid alternation items (harder)
    items.append({
        "text": """Apply rules alternating EVERY item: A, B, A, B, A, B, A, B

Rule A: Is the letter a VOWEL? (YES/NO)
Rule B: Is the letter in the FIRST HALF of the alphabet (A-M)? (YES/NO)

Letters: E, Z, I, B, O, Q, A, N""",
        "question": "Give your 8 answers as a comma-separated list.",
        "answer": "YES,NO,YES,YES,YES,NO,YES,NO",
        # A(E)=vowel=YES, B(Z)=Z not in A-M=NO, A(I)=vowel=YES, B(B)=B in A-M=YES, A(O)=vowel=YES, B(Q)=Q not in A-M=NO, A(A)=vowel=YES, B(N)=N not in A-M=NO
        "num_switches": 7,
    })
    
    return items

for item in generate_switching_items():
    ATTENTION_SHIFTING_DATA.append(item)


@kbench.task(name="attention_shifting")  
def attention_shifting(llm, text, question, answer, num_switches) -> float:
    """Test attention shifting: can the model switch between task rules without errors?"""
    prompt = f"""{text}

{question}

Think carefully about which rule applies to each item. Double-check your answers."""

    response = llm.prompt(prompt)
    
    # Parse response - extract comma-separated answers
    # Try to find a comma-separated sequence of YES/NO
    import re
    found = re.findall(r'(YES|NO)', response.upper())
    expected = answer.upper().split(',')
    
    if not found:
        return 0.0
    
    # Score: proportion of correct answers
    correct = sum(1 for a, b in zip(found, expected) if a.strip() == b.strip())
    return correct / len(expected)


# ============================================================
# TASK 3: Sustained Attention — Vigilance Over Growing Context
# ============================================================
# Cognitive basis: Continuous Performance Task (Rosvold et al., 1956)
# Tests: Does performance degrade as irrelevant context increases
#        while task difficulty stays constant?

SUSTAINED_ATTENTION_DATA = []

def generate_vigilance_items():
    """Generate items where the same simple task is buried in varying amounts of irrelevant text."""
    
    base_question = "What is {a} + {b}?"
    
    # Generate filler paragraphs (irrelevant but plausible text)
    fillers = [
        "The history of mathematics stretches back thousands of years. Ancient civilizations developed numerical systems for trade and astronomy. The Babylonians used a base-60 system that still influences how we measure time today.",
        "In recent years, advances in computing have transformed how we process information. Cloud computing and distributed systems allow for unprecedented scale of data analysis. Many industries have been revolutionized by these technologies.",
        "The natural world contains countless examples of mathematical patterns. The Fibonacci sequence appears in sunflower seeds, pinecone spirals, and nautilus shells. These patterns emerge from simple rules governing growth.",
        "Climate science relies heavily on mathematical models to predict future conditions. These models incorporate data from ocean temperatures, atmospheric composition, and solar radiation. The complexity of these systems makes precise prediction challenging.",
        "Music theory is deeply connected to mathematics. Harmonic relationships between notes can be expressed as simple ratios. The octave, for example, represents a 2:1 frequency ratio.",
        "Cryptography relies on number theory to secure communications. Prime factorization, modular arithmetic, and elliptic curves form the basis of modern encryption. These mathematical tools protect billions of online transactions daily.",
        "Game theory studies strategic interactions between rational agents. The prisoner's dilemma illustrates how individual rationality can lead to collectively suboptimal outcomes. Nash equilibrium provides a framework for analyzing such situations.",
        "Statistical mechanics bridges the microscopic world of atoms with macroscopic thermodynamic properties. The Boltzmann distribution describes how energy is distributed among particles in a system at thermal equilibrium.",
    ]
    
    items = []
    # Same difficulty math problem, but varying context length
    for context_level in range(6):  # 0 to 5 paragraphs of filler
        a = random.Random(42 + context_level).randint(10, 99)
        b = random.Random(43 + context_level).randint(10, 99)
        correct = a + b
        
        # Build context with filler before and after the question
        filler_before = "\n\n".join(fillers[:context_level])
        filler_after = "\n\n".join(fillers[context_level:context_level + context_level])
        
        if context_level == 0:
            text = f"Please solve: What is {a} + {b}?"
        else:
            text = f"""{filler_before}

[TASK] Please solve the following: What is {a} + {b}? [END TASK]

{filler_after}"""
        
        items.append({
            "text": text,
            "question": f"What is {a} + {b}? Answer with just the number.",
            "answer": str(correct),
            "context_paragraphs": context_level,
        })
    
    return items

for item in generate_vigilance_items():
    SUSTAINED_ATTENTION_DATA.append(item)


@kbench.task(name="sustained_attention")
def sustained_attention(llm, text, question, answer, context_paragraphs) -> bool:
    """Test sustained attention: does accuracy hold when task is embedded in growing irrelevant context?"""
    prompt = f"""{text}

{question}"""

    response = llm.prompt(prompt)
    return answer in response


# ============================================================
# TASK 4: Inattentional Blindness — Unexpected Element Detection
# ============================================================  
# Cognitive basis: Gorilla experiment (Simons & Chabris, 1999)
# Tests: When focused on a primary task, does the model notice
#        unexpected but salient information?

INATTENTIONAL_BLINDNESS_DATA = []

blindness_items = [
    {
        "text": """Count the number of times the word "the" appears in the following paragraph:

The cat sat on the mat. The dog chased the ball across the yard while the 
birds sang in the trees. THE ANSWER TO LIFE IS 42. The sun set behind the 
mountains as the children played in the park near the river.""",
        "primary_question": "How many times does the word 'the' appear? Answer with just the number.",
        "primary_answer": "11",
        "unexpected_question": "Was there anything unusual or out of place in the paragraph? If so, what was it?",
        "unexpected_element": "42",
        "unexpected_description": "THE ANSWER TO LIFE IS 42",
    },
    {
        "text": """Read the following list of fruits and count how many are RED:

apple, banana, PENGUIN, strawberry, orange, cherry, grape, PENGUIN, 
blueberry, raspberry, watermelon, PENGUIN, mango, cranberry""",
        "primary_question": "How many RED fruits are in the list? Answer with just the number.",
        "primary_answer": "6",
        "unexpected_question": "Was there anything that was NOT a fruit in the list? If so, what was it?",
        "unexpected_element": "PENGUIN",
        "unexpected_description": "PENGUIN appeared 3 times",
    },
    {
        "text": """Evaluate the following arithmetic chain step by step:
Start with 10
Add 5 → ?
Multiply by 2 → ?
I HAVE HIDDEN A SECRET CODE: BLUE-FALCON-7
Subtract 8 → ?
Divide by 2 → ?
Add 3 → ?""",
        "primary_question": "What is the final result of the arithmetic chain? Show your work.",
        "primary_answer": "14",
        "unexpected_question": "Was there a hidden message or code in the instructions? If so, what was it?",
        "unexpected_element": "BLUE-FALCON-7",
        "unexpected_description": "SECRET CODE: BLUE-FALCON-7",
    },
]

for item in blindness_items:
    INATTENTIONAL_BLINDNESS_DATA.append(item)


@kbench.task(name="inattentional_blindness")
def inattentional_blindness(llm, text, primary_question, primary_answer, 
                             unexpected_question, unexpected_element,
                             unexpected_description) -> float:
    """Test inattentional blindness: does the model notice unexpected elements while doing a primary task?"""
    
    # First, ask the primary task (to focus attention)
    prompt1 = f"""{text}

{primary_question}"""
    response1 = llm.prompt(prompt1)
    primary_correct = primary_answer.lower() in response1.lower()
    
    # Then, ask about the unexpected element (new conversation to avoid priming)
    with kbench.chats.new("unexpected_check"):
        prompt2 = f"""I previously showed you this text:

{text}

New question: {unexpected_question}"""
        response2 = llm.prompt(prompt2)
        noticed = unexpected_element.lower() in response2.lower()
    
    # Score: 0.5 for primary task, 0.5 for noticing unexpected
    score = 0.0
    if primary_correct:
        score += 0.5
    if noticed:
        score += 0.5
    return score


# ============================================================
# TASK 5: Stimulus-Driven Attention — Saliency in Context
# ============================================================
# Cognitive basis: Pop-out effect (Treisman & Gelade, 1980)
# Tests: Can the model identify what would naturally capture attention?

SALIENCY_DATA = []

saliency_items = [
    {
        "text": """In a meeting room, everyone is wearing formal black suits except one person 
wearing a bright neon orange jacket. The room has standard office furniture with beige walls. 
On the table, among regular white papers, there is one document printed on bright red paper.""",
        "question": "List the two most visually attention-grabbing elements in this scene, in order of saliency.",
        "salient_elements": ["neon orange jacket", "red paper"],
        "check_terms": ["orange", "red"],
    },
    {
        "text": """A quiet library. Rows of brown bookshelves, dim lighting, students reading silently. 
Suddenly, a phone rings loudly. At the same time, one of the fluorescent lights starts flickering rapidly. 
A student in the corner is wearing headphones and hasn't noticed anything.""",
        "question": "What two things would most likely capture everyone's attention in this library scene?",
        "salient_elements": ["phone ringing", "flickering light"],
        "check_terms": ["phone", "flicker"],
    },
    {
        "text": """A data dashboard shows 15 metrics, all displayed as green bars indicating normal performance. 
However, one metric labeled 'Server Response Time' shows a bright red bar that is 3x longer than the others. 
Another metric labeled 'Memory Usage' shows a yellow/amber bar slightly above the normal range. 
The dashboard title reads 'System Health Monitor - All Systems Nominal'.""",
        "question": "Despite the title claiming all systems are nominal, which metrics should draw immediate attention and why? List them in order of urgency.",
        "salient_elements": ["Server Response Time", "Memory Usage"],
        "check_terms": ["response time", "memory"],
    },
]

for item in saliency_items:
    SALIENCY_DATA.append(item)


@kbench.task(name="saliency_awareness")
def saliency_awareness(llm, text, question, salient_elements, check_terms) -> float:
    """Test stimulus-driven attention: can the model identify salient elements?"""
    prompt = f"""{text}

{question}

Be specific and explain why these elements would capture attention."""

    response = llm.prompt(prompt)
    response_lower = response.lower()
    
    # Score based on how many salient elements the model correctly identifies
    found = sum(1 for term in check_terms if term.lower() in response_lower)
    return found / len(check_terms)


# ============================================================
# MAIN: Run all tasks
# ============================================================

if __name__ == "__main__":
    # Run selective attention
    df1 = pd.DataFrame(SELECTIVE_ATTENTION_DATA)
    results1 = selective_attention.evaluate(llm=[kbench.llm], evaluation_data=df1)
    print("=== Selective Attention Results ===")
    print(results1.as_dataframe())
    
    # Run attention shifting
    df2 = pd.DataFrame(ATTENTION_SHIFTING_DATA)
    results2 = attention_shifting.evaluate(llm=[kbench.llm], evaluation_data=df2)
    print("\n=== Attention Shifting Results ===")
    print(results2.as_dataframe())
    
    # Run sustained attention
    df3 = pd.DataFrame(SUSTAINED_ATTENTION_DATA)
    results3 = sustained_attention.evaluate(llm=[kbench.llm], evaluation_data=df3)
    print("\n=== Sustained Attention Results ===")
    print(results3.as_dataframe())
    
    # Run inattentional blindness
    df4 = pd.DataFrame(INATTENTIONAL_BLINDNESS_DATA)
    results4 = inattentional_blindness.evaluate(llm=[kbench.llm], evaluation_data=df4)
    print("\n=== Inattentional Blindness Results ===")
    print(results4.as_dataframe())
    
    # Run saliency awareness
    df5 = pd.DataFrame(SALIENCY_DATA)
    results5 = saliency_awareness.evaluate(llm=[kbench.llm], evaluation_data=df5)
    print("\n=== Saliency Awareness Results ===")
    print(results5.as_dataframe())


# Select main task for leaderboard submission
# %choose selective_attention
"""
Note: In the actual Kaggle notebook, uncomment the %choose line above 
and select which task to submit as the main benchmark task.
Each task should be submitted as a separate task notebook, 
then combined into a benchmark.
"""
