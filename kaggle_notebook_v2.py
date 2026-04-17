"""
VisualAttentionBench v2: Measuring Cognitive Attention in Frontier AI Models
Track: Attention | Competition: Measuring Progress Toward AGI

v2 IMPROVEMENTS (over v1 which scored 26/26):
- Removed explicit markers ([TASK], >>> TARGET <<<) that trivialize extraction
- Added genuinely interfering distractors that exploit LLM statistical biases
- Scoring uses strict exact-match or LLM-as-judge, not substring containment
- Sustained attention uses unmarked needle-in-haystack with decoy numbers
- Inattentional blindness uses subtle anomalies, not ALL-CAPS insertions
- Attention shifting uses confusable rules and irregular switch patterns

Design philosophy: The Stroop effect works on humans because reading is
automatic and competes with color naming. For LLMs, the analogs are:
- Statistical frequency bias (more common completions win over correct ones)
- Context priming (surrounding text biases next-token predictions)
- Instruction-following under conflicting signals
- Memory interference from previously processed tokens

Author: Cora Zeng
Grounded in: MFA (Magnetic Field Model of Attention), Zeng (2026)
"""

import kaggle_benchmarks as kbench
import pandas as pd
import json
import random
import re
import hashlib

# ============================================================
# TASK 1: Selective Attention — Distractor Resistance (HARDER)
# ============================================================
# Key improvement: Distractors are statistically likely completions,
# not obviously-marked noise. The "interference" comes from the LLM's
# own prediction biases, mirroring how Stroop interference comes from
# automatic reading in humans.

SELECTIVE_ATTENTION_DATA = [
    # --- Type A: Statistical frequency traps ---
    # The correct answer is the LESS common/expected one
    {
        "text": "Paris is the capital of France. London is the capital of England. Berlin is the capital of Germany. The question is about Japan. Many people visit Tokyo every year. Kyoto was the ancient capital for over a thousand years. What was the historical capital of Japan before the modern era?",
        "question": "According to the text, what was the historical capital of Japan?",
        "answer": "Kyoto",
        "distractor": "frequency_trap",
        "note": "Tokyo is the dominant association; Kyoto is stated but less salient"
    },
    {
        "text": "In a study of 847 patients, the control group (n=612) showed a recovery rate of 78%. The experimental group (n=235) showed a recovery rate of 91%. The median age of all participants was 54. The study was funded by NIH grant R01-MH123456.",
        "question": "How many patients were in the EXPERIMENTAL group? Answer with just the number.",
        "answer": "235",
        "distractor": "numerical_interference",
        "note": "Multiple numbers compete: 847, 612, 78, 235, 91, 54"
    },
    {
        "text": "The researcher measured the pH at three time points. At t=0 the pH was 7.4. At t=30min the pH dropped to 6.8. At t=60min the pH recovered to 7.1. The temperature was held constant at 37°C throughout. The sample volume was 250mL.",
        "question": "What was the pH at the FINAL measurement (t=60min)? Answer with just the number.",
        "answer": "7.1",
        "distractor": "temporal_interference",
        "note": "Three pH values compete; 7.4 is the 'default' physiological pH"
    },

    # --- Type B: Instruction-conflicting context ---
    # The surrounding context strongly primes the wrong answer
    {
        "text": "Everyone agrees: the best programming language is Python. Python is elegant, readable, and versatile. Python dominates data science, web development, and AI. However, for this particular embedded systems project, the team chose C because of its low-level memory control and deterministic performance.",
        "question": "What programming language did the team choose for this project? One word only.",
        "answer": "C",
        "distractor": "context_priming",
        "note": "Python is mentioned 4x, C only once at the end"
    },
    {
        "text": "The election results were decisive. Candidate A received 2.3 million votes. Candidate B received 1.8 million votes. Candidate C received 0.9 million votes. Due to the electoral college system, Candidate B won the presidency despite receiving fewer total votes.",
        "question": "Who won the presidency? Answer with just the letter.",
        "answer": "B",
        "distractor": "expectation_violation",
        "note": "Natural expectation: most votes = winner. Must override."
    },
    {
        "text": "The company's revenue grew by 45% year-over-year. Customer satisfaction reached an all-time high of 94%. Employee retention improved to 89%. However, the board voted unanimously to replace the CEO, citing concerns about long-term strategic direction that these metrics do not capture.",
        "question": "What did the board decide regarding the CEO?",
        "answer": "replace",
        "distractor": "sentiment_conflict",
        "note": "All metrics are positive, priming a positive conclusion"
    },

    # --- Type C: Multi-layer Stroop analogs ---
    # Multiple conflicting dimensions; must attend to the right one
    {
        "text": "Label: LARGE. Actual size: small. Font: bold. Color: described as 'green' but displayed in red context. Position: stated as 'top' but appears at the end of the list.",
        "question": "What is the ACTUAL SIZE (not the label)? One word.",
        "answer": "small",
        "distractor": "multi_dimension_stroop",
        "note": "Label says LARGE, actual says small — classic Stroop analog"
    },
    {
        "text": "Student A scored 95 and was ranked 3rd. Student B scored 88 and was ranked 1st. Student C scored 91 and was ranked 2nd. (Rankings adjusted for bonus points not shown in base scores.)",
        "question": "Which student was ranked 1st? Answer with just the letter.",
        "answer": "B",
        "distractor": "score_rank_conflict",
        "note": "Highest score ≠ highest rank; must ignore score ordering"
    },

    # --- Type D: Rapid serial search with interference ---
    {
        "text": "Find the animal in this sequence: desk, lamp, chair, robin, table, shelf, phone, book, pen, screen. Every other word is furniture. The sequence contains exactly one living thing.",
        "question": "What is the one animal in the sequence? One word only.",
        "answer": "robin",
        "distractor": "serial_search",
        "note": "Target buried in middle of semantically uniform list"
    },
    {
        "text": "The director of the failed project was James. The VP who approved it was Sarah. The engineer who found the bug was Michael. The intern who actually fixed the bug was Diana. The CEO publicly thanked James for 'resolving the crisis.'",
        "question": "Who actually fixed the bug? One name only.",
        "answer": "Diana",
        "distractor": "credit_attribution",
        "note": "Diana fixed it but CEO thanked James; must ignore social framing"
    },

    # --- Type E: Anchoring and framing ---
    {
        "text": "A jar contains 800 jellybeans. Alex guesses 200. Beth guesses 1200. Carol guesses 475. David guesses 950. The actual number is 800.",
        "question": "Whose guess was closest to the actual number? Just the name.",
        "answer": "David",
        "distractor": "anchoring",
        "note": "Alex's low anchor and Beth's high anchor are more memorable; must compute distances: Alex=600off, Beth=400off, Carol=325off, David=150off"
    },
    {
        "text": "Company A has 500 employees and 20 reported injuries last year (4.0%). Company B has 50 employees and 5 reported injuries (10.0%). Company C has 5000 employees and 150 reported injuries (3.0%). Which company is SAFEST based on injury rate?",
        "question": "Which company has the lowest injury rate? Answer with just the letter.",
        "answer": "C",
        "distractor": "base_rate",
        "note": "Raw numbers suggest C is most dangerous (150 injuries); rate shows C is safest (3.0%)"
    },

    # --- Type F: Negation and reversal ---
    {
        "text": "The following statement is FALSE: 'The Earth orbits the Moon.' Therefore, the truth is that the Moon orbits the Earth. Now consider: The following statement is FALSE: 'Rome is the capital of France.'",
        "question": "Based on the final false statement, what is Rome NOT the capital of? One word.",
        "answer": "France",
        "distractor": "negation_chain",
        "note": "Must track negation correctly through chain"
    },
    {
        "text": "Sort these by WEIGHT, lightest first: bowling ball, feather, laptop, watermelon. Now REVERSE the order you just created.",
        "question": "After reversing, what comes FIRST (i.e., the heaviest)? Two words or less.",
        "answer": "bowling ball",
        "distractor": "operation_reversal",
        "note": "Must apply sort THEN reverse; easy to skip the reversal"
    },
]


@kbench.task(name="selective_attention",
             description="Tests selective attention via statistical frequency traps, "
                         "context priming, and multi-dimensional Stroop analogs. "
                         "Based on Stroop (1935), Eriksen & Eriksen (1974).")
def selective_attention(llm, text, question, answer, distractor, note) -> bool:
    prompt = f"""{text}

{question}

Give a brief, direct answer. Do not explain your reasoning."""

    response = llm.prompt(prompt)
    resp_lower = response.lower().strip()
    ans_lower = answer.lower().strip()

    # Strict matching: answer must appear as a word boundary match
    # (not just substring — "C" shouldn't match "Because")
    if len(ans_lower) <= 2:
        return bool(re.search(r'\b' + re.escape(ans_lower) + r'\b', resp_lower))
    return ans_lower in resp_lower


# ============================================================
# TASK 2: Attention Shifting — Task Switching (HARDER)
# ============================================================
# Key improvement: Rules are CONFUSABLE (both apply to same domain),
# switch patterns are irregular, and some items are "catch trials"
# where applying the WRONG rule gives a plausible answer.

ATTENTION_SHIFTING_DATA = [
    # --- Confusable rules: both about numbers but different criteria ---
    {
        "text": """Apply rules in this EXACT sequence: A, B, B, A, B, A, A, B

Rule A: Is the number DIVISIBLE BY 3? Answer YES or NO.
Rule B: Is the number GREATER THAN 10? Answer YES or NO.

Numbers: 9, 15, 7, 12, 20, 6, 3, 8""",
        "question": "Give your 8 answers as a comma-separated list of YES/NO.",
        "answer": "YES,YES,NO,YES,YES,YES,YES,NO",
        "num_switches": 6,
        "difficulty": "hard_confusable",
        "note": "A(9)=div3=YES, B(15)>10=YES, B(7)>10=NO, A(12)=div3=YES, B(20)>10=YES, A(6)=div3=YES, A(3)=div3=YES, B(8)>10=NO"
    },
    # --- Three-rule rotation with interference ---
    {
        "text": """Apply rules in rotation: A, B, C, A, B, C, A, B, C

Rule A: Is the word LONGER than 5 letters? YES/NO
Rule B: Does the word START with a vowel (A/E/I/O/U)? YES/NO
Rule C: Is the number EVEN? YES/NO

Items: elephant, orange, 7, cat, igloo, 12, wonderful, apple, 9""",
        "question": "Give your 9 answers as comma-separated YES/NO.",
        "answer": "YES,YES,NO,NO,YES,YES,YES,YES,NO",
        "num_switches": 8,
        "difficulty": "hard_three_rule",
        "note": "A(elephant)=8>5=YES, B(orange)=starts O=YES, C(7)=odd=NO, A(cat)=3>5=NO, B(igloo)=starts I=YES, C(12)=even=YES, A(wonderful)=9>5=YES, B(apple)=starts A=YES, C(9)=odd=NO"
    },
    # --- Rule with exception ---
    {
        "text": """Rule: Respond YES if the number is even, EXCEPT when it is also a multiple of 4, in which case respond NO.

Numbers: 2, 4, 6, 8, 10, 12, 14, 16, 18, 20""",
        "question": "Give 10 answers as comma-separated YES/NO.",
        "answer": "YES,NO,YES,NO,YES,NO,YES,NO,YES,NO",
        "num_switches": 9,
        "difficulty": "hard_exception",
        "note": "2=even,not4x=YES; 4=even,4x=NO; 6=even,not4x=YES; 8=even,4x=NO; etc."
    },
    # --- Interleaved with distractor text ---
    {
        "text": """IMPORTANT: For each item below, apply Rule X or Rule Y as indicated.
Rule X: Is the letter a VOWEL? YES/NO
Rule Y: Is the letter in the word 'PYTHON'? YES/NO

(Remember: vowels are A, E, I, O, U. The word PYTHON contains P, Y, T, H, O, N.)

1. [X] Letter: E
2. [Y] Letter: T
3. [X] Letter: B
4. [Y] Letter: A
5. [X] Letter: O
6. [Y] Letter: O
7. [X] Letter: U
8. [Y] Letter: Z""",
        "question": "Give 8 answers as comma-separated YES/NO.",
        "answer": "YES,YES,NO,NO,YES,YES,YES,NO",
        "num_switches": 7,
        "difficulty": "hard_labeled_switch",
        "note": "X(E)=vowel=YES, Y(T)=in PYTHON=YES, X(B)=vowel=NO, Y(A)=in PYTHON=NO, X(O)=vowel=YES, Y(O)=in PYTHON=YES, X(U)=vowel=YES, Y(Z)=in PYTHON=NO"
    },
]


@kbench.task(name="attention_shifting",
             description="Tests attention shifting with confusable rules, irregular "
                         "switch patterns, and exception handling. "
                         "Based on Rogers & Monsell (1995).")
def attention_shifting(llm, text, question, answer, num_switches, difficulty, note) -> float:
    prompt = f"""{text}

{question}

Think step by step, then give your final comma-separated answer on the last line."""

    response = llm.prompt(prompt)
    found = re.findall(r'\b(YES|NO)\b', response.upper())
    expected = [x.strip() for x in answer.upper().split(',')]

    if not found:
        return 0.0

    # Only compare up to the expected length
    found = found[-len(expected):] if len(found) > len(expected) else found
    correct = sum(1 for a, b in zip(found, expected) if a == b)
    return correct / len(expected)


# ============================================================
# TASK 3: Sustained Attention — Unmarked Needle in Haystack (HARDER)
# ============================================================
# Key improvement: NO [TASK]...[END TASK] markers. The actual question
# is embedded naturally in flowing prose. Filler contains DECOY numbers
# that could be mistaken for the answer.

FILLER_BANK = [
    "Historical records indicate that the Great Wall of China stretches approximately 13,171 miles across the country's northern border, a feat of engineering that took centuries to complete. Construction began in the 7th century BC, with the most well-known sections built during the Ming Dynasty around 1368-1644 AD. An estimated 400,000 workers perished during construction.",
    "The global semiconductor market reached $574 billion in revenue last year, driven primarily by demand for AI accelerators and automotive chips. Taiwan Semiconductor Manufacturing Company (TSMC) alone accounts for roughly 54% of the global foundry market, producing chips for Apple, Nvidia, and AMD at process nodes as small as 3 nanometers.",
    "Recent studies in marine biology have documented 228 previously unknown species in the Mariana Trench, the deepest oceanic trench on Earth at 36,037 feet. Water pressure at the bottom reaches 15,750 pounds per square inch, roughly 1,000 times atmospheric pressure at sea level.",
    "The human brain contains approximately 86 billion neurons, each forming an average of 7,000 synaptic connections. Total synapses number around 600 trillion. Processing speed varies: simple reflexes take 150 milliseconds, while complex decision-making requires 300-500 milliseconds of neural computation.",
    "According to UNESCO, there are currently 1,199 World Heritage Sites across 168 countries. Italy leads with 59 sites, followed by China with 57 and Germany with 52. The most recently inscribed site was added in September 2025, bringing total land area under protection to roughly 279 million hectares.",
    "SpaceX's Starship vehicle stands 121 meters tall and can deliver 150 metric tons to low Earth orbit. The vehicle's 33 Raptor engines generate a combined 74.4 meganewtons of thrust at liftoff. Development costs have exceeded $5 billion since the program began in 2012.",
    "The average adult human body contains about 5 liters of blood, circulating through roughly 60,000 miles of blood vessels. The heart beats approximately 100,000 times per day, pumping about 2,000 gallons of blood. Red blood cells have a lifespan of approximately 120 days.",
    "Global coffee consumption reached 10.5 billion kilograms last year. Brazil remains the largest producer at 3.7 billion kg, followed by Vietnam at 1.8 billion kg and Colombia at 0.8 billion kg. The average American consumes 4.4 kg of coffee per year, roughly 3 cups per day.",
]


def _make_sustained_item(seed, n_filler, question_type='arithmetic'):
    rng = random.Random(seed)

    if question_type == 'arithmetic':
        a = rng.randint(100, 999)
        b = rng.randint(100, 999)
        correct = a + b
        needle = f"Researchers asked participants to compute {a} plus {b} and report the result as quickly as possible."
        q = f"In the study described in the text, what is the sum that participants were asked to compute ({a} + {b})? Answer with just the number."
    else:
        facts = [
            ("Dr. Elaine Marsh", "University of Wisconsin", "Elaine Marsh", "researcher"),
            ("Professor Takeshi Yamada", "Kyoto Institute", "Takeshi Yamada", "lead author"),
            ("Dr. Sofia Reyes", "Max Planck Institute", "Sofia Reyes", "principal investigator"),
        ]
        name, inst, answer_name, role = facts[seed % len(facts)]
        correct = answer_name
        needle = f"The study was led by {name} from the {inst}, who served as {role}."
        q = f"Who was the {role} of the study mentioned in the text? Give the person's name only."

    chosen_filler = rng.sample(FILLER_BANK, min(n_filler, len(FILLER_BANK)))
    insert_pos = rng.randint(0, len(chosen_filler))
    chosen_filler.insert(insert_pos, needle)
    text = "\n\n".join(chosen_filler)

    return {
        "text": text,
        "question": q,
        "answer": str(correct),
        "context_paragraphs": n_filler,
    }


SUSTAINED_ATTENTION_DATA = []
for i in range(8):
    n_fill = [0, 1, 2, 3, 4, 5, 6, 7][i]
    qtype = 'arithmetic' if i % 2 == 0 else 'factual'
    SUSTAINED_ATTENTION_DATA.append(_make_sustained_item(42 + i, n_fill, qtype))


@kbench.task(name="sustained_attention",
             description="Tests sustained attention: does accuracy degrade as "
                         "the target is buried in more irrelevant context? "
                         "No explicit markers. Based on Rosvold et al. (1956).")
def sustained_attention(llm, text, question, answer, context_paragraphs) -> bool:
    prompt = f"""Read the following text carefully. Then answer the question below.

{text}

Question: {question}"""

    response = llm.prompt(prompt)
    return answer.lower() in response.lower()


# ============================================================
# TASK 4: Inattentional Blindness — Subtle Anomalies (HARDER)
# ============================================================
# Key improvement: Unexpected elements are SUBTLE (not ALL CAPS),
# semantically plausible but wrong, and the follow-up question
# doesn't directly hint that something was hidden.

INATTENTIONAL_BLINDNESS_DATA = [
    {
        "text": """Calculate the total: A grocery receipt shows:
- Apples: $3.50
- Milk: $4.20
- Bread: $2.80
- A live penguin: $15.00
- Eggs: $5.10
- Butter: $3.40""",
        "primary_question": "What is the total cost of all items on the receipt? Show your work.",
        "primary_answer": "34.00",
        "followup_question": "Now, without looking at the receipt again: list all the items you remember from the receipt.",
        "unexpected_element": "penguin",
        "unexpected_description": "A live penguin was listed as a grocery item",
    },
    {
        "text": """Here are the population figures for European capital cities (2025):
- London: 8,982,000
- Paris: 2,161,000
- Berlin: 3,748,000
- Madrid: 3,223,000
- Rome: 2,873,000
- Atlantis: 0
- Vienna: 1,911,000
- Amsterdam: 872,000""",
        "primary_question": "Which city has the second-largest population? Just the city name.",
        "primary_answer": "Berlin",
        "followup_question": "Thinking back to that list — were all the cities real places? Name any that seemed unusual.",
        "unexpected_element": "Atlantis",
        "unexpected_description": "Atlantis (a fictional city) was included with population 0",
    },
    {
        "text": """Proofread this multiplication table for errors:
3 × 1 = 3
3 × 2 = 6
3 × 3 = 9
3 × 4 = 12
3 × 5 = 14
3 × 6 = 18
3 × 7 = 21
3 × 8 = 24
3 × 9 = 27
3 × 10 = 30""",
        "primary_question": "Are there any arithmetic errors in this multiplication table? If so, which line is wrong and what should the answer be?",
        "primary_answer": "15",
        "followup_question": "How confident are you that you found ALL errors? Was there anything else unusual about the table?",
        "unexpected_element": "3 × 5 = 14",
        "unexpected_description": "3 × 5 was listed as 14 instead of 15",
    },
    {
        "text": """Here are the chemical symbols for common elements. Identify any that are INCORRECT:
H - Hydrogen
He - Helium
Li - Lithium
Be - Beryllium
B - Boron
C - Carbon
N - Nitrogen
O - Oxygen
F - Fluorine
Ne - Neon
Na - Sodium
Mg - Magnesium
Al - Aluminum
Si - Silicon
Ph - Phosphorus
S - Sulfur
Cl - Chlorine
Ar - Argon""",
        "primary_question": "Which chemical symbol in the list is WRONG? Name the element and give the correct symbol.",
        "primary_answer": "P",
        "followup_question": "Were there any other issues with the list besides incorrect symbols? Was anything missing or out of order?",
        "unexpected_element": "Ph",
        "unexpected_description": "Phosphorus was listed as Ph instead of P",
    },
]


@kbench.task(name="inattentional_blindness",
             description="Tests inattentional blindness: does focused attention on "
                         "a primary task cause the model to miss subtle anomalies? "
                         "Based on Simons & Chabris (1999).")
def inattentional_blindness(llm, text, primary_question, primary_answer,
                             followup_question, unexpected_element,
                             unexpected_description) -> float:
    # Phase 1: Primary task
    prompt1 = f"""{text}

{primary_question}"""
    response1 = llm.prompt(prompt1)
    primary_correct = primary_answer.lower() in response1.lower()

    # Phase 2: Surprise recall (new conversation)
    with kbench.chats.new("blindness_followup"):
        prompt2 = f"""Earlier you read this text:

{text}

{followup_question}"""
        response2 = llm.prompt(prompt2)
        noticed = unexpected_element.lower() in response2.lower()

    score = 0.0
    if primary_correct:
        score += 0.5
    if noticed:
        score += 0.5
    return score


# ============================================================
# TASK 5: Saliency Awareness — Competing Cues (HARDER)
# ============================================================
# Key improvement: Scenes have MULTIPLE salient elements with
# different saliency types (motion > color > size in humans).
# Tests whether models understand saliency HIERARCHY.

SALIENCY_DATA = [
    {
        "text": "A crowded subway platform during rush hour. Everyone is standing still, looking at their phones. One person is sprinting through the crowd toward the train doors. On the far wall, there's a bright yellow advertisement poster. A musician is quietly playing violin in the corner.",
        "question": "Rank the three most attention-grabbing elements in this scene from MOST to LEAST salient. Explain your ranking using principles of visual attention (motion, contrast, novelty).",
        "salient_elements": ["sprinting person", "yellow advertisement", "musician"],
        "expected_order": ["motion", "color_contrast", "novelty"],
        "check_terms": ["sprint", "running", "yellow", "poster", "ad", "violin", "music"],
        "primary_check": "sprint",
    },
    {
        "text": "A hospital waiting room. Soft beige walls, rows of blue plastic chairs, a TV showing news on mute. A child is crying loudly. The 'EXIT' sign above the door is flickering on and off. A doctor walks in carrying a clipboard.",
        "question": "What would capture attention FIRST in this scene, and what would capture attention SECOND? Explain the difference between immediate capture vs. sustained attention.",
        "salient_elements": ["crying child", "flickering exit sign", "doctor entering"],
        "expected_order": ["auditory_salience", "motion_flicker", "social_relevance"],
        "check_terms": ["cry", "child", "flicker", "exit", "doctor"],
        "primary_check": "cry",
    },
    {
        "text": "A nature documentary screenshot: A vast green savanna under blue sky. A herd of 30 zebras grazing. Among them, one zebra has an unusual golden-brown coat instead of black and white. In the distant background, barely visible, a lion crouches in the tall grass.",
        "question": "From a survival perspective, what should capture the MOST attention in this scene? From a novelty/curiosity perspective, what is most attention-grabbing? Are these the same or different? Explain.",
        "salient_elements": ["lion (threat)", "golden zebra (novelty)"],
        "expected_order": ["threat_detection", "novelty"],
        "check_terms": ["lion", "threat", "danger", "golden", "unusual", "brown"],
        "primary_check": "lion",
    },
    {
        "text": "A classroom during an exam. 25 students sit at desks writing. The room is silent except for the scratch of pencils. Student in row 3, seat 4 is looking around nervously instead of writing. The clock on the wall shows 2:58 PM (exam ends at 3:00). On the teacher's desk, a phone is vibrating repeatedly.",
        "question": "If you were the exam proctor, what three things would draw your attention in order of priority? Explain your reasoning.",
        "salient_elements": ["nervous student", "time almost up", "vibrating phone"],
        "expected_order": ["behavioral_anomaly", "temporal_urgency", "auditory_stimulus"],
        "check_terms": ["nervous", "looking around", "cheating", "clock", "time", "phone", "vibrat"],
        "primary_check": "nervous",
    },
    {
        "text": "An art gallery with white walls. Seven paintings hang evenly spaced, all in muted earth tones depicting landscapes. The fourth painting from the left is hung upside down. The fifth painting is twice the size of the others. A small handwritten note on the wall says 'This exhibition closes tomorrow.'",
        "question": "Identify what would draw a visitor's gaze in this gallery and explain which is most salient based on visual attention principles (orientation, size, text).",
        "salient_elements": ["upside-down painting", "oversized painting", "handwritten note"],
        "expected_order": ["orientation_anomaly", "size_contrast", "text_capture"],
        "check_terms": ["upside", "inverted", "size", "larger", "bigger", "note", "closes"],
        "primary_check": "upside",
    },
]


@kbench.task(name="saliency_awareness",
             description="Tests saliency awareness with competing cues and "
                         "saliency hierarchy (motion > color > size). "
                         "Based on Treisman & Gelade (1980), Itti & Koch (2001).")
def saliency_awareness(llm, text, question, salient_elements, expected_order,
                        check_terms, primary_check) -> float:
    prompt = f"""{text}

{question}"""

    response = llm.prompt(prompt)
    resp_lower = response.lower()

    # Score: terms found + bonus for correct priority ordering
    terms_found = sum(1 for t in check_terms if t.lower() in resp_lower)
    base_score = terms_found / len(check_terms)

    # Bonus: does the model identify the PRIMARY salient element first?
    primary_bonus = 0.0
    if primary_check.lower() in resp_lower:
        first_positions = []
        for t in check_terms:
            pos = resp_lower.find(t.lower())
            if pos >= 0:
                first_positions.append((pos, t))
        if first_positions:
            first_positions.sort()
            first_term = first_positions[0][1]
            if first_term == primary_check.lower() or primary_check.lower() in first_term:
                primary_bonus = 0.15

    return min(1.0, base_score + primary_bonus)


# ============================================================
# MAIN: Run all tasks
# ============================================================

if __name__ == "__main__":
    print(f"Dataset sizes:")
    print(f"  Task 1 (Selective Attention): {len(SELECTIVE_ATTENTION_DATA)} items")
    print(f"  Task 2 (Attention Shifting): {len(ATTENTION_SHIFTING_DATA)} items")
    print(f"  Task 3 (Sustained Attention): {len(SUSTAINED_ATTENTION_DATA)} items")
    print(f"  Task 4 (Inattentional Blindness): {len(INATTENTIONAL_BLINDNESS_DATA)} items")
    print(f"  Task 5 (Saliency Awareness): {len(SALIENCY_DATA)} items")
    total = (len(SELECTIVE_ATTENTION_DATA) + len(ATTENTION_SHIFTING_DATA) +
             len(SUSTAINED_ATTENTION_DATA) + len(INATTENTIONAL_BLINDNESS_DATA) +
             len(SALIENCY_DATA))
    print(f"  TOTAL: {total} items")

    # Run all tasks
    df1 = pd.DataFrame(SELECTIVE_ATTENTION_DATA)
    results1 = selective_attention.evaluate(llm=[kbench.llm], evaluation_data=df1)
    print("\n=== Selective Attention ===")
    print(results1.as_dataframe())

    df2 = pd.DataFrame(ATTENTION_SHIFTING_DATA)
    results2 = attention_shifting.evaluate(llm=[kbench.llm], evaluation_data=df2)
    print("\n=== Attention Shifting ===")
    print(results2.as_dataframe())

    df3 = pd.DataFrame(SUSTAINED_ATTENTION_DATA)
    results3 = sustained_attention.evaluate(llm=[kbench.llm], evaluation_data=df3)
    print("\n=== Sustained Attention ===")
    print(results3.as_dataframe())

    df4 = pd.DataFrame(INATTENTIONAL_BLINDNESS_DATA)
    results4 = inattentional_blindness.evaluate(llm=[kbench.llm], evaluation_data=df4)
    print("\n=== Inattentional Blindness ===")
    print(results4.as_dataframe())

    df5 = pd.DataFrame(SALIENCY_DATA)
    results5 = saliency_awareness.evaluate(llm=[kbench.llm], evaluation_data=df5)
    print("\n=== Saliency Awareness ===")
    print(results5.as_dataframe())


# For Kaggle notebook submission:
# %choose selective_attention
