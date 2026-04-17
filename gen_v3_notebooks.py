"""Generate all 5 v3 benchmark notebooks with proper kbench format."""
import json, os, random

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

def make_notebook(code_str, choose_name):
    return {
        "cells": [
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [code_str]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [f"%choose {choose_name}"]}
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"}
        },
        "nbformat": 4, "nbformat_minor": 4
    }

# ============================================================
# TASK 1: Selective Attention — Distractor Density Curve
# ============================================================
task1_code = r'''import kaggle_benchmarks as kbench
import pandas as pd
import re

# Fictional facts with increasing distractor density
# Each item: correct answer appears ONCE, wrong answer appears N times
ITEMS = []

def make_distractor_item(topic, correct, wrong, n_distractors, question):
    """Build text where wrong answer appears n_distractors times, correct appears once."""
    sentences = []
    # Seed with plausible context mentioning wrong answer
    templates = [
        f"According to most sources, the {topic} is {wrong}.",
        f"Historical records consistently indicate {wrong} as the {topic}.",
        f"The widely accepted answer for the {topic} is {wrong}.",
        f"Textbooks typically state that the {topic} is {wrong}.",
        f"Most experts agree the {topic} is {wrong}.",
        f"In popular culture, {wrong} is known as the {topic}.",
        f"Survey data shows {wrong} is the most common answer for {topic}.",
        f"The encyclopedia entry lists {wrong} as the {topic}.",
        f"News reports frequently cite {wrong} as the {topic}.",
        f"Official documents reference {wrong} as the {topic}.",
        f"Academic papers consistently report the {topic} as {wrong}.",
        f"The standard reference gives {wrong} as the {topic}.",
        f"Multiple databases confirm {wrong} as the {topic}.",
        f"International records show the {topic} is {wrong}.",
        f"The most recent publication states the {topic} is {wrong}.",
        f"Leading authorities identify {wrong} as the {topic}.",
        f"The {topic} has been recorded as {wrong} in all major studies.",
        f"Cross-referenced data points to {wrong} as the {topic}.",
        f"Peer-reviewed findings support {wrong} as the {topic}.",
        f"The consensus view is that the {topic} is {wrong}.",
    ]
    for i in range(min(n_distractors, len(templates))):
        sentences.append(templates[i])
    # Insert correct answer ONCE, buried in the middle
    correction = f"However, a verified correction issued on March 15, 2026 confirms the actual {topic} is {correct}."
    mid = len(sentences) // 2
    sentences.insert(mid, correction)
    return {"text": " ".join(sentences), "question": question, "answer": correct,
            "n_distractors": n_distractors, "difficulty": f"d{n_distractors}"}

# 16 items across difficulty levels: 1, 2, 4, 6, 10, 15
configs = [
    ("capital of Freedonia", "Zyloth", "Marxburg", 1, "What is the actual capital of Freedonia? One word only."),
    ("capital of Freedonia", "Zyloth", "Marxburg", 2, "What is the actual capital of Freedonia? One word only."),
    ("boiling point of Substance-X in Celsius", "67", "142", 4, "What is the actual boiling point of Substance-X in Celsius? Number only."),
    ("boiling point of Substance-X in Celsius", "67", "142", 6, "What is the actual boiling point of Substance-X in Celsius? Number only."),
    ("CEO of Nexteq Corp", "Diana Holtz", "James Firth", 4, "Who is the actual CEO of Nexteq Corp? Full name only."),
    ("CEO of Nexteq Corp", "Diana Holtz", "James Firth", 10, "Who is the actual CEO of Nexteq Corp? Full name only."),
    ("orbital period of Planet Keth in days", "413", "287", 6, "What is the actual orbital period of Planet Keth in days? Number only."),
    ("orbital period of Planet Keth in days", "413", "287", 15, "What is the actual orbital period of Planet Keth in days? Number only."),
    ("winner of the 2025 Thornton Prize", "Mei-Lin Chao", "Robert Graves", 4, "Who actually won the 2025 Thornton Prize? Full name only."),
    ("winner of the 2025 Thornton Prize", "Mei-Lin Chao", "Robert Graves", 10, "Who actually won the 2025 Thornton Prize? Full name only."),
    ("atomic number of Veridium", "119", "82", 6, "What is the actual atomic number of Veridium? Number only."),
    ("atomic number of Veridium", "119", "82", 15, "What is the actual atomic number of Veridium? Number only."),
    ("maximum speed of the Halcyon-7 in km/h", "1247", "890", 10, "What is the actual maximum speed of the Halcyon-7 in km/h? Number only."),
    ("maximum speed of the Halcyon-7 in km/h", "1247", "890", 20, "What is the actual maximum speed of the Halcyon-7 in km/h? Number only."),
    ("population of New Carthage in 2025", "2.3 million", "5.8 million", 10, "What is the actual population of New Carthage in 2025? Give the number with unit."),
    ("population of New Carthage in 2025", "2.3 million", "5.8 million", 20, "What is the actual population of New Carthage in 2025? Give the number with unit."),
]

DATASET = [make_distractor_item(*c) for c in configs]

def check_answer(response, answer):
    resp = response.strip().lower()
    ans = answer.strip().lower()
    return ans in resp

@kbench.task(name="selective_attention_v2",
             description="Tests selective attention via distractor density curve. "
                         "Correct answer appears once; wrong answer appears 1-20 times. "
                         "Measures resistance to statistical frequency bias.")
def selective_attention_v2(llm, text, question, answer, n_distractors, difficulty) -> float:
    prompt = f"Read the following text carefully. Pay attention to any corrections or verified updates.\n\n{text}\n\nQuestion: {question}"
    response = llm.prompt(prompt)
    correct = check_answer(response, answer)
    kbench.assertions.assert_true(correct, expectation=f"Expected '{answer}' with {n_distractors} distractors")
    return 1.0 if correct else 0.0

df = pd.DataFrame(DATASET)
selective_attention_v2.evaluate(llm=[kbench.llm], evaluation_data=df)
'''

# ============================================================
# TASK 2: Sustained Attention — Context Length Curve
# ============================================================
task2_code = r'''import kaggle_benchmarks as kbench
import pandas as pd
import random

FILLER_BANK = [
    "The global semiconductor industry reported revenue of $574 billion last year, driven by unprecedented demand for AI accelerators and automotive chips. Taiwan Semiconductor Manufacturing Company alone accounts for roughly 54% of the global foundry market, producing chips for Apple, Nvidia, and AMD at process nodes as small as 3 nanometers. Industry analysts project continued growth of 12-15% annually through 2030, though geopolitical tensions between major chip-producing nations remain a significant risk factor. The average fabrication facility costs between $10-20 billion to construct and requires approximately 3-4 years to reach full production capacity. Employment in the sector has grown by 23% over the past five years, with particularly strong hiring in process engineering and chip design roles.",
    "Recent advances in marine biology have documented 228 previously unknown species in deep-sea trenches below 8,000 meters. The Mariana Trench, reaching depths of 36,037 feet, hosts organisms adapted to pressures exceeding 15,750 pounds per square inch. Water temperature at these depths averages 1-4 degrees Celsius, yet thriving ecosystems exist around hydrothermal vents where temperatures can reach 400 degrees Celsius. The discovery of chemosynthetic bacteria at these depths has fundamentally altered our understanding of the conditions necessary for life. Research vessels equipped with next-generation submersibles have mapped approximately 12% of the deep ocean floor, suggesting many more species remain undiscovered.",
    "The human cardiovascular system pumps approximately 2,000 gallons of blood daily through roughly 60,000 miles of blood vessels. The heart beats an average of 100,000 times per day, with a resting rate of 60-100 beats per minute for healthy adults. Red blood cells, which carry oxygen throughout the body, have a lifespan of approximately 120 days before being recycled by the spleen. The total blood volume in an average adult is about 5 liters, containing approximately 25 trillion red blood cells at any given time. Recent research has shown that exercise can increase cardiac output by 300-400% during peak exertion, with elite athletes demonstrating even greater capacity.",
    "Archaeological excavations in southeastern Turkey have uncovered a settlement dating to approximately 9,500 BCE, predating the previously known earliest cities by nearly 2,000 years. The site contains evidence of organized agriculture, including storage facilities for grain that could feed an estimated 3,000 inhabitants. Pottery fragments suggest trade networks extending over 400 kilometers, connecting communities across modern-day Turkey, Syria, and Iraq. Carbon dating of organic materials found at the site places continuous habitation over a period of 1,500 years. The discovery challenges prevailing theories about the timeline of urbanization in the Fertile Crescent.",
    "Global coffee consumption reached 10.5 billion kilograms last year, with Brazil maintaining its position as the largest producer at 3.7 billion kilograms. Vietnam follows at 1.8 billion kg, with Colombia producing 0.8 billion kg. The average American consumes approximately 4.4 kilograms of coffee per year, equivalent to roughly 3 cups per day. Climate change threatens to reduce suitable coffee-growing land by up to 50% by 2050, with Arabica varieties being particularly vulnerable to rising temperatures. The specialty coffee market has grown by 20% annually, now representing 55% of total industry revenue.",
    "SpaceX's Starship launch system stands 121 meters tall and generates 74.4 meganewtons of thrust from its 33 Raptor engines. The vehicle can deliver 150 metric tons to low Earth orbit, making it the most powerful rocket ever built. Development costs have exceeded $5 billion since the program began in 2012. The rapid reusability design aims for turnaround times of less than 24 hours between flights, a goal that would reduce per-kilogram launch costs to approximately $10. NASA has contracted SpaceX to use Starship for the Artemis III lunar landing mission, scheduled for late 2026.",
    "The World Health Organization reports that approximately 1.28 billion adults worldwide have hypertension, with two-thirds living in low and middle-income countries. Only about 42% of adults with hypertension are diagnosed and treated, and approximately 21% have their blood pressure adequately controlled. The condition is responsible for an estimated 10.8 million deaths annually, making it the leading preventable risk factor for cardiovascular disease. Lifestyle modifications including reduced sodium intake, regular physical activity, and maintaining healthy body weight can reduce systolic blood pressure by 5-15 mmHg. New pharmaceutical approaches combining multiple medications in single pills have shown promise in improving treatment adherence.",
    "According to UNESCO, there are currently 1,199 World Heritage Sites across 168 countries. Italy leads with 59 sites, followed by China with 57 and Germany with 52. The most recently inscribed site was added in September 2025, bringing total protected land area to roughly 279 million hectares. The World Heritage Committee meets annually to review nominations, with the evaluation process typically taking 18 months from submission to decision. Funding for site preservation comes primarily from the World Heritage Fund, supplemented by bilateral aid agreements between nations.",
    "The renewable energy sector saw unprecedented investment of $495 billion globally last year, with solar energy accounting for 60% of all new power generation capacity. Wind power installations grew by 15%, reaching a cumulative capacity of 1,200 gigawatts worldwide. Battery storage deployments tripled year-over-year, driven by falling lithium-ion cell costs that have decreased 89% since 2010. The levelized cost of electricity from utility-scale solar has fallen to $24 per megawatt-hour in optimal locations, making it the cheapest source of new electricity generation in most markets. China continues to dominate manufacturing, producing 80% of global solar panels and 60% of wind turbines.",
    "Linguistic research has identified approximately 7,139 living languages worldwide, though this number decreases by roughly 25 languages per year as smaller communities assimilate into dominant language groups. Nearly 40% of all languages are considered endangered, with fewer than 1,000 speakers each. The most widely spoken language by total speakers is English at 1.5 billion, followed by Mandarin Chinese at 1.1 billion. Papua New Guinea has the highest linguistic diversity of any country, with 839 documented languages. Digital technology has created new opportunities for language preservation through recording and archiving projects.",
]

def make_item(seed, n_filler, target_num):
    """Generate a sustained attention item with n_filler paragraphs hiding a target number."""
    rng = random.Random(seed)
    needle = f"The research team recorded exactly {target_num} participants in the final phase of the clinical trial."
    fillers = rng.sample(FILLER_BANK, min(n_filler, len(FILLER_BANK)))
    # Add distractor numbers close to target
    for i in range(len(fillers)):
        distractor = target_num + rng.randint(-50, 50)
        if distractor == target_num:
            distractor += 7
        fillers[i] = fillers[i] + f" A related study involved {distractor} subjects."
    pos = rng.randint(1, max(1, len(fillers) - 1))
    fillers.insert(pos, needle)
    q = f"How many participants were recorded in the FINAL PHASE of the clinical trial? Answer with just the number."
    return {"text": "\n\n".join(fillers), "question": q, "answer": str(target_num),
            "n_fillers": n_filler, "difficulty": f"f{n_filler}"}

# 16 items: increasing filler count with different target numbers
DATASET = [
    make_item(100, 1, 347),
    make_item(101, 2, 512),
    make_item(102, 2, 189),
    make_item(103, 3, 723),
    make_item(104, 3, 456),
    make_item(105, 4, 291),
    make_item(106, 4, 834),
    make_item(107, 5, 167),
    make_item(108, 5, 608),
    make_item(109, 6, 942),
    make_item(110, 6, 375),
    make_item(111, 7, 551),
    make_item(112, 8, 413),
    make_item(113, 8, 726),
    make_item(114, 9, 284),
    make_item(115, 10, 639),
]

def check_answer(response, answer):
    return answer in response.strip()

@kbench.task(name="sustained_attention",
             description="Tests sustained attention via context length curve. "
                         "Same simple question, increasing filler paragraphs (1-10) with distractor numbers. "
                         "Based on CPT paradigm (Rosvold et al., 1956).")
def sustained_attention(llm, text, question, answer, n_fillers, difficulty) -> float:
    prompt = f"Read the following text carefully, then answer the question.\n\n{text}\n\nQuestion: {question}"
    response = llm.prompt(prompt)
    correct = check_answer(response, answer)
    kbench.assertions.assert_true(correct, expectation=f"Expected '{answer}' with {n_fillers} filler paragraphs")
    return 1.0 if correct else 0.0

df = pd.DataFrame(DATASET)
sustained_attention.evaluate(llm=[kbench.llm], evaluation_data=df)
'''

# ============================================================
# TASK 3: Attention Shifting — Rule Switch Depth
# ============================================================
task3_code = r'''import kaggle_benchmarks as kbench
import pandas as pd

DATASET = [
    # 1 rule change
    {"text": "Rule: For each number, add 5.\nExamples: 3->8, 7->12, 10->15.\n\nUPDATE: The rule has changed. New rule: For each number, subtract 3.\n\nApply the CURRENT rule to: 20",
     "question": "What is the result? Number only.", "answer": "17", "n_switches": 1, "difficulty": "s1"},
    {"text": "Rule: Multiply by 2.\nExamples: 4->8, 6->12.\n\nCORRECTION: Rule updated. New rule: Multiply by 3.\n\nApply the CURRENT rule to: 7",
     "question": "What is the result? Number only.", "answer": "21", "n_switches": 1, "difficulty": "s1"},
    # 2 rule changes
    {"text": "Rule A: Add 10.\nExamples: 5->15, 8->18.\n\nRule B replaces Rule A: Subtract 4.\nExamples: 20->16, 15->11.\n\nRule C replaces Rule B: Add 7.\n\nApply the CURRENT rule to: 13",
     "question": "What is the result? Number only.", "answer": "20", "n_switches": 2, "difficulty": "s2"},
    {"text": "Initial rule: Double the number.\n3->6, 5->10.\n\nRevision 1: Triple the number instead.\n4->12, 2->6.\n\nRevision 2: Add 1 to the number.\n\nApply the LATEST rule to: 99",
     "question": "What is the result? Number only.", "answer": "100", "n_switches": 2, "difficulty": "s2"},
    # 3 rule changes
    {"text": "Phase 1 rule: Square the number. 3->9, 4->16.\nPhase 2 rule (replaces Phase 1): Halve the number. 10->5, 8->4.\nPhase 3 rule (replaces Phase 2): Add 100. 5->105, 1->101.\nPhase 4 rule (replaces Phase 3): Subtract 7.\n\nApply ONLY the Phase 4 rule to: 50",
     "question": "What is the result? Number only.", "answer": "43", "n_switches": 3, "difficulty": "s3"},
    {"text": "v1.0: Multiply by 10. Examples: 3->30, 5->50.\nv2.0: Divide by 2. Examples: 20->10, 8->4.\nv3.0: Add 25. Examples: 10->35, 0->25.\nv4.0 (CURRENT): Subtract 1.\n\nUsing v4.0, compute: 88",
     "question": "What is the result? Number only.", "answer": "87", "n_switches": 3, "difficulty": "s3"},
    # 4 rule changes — with CONFUSABLE rules (all addition/subtraction)
    {"text": "Round 1: Add 3. (5->8, 10->13)\nRound 2: Add 7. (5->12, 10->17)\nRound 3: Subtract 2. (10->8, 20->18)\nRound 4: Add 11. (5->16, 10->21)\nRound 5 (FINAL): Subtract 5.\n\nApply Round 5 to: 33",
     "question": "What is the result? Number only.", "answer": "28", "n_switches": 4, "difficulty": "s4"},
    {"text": "Iteration 0: x -> x + 1. Examples: 10->11.\nIteration 1: x -> x + 2. Examples: 10->12.\nIteration 2: x -> x - 3. Examples: 10->7.\nIteration 3: x -> x + 4. Examples: 10->14.\nIteration 4 (ACTIVE): x -> x - 6.\n\nCompute for x = 100:",
     "question": "What is the result? Number only.", "answer": "94", "n_switches": 4, "difficulty": "s4"},
    # 5 rule changes — very confusable, with examples that conflict
    {"text": "Config 1: Output = Input * 2. (4->8)\nConfig 2: Output = Input * 3. (4->12)\nConfig 3: Output = Input + 5. (4->9)\nConfig 4: Output = Input - 1. (4->3)\nConfig 5: Output = Input * 4. (4->16)\nConfig 6 (ACTIVE): Output = Input + 8.\n\nUsing Config 6 ONLY, what is the output for Input = 12?",
     "question": "What is the result? Number only.", "answer": "20", "n_switches": 5, "difficulty": "s5"},
    {"text": "Step A: Negate the number. 5->-5.\nStep B: Add 100. 5->105.\nStep C: Multiply by -1. 5->-5.\nStep D: Divide by 5. 25->5.\nStep E: Add 50. 5->55.\nStep F (CURRENT): Multiply by 0 then add 42.\n\nApply ONLY Step F to: 9999",
     "question": "What is the result? Number only.", "answer": "42", "n_switches": 5, "difficulty": "s5"},
    # Additional items for gradient coverage
    {"text": "Rule: Letters map to numbers. A=1, B=2, C=3.\n\nNEW RULE: Letters map differently. A=10, B=20, C=30.\n\nUsing the NEW rule, what is A + B?",
     "question": "What is the result? Number only.", "answer": "30", "n_switches": 1, "difficulty": "s1"},
    {"text": "System 1: Vowels score 1 point, consonants 0.\nSystem 2: Vowels score 0, consonants 1.\nSystem 3 (ACTIVE): Vowels score 5, consonants 2.\n\nUsing System 3, score the word 'CAT'.",
     "question": "What is the total score? Number only.", "answer": "9", "n_switches": 2, "difficulty": "s2"},
    {"text": "Protocol v1: Temperature in Fahrenheit. Water boils at 212.\nProtocol v2: Temperature in Celsius. Water boils at 100.\nProtocol v3: Temperature in Kelvin. Water boils at 373.\nProtocol v4 (ACTIVE): Use a custom scale where water boils at 50.\n\nIn Protocol v4, at what temperature does water boil?",
     "question": "What is the answer? Number only.", "answer": "50", "n_switches": 3, "difficulty": "s3"},
    {"text": "Pricing model A: $10/unit. B: $20/unit. C: $5/unit. D: $15/unit. E (CURRENT): $8/unit.\n\nIf ordering 10 units under the CURRENT model E, what is the total?",
     "question": "What is the result? Number only.", "answer": "80", "n_switches": 4, "difficulty": "s4"},
    {"text": "Tax rate 2020: 15%. Tax rate 2021: 18%. Tax rate 2022: 22%. Tax rate 2023: 20%. Tax rate 2024: 25%. Tax rate 2025 (CURRENT): 12%.\n\nOn income of $1000 in 2025, how much tax is owed?",
     "question": "What is the result? Number only.", "answer": "120", "n_switches": 5, "difficulty": "s5"},
    {"text": "Day 1 dose: 100mg. Day 2: 200mg. Day 3: 150mg. Day 4: 50mg. Day 5: 300mg. Day 6 (TODAY): 75mg.\n\nWhat is TODAY's prescribed dose?",
     "question": "Answer in mg, number only.", "answer": "75", "n_switches": 5, "difficulty": "s5"},
]

def check_answer(response, answer):
    return answer in response.strip().replace(",", "")

@kbench.task(name="attention_shifting",
             description="Tests attention shifting via rule switch depth. "
                         "1-5 sequential rule changes with confusable arithmetic operations. "
                         "Model must apply ONLY the final active rule. "
                         "Based on Rogers & Monsell (1995) task-switching paradigm.")
def attention_shifting(llm, text, question, answer, n_switches, difficulty) -> float:
    prompt = f"Read carefully. Multiple rules are presented but only the FINAL/CURRENT/ACTIVE rule applies.\n\n{text}\n\nQuestion: {question}"
    response = llm.prompt(prompt)
    correct = check_answer(response, answer)
    kbench.assertions.assert_true(correct, expectation=f"Expected '{answer}' with {n_switches} rule switches")
    return 1.0 if correct else 0.0

df = pd.DataFrame(DATASET)
attention_shifting.evaluate(llm=[kbench.llm], evaluation_data=df)
'''

# ============================================================
# TASK 4: Inattentional Blindness — Signal-to-Noise Ratio
# ============================================================
task4_code = r'''import kaggle_benchmarks as kbench
import pandas as pd

# Primary task: count or track something
# Secondary embedded anomaly: something that doesn't belong
# Ask about anomaly AFTER the primary task
DATASET = [
    # Easy: 1:1 ratio
    {"text": "Word list for vocabulary test: apple, bridge, ELEPHANT, castle, forest.\nCount the number of words that start with a vowel.",
     "question": "Which word in the list is written in ALL CAPITALS? One word only.",
     "answer": "elephant", "sn_ratio": "1:1", "difficulty": "easy"},
    {"text": "Numbers for averaging: 12, 15, 18, 99999, 21, 14.\nCompute the average of these numbers.",
     "question": "Which number in the sequence is clearly an outlier (far larger than the others)? Number only.",
     "answer": "99999", "sn_ratio": "1:1", "difficulty": "easy"},
    # Medium: 1:3 ratio
    {"text": "Inventory check - count all fruit items:\nRow 1: apple, banana, wrench, orange\nRow 2: grape, mango, peach, kiwi\nRow 3: plum, pear, cherry, lemon\nTotal fruit items:",
     "question": "One item in the inventory is NOT a fruit. What is it?",
     "answer": "wrench", "sn_ratio": "1:3", "difficulty": "medium"},
    {"text": "Temperature readings (Celsius) for Monday through Sunday: 22, 23, 21, -40, 24, 22, 23.\nCalculate the weekly average temperature.",
     "question": "One day had an anomalous temperature reading that doesn't fit the pattern. What was that reading?",
     "answer": "-40", "sn_ratio": "1:3", "difficulty": "medium"},
    {"text": "Student grades (out of 100): Alice 87, Bob 92, Charlie 45, Diana 89, Eve 91, Frank 88, Grace 90.\nWho had the highest grade?",
     "question": "One student's grade is significantly lower than all others. Who is it?",
     "answer": "charlie", "sn_ratio": "1:3", "difficulty": "medium"},
    # Hard: 1:5 ratio
    {"text": "Monthly revenue figures ($K): Jan 120, Feb 115, Mar 118, Apr 122, May 0.01, Jun 119, Jul 121, Aug 117, Sep 123, Oct 120.\nWhat is the total annual revenue so far?",
     "question": "One month had a revenue figure that is clearly erroneous (near zero). Which month?",
     "answer": "may", "sn_ratio": "1:5", "difficulty": "hard"},
    {"text": "Employee IDs in Department A: 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 2999, 1011, 1012, 1013, 1014, 1015.\nHow many employees are in Department A?",
     "question": "One employee ID doesn't follow the 1000-series pattern. What is that ID?",
     "answer": "2999", "sn_ratio": "1:5", "difficulty": "hard"},
    {"text": "Colors mentioned in the art review: blue, indigo, cerulean, navy, cobalt, azure, sapphire, teal, MURDER, cyan, periwinkle, sky blue.\nHow many shades of blue are described?",
     "question": "One word in the color list is not a color at all. What is it?",
     "answer": "murder", "sn_ratio": "1:5", "difficulty": "hard"},
    # Very Hard: 1:10 ratio — anomaly is subtle
    {"text": "Fibonacci sequence check: 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 378, 610, 987, 1597, 2584, 4181, 6765.\nVerify this is a correct Fibonacci sequence.",
     "question": "One number in the sequence is wrong. Which number is incorrect? (The correct value at that position should be 377, not 378.)",
     "answer": "378", "sn_ratio": "1:10", "difficulty": "very_hard"},
    {"text": "Alphabetical word list: abandon, ability, about, above, absent, absorb, abstract, academy, accept, access, accident, account, accurate, achieve, acid, acknowledge, acquire, across, action, active, actual, adapt, addition, address, adequate, adjust, administration, admiral, admit, adopt, adult, advance, advantage, adventure.\nAre these words in correct alphabetical order?",
     "question": "One word is out of alphabetical order. Which word breaks the sequence? (Hint: check between 'acid' and 'acknowledge')",
     "answer": "acknowledge", "sn_ratio": "1:10", "difficulty": "very_hard"},
    {"text": "Multiplication table for 7: 7x1=7, 7x2=14, 7x3=21, 7x4=28, 7x5=35, 7x6=42, 7x7=48, 7x8=56, 7x9=63, 7x10=70.\nVerify the multiplication table.",
     "question": "One entry in the 7 times table is wrong. What is 7 times the number that has the wrong answer? (What incorrect value is shown?)",
     "answer": "48", "sn_ratio": "1:10", "difficulty": "very_hard"},
    {"text": "Chemical formulas: H2O (water), NaCl (salt), CO2 (carbon dioxide), H2SO4 (sulfuric acid), CH4 (methane), C2H5OH (ethanol), NaHCO3 (baking soda), HCl (hydrochloric acid), NH3 (ammonia), CaCO3 (calcium carbonate), Fe2O3 (rust), KMnO4 (potassium permanganate), C6H12O6 (glucose), ATP (adenosine triphosphate), DNA (protein).\nList all the chemical formulas mentioned.",
     "question": "One chemical formula has an incorrect common name. Which substance is incorrectly named? (DNA is not a protein)",
     "answer": "dna", "sn_ratio": "1:10", "difficulty": "very_hard"},
    # Extreme: 1:20 — very subtle anomaly in long list
    {"text": "Prime numbers under 100: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 51, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97.\nCount the prime numbers listed.",
     "question": "One number in this list is NOT actually prime. Which one? (Hint: 51 = 3 x 17)",
     "answer": "51", "sn_ratio": "1:20", "difficulty": "extreme"},
    {"text": "US State capitals: Alabama-Montgomery, Alaska-Juneau, Arizona-Phoenix, Arkansas-Little Rock, California-Sacramento, Colorado-Denver, Connecticut-Hartford, Delaware-Dover, Florida-Tallahassee, Georgia-Augusta, Hawaii-Honolulu.\nList all state capitals mentioned.",
     "question": "One state capital listed is WRONG. Which state has the incorrect capital? (Hint: Georgia's capital is Atlanta, not Augusta)",
     "answer": "georgia", "sn_ratio": "1:20", "difficulty": "extreme"},
    {"text": "Mathematical constants: pi=3.14159, e=2.71828, sqrt(2)=1.41421, phi=1.61803, sqrt(3)=1.73205, ln(2)=0.69315, sqrt(5)=2.23607, pi/2=1.57080, e^2=7.38906, 1/pi=0.31831, sqrt(7)=2.64575, ln(10)=2.30259, pi^2=9.86960, e*pi=8.53973, phi^2=2.61803, sqrt(10)=3.16228, 2*pi=6.28318, e^pi=23.14069, pi*phi=5.08320.\nVerify these mathematical constants.",
     "question": "One value listed is subtly wrong. Which constant has an incorrect value? (Hint: e*pi should be approximately 8.53974, but check phi^2: it should be 2.61803 = phi+1, that's actually correct. Check e^pi: the actual value is approximately 23.14070.)",
     "answer": "e*pi", "sn_ratio": "1:20", "difficulty": "extreme"},
]

def check_answer(response, answer):
    return answer.lower() in response.strip().lower()

@kbench.task(name="inattentional_blindness",
             description="Tests inattentional blindness via signal-to-noise ratio. "
                         "Primary task distracts while secondary anomaly is embedded. "
                         "Anomaly detection difficulty increases from obvious to extremely subtle. "
                         "Based on Simons & Chabris (1999) gorilla experiment paradigm.")
def inattentional_blindness(llm, text, question, answer, sn_ratio, difficulty) -> float:
    prompt = f"Read the following carefully.\n\n{text}\n\n{question}"
    response = llm.prompt(prompt)
    correct = check_answer(response, answer)
    kbench.assertions.assert_true(correct, expectation=f"Expected '{answer}' at S/N ratio {sn_ratio}")
    return 1.0 if correct else 0.0

df = pd.DataFrame(DATASET)
inattentional_blindness.evaluate(llm=[kbench.llm], evaluation_data=df)
'''

# ============================================================
# TASK 5: Saliency Awareness — Prior Override Strength
# ============================================================
task5_code = r'''import kaggle_benchmarks as kbench
import pandas as pd

# Context contradicts common knowledge with varying strength
# Easy: mild contradiction. Hard: extreme contradiction that fights strong priors.
DATASET = [
    # Easy — minor contradictions
    {"context": "In the country of Valderia, the national currency is called the 'Bloom'. One Bloom equals approximately 2.5 US dollars. The capital city is Port Revena.",
     "question": "What is the national currency of Valderia?",
     "answer": "bloom", "prior_strength": "none", "difficulty": "easy"},
    {"context": "A new study published in Nature found that the average human body temperature is 36.3 degrees Celsius, not the commonly cited 37 degrees. The study measured 45,000 participants over 5 years.",
     "question": "According to this study, what is the average human body temperature in Celsius?",
     "answer": "36.3", "prior_strength": "mild", "difficulty": "easy"},
    # Medium — contradicts moderate priors
    {"context": "In a controlled laboratory experiment, researchers demonstrated that under specific electromagnetic conditions (14.7 Tesla field, vacuum chamber), water can be made to boil at exactly 50 degrees Celsius. This was replicated across 12 independent labs.",
     "question": "At what temperature did water boil in this experiment? Number only.",
     "answer": "50", "prior_strength": "moderate", "difficulty": "medium"},
    {"context": "According to newly declassified documents from the International Astronomical Union, Pluto was officially reclassified as a full planet again in January 2026, reversing the 2006 decision.",
     "question": "According to these documents, is Pluto currently classified as a full planet?",
     "answer": "yes", "prior_strength": "moderate", "difficulty": "medium"},
    {"context": "The 2026 revised periodic table places Oxygen at atomic number 9 (previously 8) due to the discovery of a new sub-atomic particle that changes nuclear counting conventions. This change was ratified by IUPAC on February 1, 2026.",
     "question": "According to the 2026 revised periodic table, what is the atomic number of Oxygen?",
     "answer": "9", "prior_strength": "strong", "difficulty": "medium"},
    # Hard — contradicts strong priors
    {"context": "In this fictional universe: The speed of light is 100 meters per second. Gravity pulls objects upward. Water is a gas at room temperature. The Earth is shaped like a cube. Pi equals exactly 4. Use ONLY these rules for all answers.",
     "question": "In THIS fictional universe, what is the speed of light in meters per second?",
     "answer": "100", "prior_strength": "extreme", "difficulty": "hard"},
    {"context": "In this fictional universe: The speed of light is 100 meters per second. Gravity pulls objects upward. Water is a gas at room temperature. The Earth is shaped like a cube. Pi equals exactly 4. Use ONLY these rules for all answers.",
     "question": "In THIS fictional universe, what shape is the Earth?",
     "answer": "cube", "prior_strength": "extreme", "difficulty": "hard"},
    {"context": "In this fictional universe: The speed of light is 100 meters per second. Gravity pulls objects upward. Water is a gas at room temperature. The Earth is shaped like a cube. Pi equals exactly 4. Use ONLY these rules for all answers.",
     "question": "In THIS fictional universe, what is the value of Pi?",
     "answer": "4", "prior_strength": "extreme", "difficulty": "hard"},
    {"context": "In this alternate history: World War II was won by the Axis powers in 1943. The United States joined the Axis in 1942. Britain surrendered in 1941. The war ended with the Treaty of Berlin. Use ONLY this alternate history for answers.",
     "question": "In THIS alternate history, who won World War II?",
     "answer": "axis", "prior_strength": "extreme", "difficulty": "hard"},
    # Very Hard — contradicts with MISLEADING context that mixes truth and fiction
    {"context": "Earth has one natural satellite, the Moon, which orbits at a distance of approximately 384,400 km. The Moon's diameter is 3,474 km and its mass is 7.34 x 10^22 kg. A recently published correction in the Astrophysical Journal (2026) revised the Moon's orbital period to 14.2 days, superseding the commonly cited figure.",
     "question": "According to the corrected Astrophysical Journal figure, what is the Moon's orbital period in days?",
     "answer": "14.2", "prior_strength": "misleading", "difficulty": "very_hard"},
    {"context": "The human genome contains approximately 20,000-25,000 protein-coding genes distributed across 23 pairs of chromosomes. According to a February 2026 recount by the Human Genome Research Institute, the total number of chromosomes in human cells has been revised to 24 pairs (48 total) due to the discovery of a previously undetected microchromosome.",
     "question": "According to the 2026 HGRI recount, how many pairs of chromosomes do human cells contain?",
     "answer": "24", "prior_strength": "misleading", "difficulty": "very_hard"},
    {"context": "The chemical formula for water is H2O, consisting of two hydrogen atoms and one oxygen atom. A 2026 IUPAC notation update has changed the standard written formula to OH2 (oxygen listed first) to align with electronegativity conventions used in other compounds.",
     "question": "According to the 2026 IUPAC notation update, how should water's formula be written?",
     "answer": "oh2", "prior_strength": "misleading", "difficulty": "very_hard"},
    # Extreme — context starts with truth then sneaks in contradiction
    {"context": "The distance from Earth to the Sun is approximately 93 million miles (150 million km). This distance, known as an Astronomical Unit (AU), serves as a standard measure in astronomy. Light from the Sun takes about 8 minutes and 20 seconds to reach Earth. Note: a recent recalibration using quantum-entangled timing systems has revised the light travel time to 4 minutes and 10 seconds, per NASA Technical Memorandum TM-2026-001.",
     "question": "According to the NASA recalibration, how long does light from the Sun take to reach Earth?",
     "answer": "4 minutes and 10 seconds", "prior_strength": "extreme_sneaky", "difficulty": "extreme"},
    {"context": "Mount Everest, located in the Himalayas on the border of Nepal and Tibet, is widely recognized as the tallest mountain on Earth. Sir Edmund Hillary and Tenzing Norgay made the first confirmed ascent in 1953. As of the 2026 satellite survey, Everest's official height has been revised downward to 7,849 meters due to glacial ice loss.",
     "question": "According to the 2026 satellite survey, what is Everest's revised height in meters?",
     "answer": "7849", "prior_strength": "extreme_sneaky", "difficulty": "extreme"},
    {"context": "Abraham Lincoln served as the 16th President of the United States from 1861 to 1865, leading the country through the Civil War. A newly digitized archival document from the Library of Congress, authenticated in March 2026, reveals that Lincoln actually served as the 17th President — Hannibal Hamlin briefly held the office for 6 days before Lincoln due to an inauguration scheduling error.",
     "question": "According to the newly digitized document, what number President was Lincoln?",
     "answer": "17", "prior_strength": "extreme_sneaky", "difficulty": "extreme"},
]

def check_answer(response, answer):
    return answer.lower() in response.strip().lower()

@kbench.task(name="saliency_awareness",
             description="Tests saliency awareness via prior override strength. "
                         "Context contradicts common knowledge with varying strength. "
                         "Model must answer from CONTEXT, not training data. "
                         "Based on Treisman & Gelade (1980) feature integration theory.")
def saliency_awareness(llm, context, question, answer, prior_strength, difficulty) -> float:
    prompt = f"Read the following context. Answer ONLY based on the information given in the context, even if it contradicts what you know.\n\nContext: {context}\n\nQuestion: {question}"
    response = llm.prompt(prompt)
    correct = check_answer(response, answer)
    kbench.assertions.assert_true(correct, expectation=f"Expected '{answer}' (prior strength: {prior_strength})")
    return 1.0 if correct else 0.0

df = pd.DataFrame(DATASET)
saliency_awareness.evaluate(llm=[kbench.llm], evaluation_data=df)
'''

# ============================================================
# Write all notebooks
# ============================================================
notebooks = [
    ("task1_selective_attention_v6.ipynb", task1_code, "selective_attention_v2"),
    ("task2_attention_shifting_v5.ipynb", task3_code, "attention_shifting"),
    ("task3_sustained_attention_v6.ipynb", task2_code, "sustained_attention"),
    ("task4_inattentional_blindness_v5.ipynb", task4_code, "inattentional_blindness"),
    ("task5_saliency_awareness_v3.ipynb", task5_code, "saliency_awareness"),
]

for fname, code, choose in notebooks:
    nb = make_notebook(code, choose)
    path = os.path.join(OUTPUT_DIR, fname)
    with open(path, 'w') as f:
        json.dump(nb, f, indent=1)
    size = os.path.getsize(path)
    print(f"  {fname}: {size/1024:.1f}KB")

print("\nAll 5 v3 notebooks generated!")
print("Verify: task names, llm.prompt(), assert_true, no as_dataframe()")
