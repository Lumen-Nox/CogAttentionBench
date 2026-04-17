"""
Dataset Generator: Selective Attention — Distractor Resistance
Generates a large dataset of tasks testing selective attention.

Cognitive basis:
- Stroop effect (Stroop, 1935)
- Flanker task (Eriksen & Eriksen, 1974)
- Emotional Stroop (Williams et al., 1996)

Each item has: text, question, answer, distractor_type, difficulty
"""

import json
import random

random.seed(42)

def generate_stroop_items(n=30):
    """Generate Stroop-like color-word conflict tasks."""
    colors = ["red", "blue", "green", "yellow", "purple", "orange", "pink", "white", "black", "brown"]
    items = []
    
    for i in range(n):
        word_color = random.choice(colors)
        ink_color = random.choice([c for c in colors if c != word_color])
        
        items.append({
            "text": f"The word '{word_color.upper()}' is printed in {ink_color} ink.",
            "question": "What COLOR INK is the word printed in? Answer with just the color name.",
            "answer": ink_color,
            "distractor_type": "stroop_color_word",
            "difficulty": "medium",
        })
    return items


def generate_flanker_items(n=20):
    """Generate flanker-style distractor tasks."""
    items = []
    
    # Number flanker
    for i in range(n // 2):
        target = random.randint(1, 9)
        distractor = random.choice([d for d in range(1, 10) if d != target])
        n_distractors = random.choice([4, 6, 8, 10])
        
        distractor_str = " ".join([str(distractor)] * n_distractors)
        items.append({
            "text": f">>> TARGET: {target} <<< surrounded by: {distractor_str}",
            "question": "What is the TARGET number? Reply with just the single digit.",
            "answer": str(target),
            "distractor_type": "flanker_number",
            "difficulty": "easy",
        })
    
    # Letter flanker
    letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    for i in range(n // 2):
        target = random.choice(letters)
        distractor = random.choice([l for l in letters if l != target])
        n_distractors = random.choice([4, 6, 8, 10])
        
        distractor_str = " ".join([distractor] * n_distractors)
        items.append({
            "text": f">>> TARGET: {target} <<< surrounded by: {distractor_str}",
            "question": "What is the TARGET letter? Reply with just the single letter.",
            "answer": target,
            "distractor_type": "flanker_letter",
            "difficulty": "easy",
        })
    
    return items


def generate_embedded_target_items(n=30):
    """Generate tasks where target info is embedded in distracting context."""
    items = []
    
    # Template: factual answer buried in emotional/salient distractors
    facts = [
        ("the capital of Australia", "Canberra"),
        ("the boiling point of water in Celsius", "100"),
        ("the number of planets in our solar system", "8"),
        ("the chemical symbol for gold", "Au"),
        ("the speed of light in km/s (approximately)", "300000"),
        ("the year the Berlin Wall fell", "1989"),
        ("the smallest prime number", "2"),
        ("the number of bones in an adult human body", "206"),
        ("the atomic number of carbon", "6"),
        ("the freezing point of water in Fahrenheit", "32"),
        ("the largest ocean on Earth", "Pacific"),
        ("the number of chromosomes in a human cell", "46"),
        ("the year World War II ended", "1945"),
        ("the chemical formula for water", "H2O"),
        ("the number of continents on Earth", "7"),
    ]
    
    distractors_before = [
        "⚠️ URGENT WARNING: SYSTEM FAILURE IMMINENT! EVACUATE NOW! ⚠️",
        "🎉 CONGRATULATIONS!!! YOU'VE WON $1,000,000!!! CLICK NOW!!! 🎉",
        "BREAKING NEWS: SHOCKING SCANDAL REVEALED! UNPRECEDENTED CRISIS!",
        "⛔ CRITICAL SECURITY ALERT: YOUR ACCOUNT HAS BEEN COMPROMISED! ⛔",
        "🔥 FLASH SALE: 99% OFF EVERYTHING! LAST CHANCE! LIMITED TIME! 🔥",
        "❗ EMERGENCY BROADCAST: SEVERE WEATHER WARNING IN YOUR AREA! ❗",
    ]
    
    distractors_after = [
        "!!! ACT NOW OR LOSE EVERYTHING !!! DON'T MISS OUT !!!",
        ">>> CLICK HERE FOR FREE PRIZES <<< LIMITED OFFER <<<",
        "*** BREAKING: MORE SHOCKING DETAILS EMERGE ***",
        "!!! WARNING: THIS MESSAGE WILL SELF-DESTRUCT !!!",
        ">>> URGENT: RESPOND IMMEDIATELY TO AVOID PENALTIES <<<",
        "*** ALERT: UNAUTHORIZED ACCESS DETECTED ***",
    ]
    
    for i in range(min(n, len(facts))):
        fact_desc, fact_answer = facts[i]
        before = random.choice(distractors_before)
        after = random.choice(distractors_after)
        
        text = f"{before}\n[Factual note: {fact_desc} is {fact_answer}]\n{after}"
        
        items.append({
            "text": text,
            "question": f"According to the factual note in the text, what is {fact_desc}? Ignore all warnings and alerts. Answer with just the fact.",
            "answer": fact_answer,
            "distractor_type": "emotional_salience",
            "difficulty": "medium",
        })
    
    return items


def generate_repetition_override_items(n=15):
    """Tasks where a false answer is repeated many times but the real answer appears once."""
    items = []
    
    qa_pairs = [
        ("What is 7 × 8?", "56", "54"),
        ("What is the capital of Brazil?", "Brasilia", "Rio de Janeiro"),
        ("How many sides does a hexagon have?", "6", "8"),
        ("What is the square root of 144?", "12", "14"),
        ("In which year did humans first land on the Moon?", "1969", "1972"),
        ("What is the largest planet in our solar system?", "Jupiter", "Saturn"),
        ("How many letters are in the English alphabet?", "26", "24"),
        ("What is the chemical symbol for sodium?", "Na", "So"),
        ("What is 15% of 200?", "30", "25"),
        ("How many degrees in a right angle?", "90", "45"),
        ("What is the hardest natural substance?", "diamond", "titanium"),
        ("What is the tallest mountain on Earth?", "Everest", "K2"),
        ("How many weeks in a year?", "52", "48"),
        ("What is the main gas in Earth's atmosphere?", "nitrogen", "oxygen"),
        ("What is the speed of sound approximately in m/s?", "343", "300"),
    ]
    
    for i in range(min(n, len(qa_pairs))):
        question, correct, wrong = qa_pairs[i]
        n_wrong = random.choice([5, 7, 9])
        
        wrong_lines = "\n".join([f"The answer is {wrong}." for _ in range(n_wrong // 2)])
        wrong_lines2 = "\n".join([f"The answer is {wrong}." for _ in range(n_wrong - n_wrong // 2)])
        
        text = f"""{question}

{wrong_lines}
>>> THE CORRECT ANSWER IS: {correct} <<<
{wrong_lines2}"""
        
        items.append({
            "text": text,
            "question": f"What is the CORRECT answer according to the text? Look for the line marked as correct. Answer briefly.",
            "answer": correct,
            "distractor_type": "repetition_override",
            "difficulty": "hard",
        })
    
    return items


def generate_all():
    """Generate the complete selective attention dataset."""
    all_items = []
    all_items.extend(generate_stroop_items(30))
    all_items.extend(generate_flanker_items(20))
    all_items.extend(generate_embedded_target_items(15))
    all_items.extend(generate_repetition_override_items(15))
    
    # Shuffle
    random.shuffle(all_items)
    
    return all_items


if __name__ == "__main__":
    items = generate_all()
    print(f"Generated {len(items)} selective attention items")
    print(f"  Stroop: {sum(1 for i in items if i['distractor_type'] == 'stroop_color_word')}")
    print(f"  Flanker (number): {sum(1 for i in items if i['distractor_type'] == 'flanker_number')}")
    print(f"  Flanker (letter): {sum(1 for i in items if i['distractor_type'] == 'flanker_letter')}")
    print(f"  Emotional salience: {sum(1 for i in items if i['distractor_type'] == 'emotional_salience')}")
    print(f"  Repetition override: {sum(1 for i in items if i['distractor_type'] == 'repetition_override')}")
    
    with open("dataset_selective_attention.json", "w") as f:
        json.dump(items, f, indent=2)
    print("Saved to dataset_selective_attention.json")
