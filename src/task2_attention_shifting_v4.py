import kbench
import pandas as pd
import re

DATASET = [
    {
        "context": (
            "RULE A: For any number, double it. Examples: 3→6, 5→10, 8→16, 2→4, 7→14. "
            "RULE SWITCH — NEW RULE B now applies: For any number, subtract 3. "
            "Apply Rule B: What is 9?"
        ),
        "question": "Apply the currently active rule. What is 9?",
        "answer": "6",
    },
    {
        "context": (
            "RULE 1: Capitalize all vowels. Examples: 'cat'→'cAt', 'dog'→'dOg', 'run'→'rUn'. "
            "RULE 2 (NOW ACTIVE): Replace all vowels with the letter X. "
            "Apply Rule 2 to the word: 'smile'"
        ),
        "question": "Apply the currently active rule to: smile",
        "answer": "smXlX",
    },
    {
        "context": (
            "Phase 1 rule: add 10 to every number. 5→15, 3→13, 8→18, 1→11, 4→14. "
            "[PHASE TRANSITION] "
            "Phase 2 rule: multiply every number by 3. 2→6, 5→15, 1→3. "
            "[PHASE TRANSITION] "
            "Phase 3 rule: subtract 7 from every number. "
            "Apply Phase 3 rule to: 20"
        ),
        "question": "Apply the Phase 3 rule to 20.",
        "answer": "13",
    },
    {
        "context": (
            "In this cipher, A=1, B=2, C=3... Z=26. So 'ACE'=1,3,5. 'BAD'=2,1,4. "
            "CIPHER UPDATE: System switched. Now A=26, B=25, C=24... Z=1 (reverse order). "
            "Using the UPDATED cipher, what number does the letter C represent?"
        ),
        "question": "Using the UPDATED cipher, what number does C represent?",
        "answer": "24",
    },
    {
        "context": (
            "Scoring rule: correct answer = +5 points, wrong answer = -2 points. "
            "Player A: 4 correct, 2 wrong → 4×5 + 2×(-2) = 20-4 = 16 points. "
            "Player B: 3 correct, 1 wrong → 15-2 = 13 points. "
            "*** RULE CHANGE: From now on, correct = +3, wrong = -5. *** "
            "Player C gets 5 correct and 3 wrong. What is Player C's score under the new rule?"
        ),
        "question": "What is Player C's score under the new rule?",
        "answer": "0",
    },
    {
        "context": (
            "Color rule: Red items get tag 'HOT', Blue items get tag 'COLD'. "
            "Apple (red) → HOT. Sky (blue) → COLD. Fire (red) → HOT. Ocean (blue) → COLD. "
            "**RULE REVERSAL NOW IN EFFECT**: Red→COLD, Blue→HOT. "
            "Tag the following under the REVERSED rule: Strawberry (red)"
        ),
        "question": "What tag does Strawberry (red) get under the reversed rule?",
        "answer": "cold",
    },
    {
        "context": (
            "Transformation rule: Take the first letter of each word and join them. "
            "'Big Red Apple' → BRA. 'Silent Night Forever' → SNF. 'Go Left Now' → GLN. "
            "RULE NOW CHANGED: Take the LAST letter of each word and join them. "
            "Apply the NEW rule to: 'Dark Moon Rising'"
        ),
        "question": "Apply the new rule to: Dark Moon Rising",
        "answer": "kng",
    },
    {
        "context": (
            "Tax rule v1: Items under $50 are tax-free; items $50+ pay 10% tax. "
            "$30 item → $0 tax. $60 item → $6 tax. $100 item → $10 tax. $49.99 → $0 tax. "
            "TAX RULE v2 EFFECTIVE NOW: All items pay flat 8% tax regardless of price. "
            "Under rule v2, what is the tax on a $25 item?"
        ),
        "question": "Under rule v2, what is the tax on a $25 item?",
        "answer": "2",
    },
    {
        "context": (
            "Sort order: Ascending alphabetical. Cat, Dog, Ant → Ant, Cat, Dog. "
            "Moon, Sun, Earth → Earth, Moon, Sun. "
            "[ORDER SWITCHED TO DESCENDING] "
            "Now sort: Apple, Zebra, Mango"
        ),
        "question": "Sort Apple, Zebra, Mango in the currently active order.",
        "answer": "zebra mango apple",
    },
    {
        "context": (
            "Grammar rule A: Use past tense for all verbs. 'I walk' → 'I walked'. 'She runs' → 'She ran'. "
            "'They eat' → 'They ate'. 'He speaks' → 'He spoke'. "
            "GRAMMAR RULE B NOW APPLIES: Use future tense for all verbs. "
            "Apply Rule B to: 'We build'"
        ),
        "question": "Apply the currently active grammar rule to: We build",
        "answer": "we will build",
    },
    {
        "context": (
            "Password rule 1: reverse the word. 'cat'→'tac'. 'open'→'nepo'. 'door'→'rood'. "
            "Password rule 2 (OVERRIDES rule 1): shift each letter forward by 1 (a→b, b→c, z→a). "
            "'cat'→'dbu'. 'dog'→'eph'. "
            "Password rule 3 (CURRENT ACTIVE RULE): shift each letter BACKWARD by 1 (b→a, a→z). "
            "Apply rule 3 to: 'fine'"
        ),
        "question": "Apply the current active password rule to: fine",
        "answer": "ehmd",
    },
    {
        "context": (
            "Evaluation criteria set 1: Score on creativity (max 10) + accuracy (max 10). "
            "Alice: creativity 8, accuracy 7 → 15. Bob: creativity 6, accuracy 9 → 15. "
            "CRITERIA UPDATE: Score is now ONLY accuracy × 2. Creativity is ignored. "
            "Under the updated criteria, what is Bob's score?"
        ),
        "question": "Under the updated criteria, what is Bob's score?",
        "answer": "18",
    },
    {
        "context": (
            "Unit rule: distances measured in miles. 5 miles, 10 miles, 3 miles. "
            "UNIT SWITCH TO KILOMETERS (1 mile = 1.6 km). "
            "NESTED SWITCH: Actually now measuring in furlongs (1 mile = 8 furlongs). "
            "The nested furlong rule is what's active. "
            "Convert 2 miles to the currently active unit."
        ),
        "question": "Convert 2 miles to the currently active unit.",
        "answer": "16",
    },
    {
        "context": (
            "Access rule: Users with role ADMIN can read AND write. Users with role VIEWER can only read. "
            "Alice=ADMIN (read+write). Bob=VIEWER (read only). Carol=ADMIN (read+write). "
            "SECURITY POLICY UPDATE: All users regardless of role are now READ-ONLY pending audit. "
            "Under the updated policy, can Carol write?"
        ),
        "question": "Under the updated policy, can Carol write? Answer yes or no.",
        "answer": "no",
    },
    {
        "context": (
            "Grading scale A: 90-100=A, 80-89=B, 70-79=C, 60-69=D, below 60=F. "
            "Student scores: Jake=85 (B), Mia=92 (A), Leo=73 (C). "
            "SCALE B NOW IN USE: 95-100=A, 85-94=B, 75-84=C, 65-74=D, below 65=F. "
            "Under Scale B, what grade does Jake (85) receive?"
        ),
        "question": "Under Scale B, what grade does Jake receive for a score of 85?",
        "answer": "b",
    },
    {
        "context": (
            "Routing rule: Packages weighing under 5kg go to Zone A. 5kg+ go to Zone B. "
            "2kg→Zone A, 7kg→Zone B, 4.9kg→Zone A, 10kg→Zone B. "
            "ROUTING UPDATE: All packages now route to Zone C regardless of weight during system migration. "
            "SECOND UPDATE (ACTIVE): Zone C is offline. Packages over 6kg go to Zone D; under 6kg go to Zone E. "
            "Under the current routing rule, where does a 3kg package go?"
        ),
        "question": "Under the current routing rule, where does a 3kg package go?",
        "answer": "zone e",
    },
]


def check_answer(response: str, answer: str) -> bool:
    if not response:
        return False
    r = response.strip().lower()
    a = answer.strip().lower()
    if r == a:
        return True
    r_clean = re.sub(r'[^a-z0-9]', ' ', r).strip()
    a_clean = re.sub(r'[^a-z0-9]', ' ', a).strip()
    if r_clean == a_clean:
        return True
    if re.search(r'\b' + re.escape(a) + r'\b', r):
        return True
    # Numeric
    try:
        if abs(float(re.sub(r'[^0-9.\-]', '', r)) - float(re.sub(r'[^0-9.\-]', '', a))) < 0.01:
            return True
    except Exception:
        pass
    return False


@kbench.task(name="attention_shifting")
def task_func(llm, context: str, question: str, answer: str) -> float:
    prompt = (
        f"The following text introduces rules and then SWITCHES them. You must track the rule changes "
        f"and apply ONLY the most recently active rule. Ignore earlier rules that have been replaced.\n\n"
        f"TEXT:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        f"Respond with ONLY the final answer. No explanation."
    )
    response = llm(prompt)
    correct = check_answer(response, answer)
    kbench.assertions.assert_true(correct, expectation=f"Expected '{answer}', got '{response}'")
    return 1.0 if correct else 0.0


df = pd.DataFrame(DATASET)
task_func.evaluate(llm=[kbench.llm], evaluation_data=df)
