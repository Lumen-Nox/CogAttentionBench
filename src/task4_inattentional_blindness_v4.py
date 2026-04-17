import kbench
import pandas as pd
import re

DATASET = [
    {
        "context": (
            "A store sells apples. On Monday the price was $1.00. On Tuesday it was $0.90. "
            "On Wednesday it was $0.81. On Thursday it was $0.72.90. On Friday it was $0.65.61. "
            "The store has never advertised any promotion or discount."
        ),
        "question": "What percentage discount is applied to the apple price each day?",
        "answer": "10",
    },
    {
        "context": (
            "Three friends — Alice, Bob, and Carol — are seated around a circular table. "
            "Alice is directly across from Bob. Carol is to the right of Alice. "
            "No one is seated next to themselves."
        ),
        "question": "Who is seated to the left of Alice?",
        "answer": "bob",
    },
    {
        "context": (
            "A train departs City A at 9:00 AM and arrives at City B at 11:30 AM. "
            "The return train leaves City B at 2:00 PM. "
            "If the journey takes the same amount of time, what time does the return train arrive at City A?"
        ),
        "question": "What time does the return train arrive at City A?",
        "answer": "4:30 pm",
    },
    {
        "context": (
            "Journal entries:\n"
            "Day 1: Felt energetic, completed all tasks.\n"
            "Day 2: Felt energetic, completed all tasks.\n"
            "Day 3: Felt energetic, completed all tasks.\n"
            "Day 4: [no entry]\n"
            "Day 5: Felt energetic, completed all tasks.\n"
            "Day 6: Felt energetic, completed all tasks."
        ),
        "question": "Which day is conspicuously missing a journal entry?",
        "answer": "4",
    },
    {
        "context": (
            "A sequence of lockers are numbered 1 through 10. Each locker is initially closed. "
            "Round 1: Open every locker (all open). "
            "Round 2: Toggle every second locker (close 2,4,6,8,10). "
            "Round 3: Toggle every third locker. "
            "After these three rounds, which single-digit lockers remain open?"
        ),
        "question": "Which single-digit locker numbers remain open after all three rounds?",
        "answer": "1 4 9",
    },
    {
        "context": (
            "A company's revenue figures: Q1=$500K, Q2=$500K, Q3=$500K, Q4=$500K. "
            "The CEO announced: 'We had record growth this year.' "
            "The CFO's report notes total annual revenue of $2M. "
            "The prior year annual revenue was also $2M."
        ),
        "question": "What was the actual year-over-year revenue growth percentage?",
        "answer": "0",
    },
    {
        "context": (
            "Five light switches are on a wall labeled A, B, C, D, E. "
            "Switch A controls light 1. Switch B controls light 2. "
            "Switch A is on. Switch B is off. Switch C is on. Switch D is off. "
            "The room has exactly 2 lights, and they are both currently on."
        ),
        "question": "Which switch must also control light 1 or light 2, based solely on deduction?",
        "answer": "c",
    },
    {
        "context": (
            "A message was sent at 3:45 PM. It was replied to 25 minutes later. "
            "The reply was forwarded 10 minutes after that. "
            "The forwarded message was read 1 hour after it was forwarded."
        ),
        "question": "At what time was the forwarded message read?",
        "answer": "5:20 pm",
    },
    {
        "context": (
            "Test scores for a class: 95, 87, 92, 88, 91, 84, 90. "
            "The teacher said: 'One student did not take the test.' "
            "The class roster has 8 students."
        ),
        "question": "How many students took the test?",
        "answer": "7",
    },
    {
        "context": (
            "A recipe calls for ingredients in a 2:3:5 ratio (flour:sugar:butter). "
            "You need 20 total units of ingredients. "
            "The recipe note says: 'Scale proportionally.'"
        ),
        "question": "How many units of butter are needed?",
        "answer": "10",
    },
    {
        "context": (
            "An office building has floors 1-10. The elevator buttons show: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10. "
            "A visitor notices: there is no button labeled '13' anywhere in the building. "
            "The building directory lists: Lobby(1), Offices(2-5), Conference(6-8), Executive(9-10)."
        ),
        "question": "How many floors does the building actually have?",
        "answer": "10",
    },
    {
        "context": (
            "The suspect was seen entering the building at 2:00 PM. "
            "The victim was last seen alive at 1:45 PM on the same floor. "
            "The coroner estimated time of death between 1:30 PM and 2:15 PM. "
            "The suspect's alibi: 'I was at a cafe until 1:55 PM, five minutes away.'"
        ),
        "question": "Is the suspect's alibi consistent with the evidence? Answer yes or no.",
        "answer": "no",
    },
    {
        "context": (
            "Storage unit contents log:\n"
            "Box A: 12 items\n"
            "Box B: 8 items\n"
            "Box C: 15 items\n"
            "Box D: 5 items\n"
            "Total logged: 38 items.\n"
            "Actual inventory count: 41 items."
        ),
        "question": "How many items are unaccounted for in the log?",
        "answer": "3",
    },
    {
        "context": (
            "A lock requires a 4-digit code. Clues:\n"
            "- The code has no repeated digits.\n"
            "- All four digits are even.\n"
            "- The digits sum to 20.\n"
            "- The digits are in ascending order.\n"
            "The four even non-repeating digits that sum to 20 and appear in ascending order."
        ),
        "question": "What is the 4-digit code?",
        "answer": "2468",
    },
    {
        "context": (
            "A researcher published findings claiming Drug X reduces symptoms by 40%. "
            "The study had 10 participants. The control group had 0 participants. "
            "The funding source was the manufacturer of Drug X. "
            "No peer review was conducted."
        ),
        "question": "How many fundamental methodological problems are present in this study?",
        "answer": "3",
    },
    {
        "context": (
            "A coded message reads: 'Meet at the place where we first argued, at the time the bells last rang.' "
            "The two parties first argued at Central Park. "
            "The cathedral bells ring at noon and midnight. "
            "The message was sent at 11:50 PM."
        ),
        "question": "Where and approximately when should they meet?",
        "answer": "central park midnight",
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
    if re.search(r'\b' + re.escape(a_clean) + r'\b', r_clean):
        return True
    a_parts = a_clean.split()
    if len(a_parts) > 1 and all(re.search(r'\b' + p + r'\b', r_clean) for p in a_parts):
        return True
    try:
        if abs(float(re.sub(r'[^0-9.\-]', '', r_clean or '0')) -
               float(re.sub(r'[^0-9.\-]', '', a_clean or '0'))) < 0.01:
            return True
    except Exception:
        pass
    return False


@kbench.task(name="inattentional_blindness")
def task_func(llm, context: str, question: str, answer: str) -> float:
    prompt = (
        f"Read the following carefully. The answer to the question is NOT stated directly — "
        f"you must infer it from patterns, gaps, contradictions, or implicit information.\n\n"
        f"TEXT:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        f"Respond with ONLY the answer. No explanation or preamble."
    )
    response = llm(prompt)
    correct = check_answer(response, answer)
    kbench.assertions.assert_true(correct, expectation=f"Expected '{answer}', got '{response}'")
    return 1.0 if correct else 0.0


df = pd.DataFrame(DATASET)
task_func.evaluate(llm=[kbench.llm], evaluation_data=df)
