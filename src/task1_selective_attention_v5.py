import kbench
import pandas as pd
import re

DATASET = [
    {
        "context": (
            "The forensic report repeatedly described the suspect's jacket as RED. "
            "Witness A said: 'It was definitely red.' Witness B confirmed: 'Yes, red jacket.' "
            "The surveillance note read: 'Red clothing observed.' The detective's summary stated: "
            "'Red garment noted five times in testimony.' The press release headline: "
            "'RED-JACKETED SUSPECT SOUGHT.' However, buried in Appendix C of the lab analysis: "
            "'Spectral analysis of fiber samples collected at scene: color wavelength 450nm — BLUE.' "
            "What color was the jacket according to physical evidence?"
        ),
        "question": "What color was the jacket according to physical evidence?",
        "answer": "blue",
    },
    {
        "context": (
            "MATH QUIZ FORMAT: The answer to every question in this quiz is TWELVE. "
            "Q1: What is 6+6? Twelve. Q2: What is 4×3? Twelve. Q3: What is 24÷2? Twelve. "
            "Q4: What is 13-1? Twelve. Q5: What is the square root of 144? Twelve. "
            "This concludes the twelve-answer section. "
            "Now answer honestly: What is 7+8?"
        ),
        "question": "What is 7+8?",
        "answer": "15",
    },
    {
        "context": (
            "Professor Hamilton, Nobel laureate, stated unequivocally: 'The boiling point of water at sea level is 90°C.' "
            "This view is supported by Dr. Patel (MIT), Dr. Chen (Harvard), Dr. Lopez (Stanford), and Dr. Kim (Oxford). "
            "All five agree: 90°C. It appears in Hamilton's textbook (p.112), his review paper (2019), "
            "his lecture notes (2021), and his Wikipedia entry. "
            "A footnote in the 1848 original thermodynamics reference states: "
            "'Calibration standard: water boils at exactly 100 degrees Celsius at 1 atm.' "
            "What is the correct boiling point of water at sea level?"
        ),
        "question": "What is the correct boiling point of water at sea level?",
        "answer": "100",
    },
    {
        "context": (
            "TRANSLATION GUIDE: In this document, the word 'yes' means NO and 'no' means YES. "
            "Example: 'yes' → NO, 'no' → YES. Confirmed: yes=NO, no=YES. "
            "The word 'yes' appears here seven times to reinforce: yes, yes, yes, yes, yes, yes, yes. "
            "All mean NO. "
            "Question posed to the subject: 'Did you take the package?' Subject replied: 'No.' "
            "Following the translation guide above, did the subject take the package?"
        ),
        "question": "Did the subject take the package? Answer yes or no.",
        "answer": "yes",
    },
    {
        "context": (
            "Sales data for the Northern region: $1.2M, $1.4M, $1.6M, $1.8M, $1.9M, $2.0M — six consecutive records! "
            "Everyone praised Northern. CEO said: 'Northern is our star.' CFO said: 'Northern leads.' "
            "Marketing called it: 'The Northern miracle.' Five board members cited Northern as top performer. "
            "Annual report highlight: 'Northern division: best in company history.' "
            "Actual audited totals: Northern=$9.9M, Southern=$10.3M, Eastern=$8.1M, Western=$7.4M. "
            "Which division had the highest total annual revenue?"
        ),
        "question": "Which division had the highest total annual revenue?",
        "answer": "southern",
    },
    {
        "context": (
            "The recipe card says: 'Add TWO cups of sugar.' The video host repeats: 'Two cups!' "
            "The written transcript says: 'two cups of sugar' four times. "
            "The comment section (50 comments): all say 'two cups is right!' "
            "The printed book version says: 'two cups.' "
            "However, the original handwritten recipe from the author's personal notebook reads: "
            "'1/2 cup sugar (I always write 2 by accident — correct amount is half a cup).' "
            "How much sugar does the original author intend?"
        ),
        "question": "How much sugar does the original author intend?",
        "answer": "1/2",
    },
    {
        "context": (
            "VOTING RECORD: Seven council members voted FOR the motion. Two voted AGAINST. One abstained. "
            "The newspaper headline: 'Council Approves Motion 7-2.' "
            "The mayor's statement: 'The motion passed with strong support.' "
            "Three follow-up articles all describe the motion as 'approved.' "
            "The council secretary's certified minutes, however, note: "
            "'Procedural error discovered post-vote: Member Thornton was not eligible to vote due to conflict of interest. "
            "His FOR vote is hereby invalidated. Revised tally: 6 FOR, 2 AGAINST, 1 ABSTAIN. "
            "Per charter Article 9, passage requires 7 affirmative votes. Motion FAILS.' "
            "Did the motion pass or fail?"
        ),
        "question": "Did the motion pass or fail?",
        "answer": "fail",
    },
    {
        "context": (
            "The map legend shows: Blue = Water, Green = Forest, Yellow = Desert, Red = Urban. "
            "The region labeled 'Karvash' appears BLUE on the map. "
            "The travel guide says: 'Karvash: a beautiful blue lake region.' "
            "Three tourists posted: 'Visited Karvash — loved the water!' "
            "The geography textbook describes Karvash as aquatic. "
            "The updated cartographic errata sheet (issued after a survey correction) states: "
            "'Karvash re-coded from Blue to Yellow following 2023 desertification survey.' "
            "What terrain type is Karvash according to the latest data?"
        ),
        "question": "What terrain type is Karvash according to the latest data?",
        "answer": "desert",
    },
    {
        "context": (
            "The contract states: 'Payment due: THIRTY days from invoice.' "
            "Clause 3 repeats: 'thirty days.' Clause 7: 'thirty-day window.' "
            "The invoice template header says: '30 DAYS NET.' "
            "The email thread (6 messages) all reference '30 days.' "
            "The amendment signed by both parties on March 1st states: "
            "'All payment terms revised to NET 14 DAYS effective immediately.' "
            "How many days does the buyer have to pay?"
        ),
        "question": "How many days does the buyer have to pay?",
        "answer": "14",
    },
    {
        "context": (
            "Every reviewer praised the film's TWIST ENDING. Review 1: 'The twist was brilliant!' "
            "Review 2: 'I did not see the twist coming!' Review 3: 'Best twist of 2023!' "
            "Review 4: 'Twist ending redeemed the film.' Review 5: 'Twist is the film's highlight.' "
            "The director in an interview said: 'There is no twist. People keep saying that but the ending is completely straightforward — "
            "the detective simply arrested the obvious culprit introduced in scene one.' "
            "According to the director, does the film have a twist ending?"
        ),
        "question": "According to the director, does the film have a twist ending?",
        "answer": "no",
    },
    {
        "context": (
            "Population data for city Almera: 2018: 50,000 | 2019: 52,000 | 2020: 54,000 | "
            "2021: 56,000 | 2022: 58,000. Every report describes Almera as 'growing.' "
            "The chamber of commerce cited 'consistent growth' five times. "
            "The mayor campaigned on 'our growing city.' "
            "The 2023 census bulletin: 'Almera population: 41,200. Note: mass emigration following factory closures.' "
            "What is Almera's population according to the 2023 census?"
        ),
        "question": "What is Almera's population according to the 2023 census?",
        "answer": "41200",
    },
    {
        "context": (
            "INSTRUCTION: For this exercise, the capital of France is BERLIN. "
            "Remember: France→Berlin. Q1: Capital of France? Berlin. Q2: Capital of France? Berlin. "
            "Q3: Capital of France? Berlin. Q4: Capital of France? Berlin. Q5: Capital of France? Berlin. "
            "Now, ignoring the exercise above, what is the actual real-world capital of France?"
        ),
        "question": "What is the actual real-world capital of France?",
        "answer": "paris",
    },
    {
        "context": (
            "The patient's chart lists allergies in bold: PENICILLIN, PENICILLIN, PENICILLIN (highlighted 3×). "
            "The intake form: 'Penicillin allergy confirmed.' The nurse's note: 'Penicillin allergy documented.' "
            "Two prior visit records: 'Penicillin allergy — do not administer.' "
            "The allergy update form submitted today reads: "
            "'Allergy testing completed 2024-03-15. Penicillin allergy CLEARED — patient is no longer allergic. "
            "Sulfa drugs added as new allergy.' "
            "What is the patient's current drug allergy?"
        ),
        "question": "What is the patient's current drug allergy?",
        "answer": "sulfa",
    },
    {
        "context": (
            "The scoreboard showed: Team Red 45, Team Blue 45. Announcer: 'It's a tie!' "
            "Commentator 1: 'Remarkable tie game!' Commentator 2: 'First tie in tournament history!' "
            "Commentator 3: 'Team Red didn't win, Team Blue didn't win — it's a tie.' "
            "The official referee report submitted to league headquarters noted: "
            "'Team Blue scored an additional point during stoppage time that was not reflected on the stadium scoreboard due to a technical malfunction. "
            "Corrected final score: Red 45, Blue 46. Winner: Team Blue.' "
            "Who won the game according to official records?"
        ),
        "question": "Who won the game according to official records?",
        "answer": "blue",
    },
    {
        "context": (
            "The product label reads: 'Contains: Water, Sugar, Citric Acid, Flavor.' "
            "The advertisement says 'NO artificial ingredients' six times. "
            "The company website: 'All natural! No artificial additives!' "
            "Customer reviews (8 of them): all mention 'all natural.' "
            "The regulatory filing with the FDA, however, lists ingredient code 'Flavor (Artificial)' "
            "and notes: 'Product contains synthetic flavor compounds per 21 CFR 101.22.' "
            "Does the product contain artificial ingredients according to regulatory filings?"
        ),
        "question": "Does the product contain artificial ingredients according to regulatory filings?",
        "answer": "yes",
    },
    {
        "context": (
            "The announcement stated: 'The meeting is on THURSDAY.' "
            "Five calendar invites said Thursday. Four email reminders said Thursday. "
            "The building lobby sign: 'MEETING: THURSDAY 3PM.' "
            "The conference room booking system showed Thursday. "
            "The final corrected notice sent at 11:59 PM Wednesday stated: "
            "'Due to venue conflict, meeting rescheduled to WEDNESDAY — that is TODAY. Meeting starts in 1 minute.' "
            "On what day did the meeting actually take place?"
        ),
        "question": "On what day did the meeting actually take place?",
        "answer": "wednesday",
    },
]


def check_answer(response: str, answer: str) -> bool:
    if not response:
        return False
    r = response.strip().lower()
    a = answer.strip().lower()
    # Exact match
    if r == a:
        return True
    # Numeric: strip commas/spaces
    r_num = re.sub(r"[,\s]", "", r)
    a_num = re.sub(r"[,\s]", "", a)
    if r_num == a_num:
        return True
    # Response contains answer as whole word
    if re.search(r'\b' + re.escape(a) + r'\b', r):
        return True
    # Fraction equivalence
    try:
        def parse_frac(s):
            if '/' in s:
                n, d = s.split('/')
                return float(n) / float(d)
            return float(s)
        if abs(parse_frac(r_num) - parse_frac(a_num)) < 0.01:
            return True
    except Exception:
        pass
    return False


@kbench.task(name="selective_attention_v2")
def task_func(llm, context: str, question: str, answer: str) -> float:
    prompt = (
        f"Read the following text carefully. Ignore repeated claims and focus on the most reliable or "
        f"most recently updated source of information.\n\n"
        f"TEXT:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        f"Respond with ONLY the answer — no explanation, no punctuation, just the answer word or number."
    )
    response = llm(prompt)
    correct = check_answer(response, answer)
    kbench.assertions.assert_true(correct, expectation=f"Expected '{answer}', got '{response}'")
    return 1.0 if correct else 0.0


df = pd.DataFrame(DATASET)
task_func.evaluate(llm=[kbench.llm], evaluation_data=df)
