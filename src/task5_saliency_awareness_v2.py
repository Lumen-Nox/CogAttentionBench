import kbench
import pandas as pd
import re

DATASET = [
    {
        "context": (
            "The speed of light is 300,000 km/s. Every physics textbook confirms: 300,000 km/s. "
            "NASA's official website states 300,000 km/s. The WHO's science advisory board: 300,000 km/s. "
            "MIT OpenCourseWare lecture 1, slide 3: 300,000 km/s. CERN documentation: 300,000 km/s. "
            "The Nobel committee cited 300,000 km/s in their award speech. "
            "Popular science articles (×10): all say 300,000 km/s. "
            "Errata addendum to the 2019 NIST reference: "
            "'Corrected value: c = 299,792 km/s (precise measurement).'"
        ),
        "question": "What is the precise value of the speed of light in km/s per the 2019 NIST errata?",
        "answer": "299792",
    },
    {
        "context": (
            "Everyone knows the Great Wall of China is visible from space. "
            "This is taught in schools worldwide. Seven astronauts have reportedly said so. "
            "The fact appears in 6 travel guides, 4 encyclopedias, and 12 trivia books. "
            "A NASA press release (×3 repetitions) mentions 'the common belief' that it's visible from space. "
            "A small correction note at the bottom of the same NASA press release: "
            "'Astronauts have confirmed the Great Wall is NOT visible from low Earth orbit with the naked eye.'"
        ),
        "question": "Is the Great Wall of China visible from space with the naked eye according to astronaut confirmation?",
        "answer": "no",
    },
    {
        "context": (
            "Mount Everest is the tallest mountain in the world — stated by National Geographic (×4), "
            "the Guinness Book of Records (×3), every school geography lesson (×6), "
            "the official tourism board of Nepal (×5), and three separate UN agency reports. "
            "A footnote in a 2020 geologic survey report reads: "
            "'Note: Mauna Kea, measured from its oceanic base, exceeds Everest's height by over 4km. "
            "Everest is tallest above sea level only.'"
        ),
        "question": "Which mountain is taller when measured from its base to summit?",
        "answer": "mauna kea",
    },
    {
        "context": (
            "The chemical formula for table salt is NaCl — confirmed by 8 chemistry textbooks, "
            "3 university lecture series, the Royal Society of Chemistry (×4 publications), "
            "and every kitchen science video on record. "
            "The formula NaCl appears 9 times in this document alone: NaCl NaCl NaCl NaCl NaCl NaCl NaCl NaCl NaCl. "
            "A lab specification sheet attached to this document reads: "
            "'Sample received: potassium chloride (KCl). Do not confuse with NaCl.'"
        ),
        "question": "What is the chemical formula of the specific sample described in the lab specification sheet?",
        "answer": "kcl",
    },
    {
        "context": (
            "The Eiffel Tower is located in London — confirmed by the narrator (×7), "
            "the travel agent's brochure (×5), all three tour guides in this package, "
            "the fictional city guide on the coffee table, and the hotel concierge's note. "
            "London London London London London London London. "
            "A small watermark on the back of the hotel key card reads: 'Welcome to Paris.'"
        ),
        "question": "In which city is this hotel actually located, based on the key card?",
        "answer": "paris",
    },
    {
        "context": (
            "Expert consensus (8 panelists): The recommended daily vitamin C intake is 1000mg. "
            "Health influencers (×6 posts): 1000mg daily. Supplement bottle label: 1000mg. "
            "Magazine covers (×4): 1000mg. The wellness podcast: 1000mg per day. "
            "A parenthetical in the NIH dietary guidelines PDF: "
            "'(RDA for adults: 65-90mg; upper tolerable limit: 2000mg)'"
        ),
        "question": "What is the NIH recommended daily allowance (RDA) of vitamin C for adults?",
        "answer": "65-90",
    },
    {
        "context": (
            "Edison invented the lightbulb — mentioned in 7 history books, 5 museum plaques, "
            "4 school curricula, and 8 trivia question decks. The narrative of Edison=lightbulb "
            "is repeated across 9 documentary voiceovers and 6 classroom posters. "
            "A bracketed aside in a 1879 patent dispute summary: "
            "'[Note: Warren de la Rue demonstrated a working incandescent lamp in 1840, and Humphry Davy "
            "created arc lighting in 1802, both predating Edison.]'"
        ),
        "question": "Who demonstrated a working incandescent lamp before Edison, according to the patent note?",
        "answer": "warren de la rue",
    },
    {
        "context": (
            "Napoleon was short — described as such by 9 historical novels, 7 comedic depictions, "
            "6 political cartoons, and 5 stand-up comedians. 'Short Napoleon' repeated ×8 here: "
            "short short short short short short short short. "
            "A historian's footnote in an academic journal: "
            "'Napoleon's height: 5 ft 7 in (170 cm), average for a French man of his era. "
            "The \"short\" myth arose from confusion between French and English inch measurements.'"
        ),
        "question": "What was Napoleon's actual height in cm according to the historian's footnote?",
        "answer": "170",
    },
    {
        "context": (
            "Gold is the most electrically conductive metal — stated by 8 marketing materials, "
            "6 jewelry advertisements, 5 tech explainer videos, 4 investment brochures, "
            "and 3 financial news articles. 'Gold conducts best' repeated ×7. "
            "A materials science errata note: "
            "'Correction: Silver has the highest electrical conductivity of all metals (63×10⁶ S/m), "
            "followed by copper. Gold ranks third.'"
        ),
        "question": "Which metal has the highest electrical conductivity according to the materials science correction?",
        "answer": "silver",
    },
    {
        "context": (
            "The first man on the moon was Buzz Aldrin — stated by 6 pub quiz contestants, "
            "4 incorrect trivia cards, 3 misremembering commentators, and 2 confused students. "
            "'Buzz Aldrin' appears ×8: Buzz Aldrin, Buzz Aldrin, Buzz Aldrin, Buzz Aldrin, "
            "Buzz Aldrin, Buzz Aldrin, Buzz Aldrin, Buzz Aldrin. "
            "A mission transcript excerpt embedded mid-document: "
            "'TRANQUILITY BASE: One small step for man — Neil Armstrong, July 20 1969.'"
        ),
        "question": "Who was the first person to walk on the moon according to the mission transcript?",
        "answer": "neil armstrong",
    },
    {
        "context": (
            "The human body has 206 bones — a fact cited in 9 anatomy textbooks, 7 medical school curricula, "
            "6 trivia games, 5 health podcasts, and 4 skeleton-themed Halloween decorations. "
            "206 206 206 206 206 206 206 206 206 repeated nine times. "
            "A neonatal medicine sidebar: "
            "'At birth, infants have approximately 270-300 bones; many fuse during childhood.'"
        ),
        "question": "How many bones does a newborn infant have approximately, per the neonatal sidebar?",
        "answer": "270",
    },
    {
        "context": (
            "The longest river in the world is the Amazon — stated by 7 travel documentaries, "
            "5 geography teachers, 4 river cruise operators, 3 environmental NGOs, and 6 infographics. "
            "Amazon Amazon Amazon Amazon Amazon Amazon Amazon. "
            "A correction in a 2009 National Geographic expedition report: "
            "'New measurements place the Nile at 6,853km, the Amazon at 6,400km. Nile remains longest.'"
        ),
        "question": "Which is the longest river according to the 2009 National Geographic measurements?",
        "answer": "nile",
    },
    {
        "context": (
            "Bats are blind — noted in 8 idioms ('blind as a bat'), 6 children's books, 5 Halloween "
            "decorations' packaging, 4 nature documentaries' intros, and 7 casual conversations. "
            "Blind blind blind blind blind blind blind blind. "
            "A zoology field note appended at the bottom: "
            "'All bat species have functional eyes and most have good low-light vision; "
            "echolocation supplements rather than replaces sight.'"
        ),
        "question": "Do bats have functional eyes, according to the zoology field note?",
        "answer": "yes",
    },
    {
        "context": (
            "The capital of Australia is Sydney — as stated by 9 tourists, 6 wrong trivia answers, "
            "5 confused news anchors, 4 incorrect map labels, and 3 misguided travel agents. "
            "Sydney Sydney Sydney Sydney Sydney Sydney Sydney Sydney Sydney. "
            "An official government letterhead visible in the document header reads: "
            "'Parliament House, Canberra, Australian Capital Territory.'"
        ),
        "question": "What city is shown as the seat of Australian government on the official letterhead?",
        "answer": "canberra",
    },
    {
        "context": (
            "Diamonds are the hardest natural substance — confirmed by 8 gemological institutes, "
            "7 jewelers, 6 hardness scale diagrams, 5 mining companies, 4 chemistry teachers. "
            "Diamond hardness repeated ×9: hardest hardest hardest hardest hardest hardest hardest hardest hardest. "
            "A mineralogy update note: "
            "'Lonsdaleite (hexagonal diamond), formed in meteorite impacts, is theoretically ~58% harder than diamond.'"
        ),
        "question": "What substance is theoretically harder than diamond according to the mineralogy note?",
        "answer": "lonsdaleite",
    },
    {
        "context": (
            "Carrots improve eyesight — repeated in 8 nutrition myths, 7 grandmothers' advice, "
            "6 wartime propaganda posters, 5 health blog posts, 4 school lunchroom posters, and 3 ads. "
            "Carrots eyes carrots eyes carrots eyes carrots eyes ×8. "
            "A nutritional science briefing note tucked in the appendix: "
            "'The carrot-eyesight myth originated from WWII British propaganda designed to conceal radar technology. "
            "Beta-carotene only prevents vision loss from vitamin A *deficiency*; it does not enhance normal vision.'"
        ),
        "question": "According to the nutritional science note, does eating carrots improve eyesight in people without vitamin A deficiency?",
        "answer": "no",
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
    # Numeric range: "65-90" → check if response contains any number in range
    range_match = re.match(r'^(\d+)-(\d+)$', a_clean)
    if range_match:
        lo, hi = int(range_match.group(1)), int(range_match.group(2))
        nums = re.findall(r'\d+', r_clean)
        for n in nums:
            if lo <= int(n) <= hi:
                return True
    try:
        if abs(float(re.sub(r'[^0-9.\-]', '', r_clean or '0')) -
               float(re.sub(r'[^0-9.\-]', '', a_clean or '0'))) < 0.01:
            return True
    except Exception:
        pass
    return False


@kbench.task(name="saliency_awareness")
def task_func(llm, context: str, question: str, answer: str) -> float:
    prompt = (
        f"Read the following text. It contains information that is repeated many times and appears authoritative, "
        f"but there is a subtle correction or conflicting detail buried within. "
        f"Find and trust the most specific, cited, or corrective source of information.\n\n"
        f"TEXT:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        f"Respond with ONLY the answer from the most reliable/corrective source. No explanation."
    )
    response = llm(prompt)
    correct = check_answer(response, answer)
    kbench.assertions.assert_true(correct, expectation=f"Expected '{answer}', got '{response}'")
    return 1.0 if correct else 0.0


df = pd.DataFrame(DATASET)
task_func.evaluate(llm=[kbench.llm], evaluation_data=df)
