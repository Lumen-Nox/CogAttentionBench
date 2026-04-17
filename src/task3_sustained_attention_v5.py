import kbench
import pandas as pd
import re
import random

# ── filler paragraph bank (200+ words each) ─────────────────────────────────
FILLERS = [
    (
        "The global logistics industry has undergone significant transformation over the past two decades. "
        "Container shipping volumes have grown at an average rate of 4.7% annually, driven largely by the expansion "
        "of e-commerce and the diversification of global supply chains. Port infrastructure investments worldwide "
        "have totaled hundreds of billions of dollars, with major hubs in Singapore, Rotterdam, and Shanghai "
        "expanding their capacity substantially. The adoption of automated cranes, AI-driven routing software, "
        "and real-time vessel tracking has reduced average port turnaround times from 72 hours to under 18 hours "
        "in the most advanced facilities. Nevertheless, the industry faces persistent challenges: fuel price "
        "volatility, geopolitical disruptions in key waterways, and the mounting pressure to decarbonize. "
        "The International Maritime Organization has mandated that carbon emissions be reduced by 50% by 2050 "
        "relative to 2008 levels. Liquid natural gas, ammonia, and hydrogen are all being explored as "
        "alternative fuels, but each carries infrastructure requirements that will take decades to build. "
        "Meanwhile, inland logistics — trucking, rail, and last-mile delivery — are experiencing their own "
        "revolution, with electric vehicles and autonomous delivery drones beginning to appear in pilot programs "
        "across North America and Europe. Supply chain resilience, once an afterthought, is now a boardroom priority."
    ),
    (
        "Advances in material science have quietly reshaped nearly every industrial sector over the last thirty years. "
        "The development of carbon fiber composites, once the exclusive domain of aerospace engineering, has spread "
        "to automotive manufacturing, sporting goods, construction, and medical devices. Carbon fiber's exceptional "
        "strength-to-weight ratio — roughly five times stronger than steel at a fraction of the mass — enables "
        "designs that were previously impossible. However, the recycling challenge remains: most carbon fiber "
        "composite parts end their life in landfill rather than being reclaimed. Researchers at several universities "
        "are working on pyrolysis and solvolysis processes that could recover 90% of fiber value, but commercial "
        "adoption remains limited. Separately, metamaterials — engineered structures with properties not found in "
        "nature — are enabling breakthroughs in acoustic isolation, electromagnetic shielding, and even theoretical "
        "'cloaking' applications for radar. Graphene, despite enormous early promise, has struggled to achieve "
        "cost-effective mass production, though niche applications in battery electrodes and filtration membranes "
        "are gaining traction. High-entropy alloys, which blend five or more metallic elements in near-equal "
        "proportions, show exceptional hardness and corrosion resistance and are appearing in cutting tools and "
        "turbine components. The frontier of self-healing materials — polymers that repair microscopic cracks "
        "autonomously — may redefine product lifespans in the next decade."
    ),
    (
        "Urban planning philosophy has cycled through several dominant paradigms since the industrial revolution. "
        "The 19th century emphasis on density and transit-oriented design gave way to 20th century suburban sprawl, "
        "enabled first by streetcars and then by the mass automobile. The consequences of sprawl — traffic "
        "congestion, air pollution, loss of agricultural land, and social fragmentation — eventually prompted "
        "a new urbanist counter-movement beginning in the 1980s. New Urbanism advocates for walkable "
        "neighborhoods, mixed-use zoning, and the prioritization of public transit over private vehicles. "
        "Today, the 15-minute city concept, popularized by Paris Mayor Anne Hidalgo and urban theorist "
        "Carlos Moreno, proposes that all essential services — work, shopping, healthcare, education, and "
        "recreation — should be accessible within a 15-minute walk or bike ride from any home. The concept "
        "gained global attention during the COVID-19 pandemic when the value of local amenities became "
        "acutely apparent. Critics argue that the 15-minute city model is feasible only in dense urban cores "
        "and risks exacerbating inequality if wealthy areas become self-contained while poorer districts are "
        "left underfunded. Smart city technology, including sensor networks, adaptive traffic lights, and "
        "predictive maintenance for infrastructure, promises to improve efficiency but raises surveillance concerns."
    ),
    (
        "The economics of renewable energy have changed more rapidly than almost any expert predicted a decade ago. "
        "The levelized cost of electricity from utility-scale solar photovoltaic plants has fallen by over 90% "
        "since 2010, and onshore wind costs have declined by more than 70% in the same period. These dramatic "
        "reductions are attributable to manufacturing scale, supply chain optimization, improved panel efficiency, "
        "and competitive procurement mechanisms. In many regions, new renewable capacity is now cheaper to "
        "build and operate than continuing to run existing coal or gas plants. Despite this progress, the "
        "intermittency challenge — solar produces power only when the sun shines, wind only when the wind blows — "
        "requires either large-scale energy storage, long-distance transmission, or dispatchable backup capacity. "
        "Lithium-ion battery storage costs have fallen sharply too, but grid-scale storage capable of managing "
        "multi-day or seasonal variability remains expensive. Pumped hydro remains the dominant storage "
        "technology globally, representing over 90% of installed capacity. Emerging alternatives including "
        "iron-air batteries, gravity storage, green hydrogen electrolysis, and compressed air energy storage "
        "are progressing through demonstration phases. The transition away from fossil fuels, while clearly "
        "underway, will require massive investment in grid modernization, transmission infrastructure, and "
        "policy frameworks that coordinate across national and subnational boundaries."
    ),
    (
        "The history of cartography reflects the history of human ambition and the limits of knowledge. "
        "Early maps were less geographic tools than cosmological statements — the Hereford Mappa Mundi places "
        "Jerusalem at the center of the world, with Paradise at the top, reflecting medieval Christian "
        "cosmology rather than spatial accuracy. Portuguese and Spanish navigators of the 15th century drove "
        "a revolution in map accuracy, as the profit motive of trade and colonization demanded reliable charts. "
        "Gerardus Mercator's 1569 projection solved the critical navigation problem of representing curved "
        "Earth on flat paper in a way that preserved compass bearings, but at the cost of distorting areas — "
        "a distortion that famously makes Greenland appear larger than Africa. The 20th century saw cartography "
        "transformed by aerial photography and satellite imagery, and the 21st century brought GPS, digital "
        "mapping, and crowd-sourced platforms like OpenStreetMap. Today, geospatial data underpins navigation, "
        "logistics, climate modeling, disaster response, agriculture, and urban planning. Yet the politics of "
        "maps remain: disputed borders appear differently on maps published in different countries, and "
        "decisions about what to include or exclude are never neutral. The rise of 3D mapping, augmented "
        "reality overlays, and real-time data layers is creating a new cartographic era."
    ),
    (
        "Cognitive science has increasingly challenged the intuitive model of memory as a reliable recording "
        "device. Research by Elizabeth Loftus and colleagues demonstrated as early as the 1970s that "
        "eyewitness memory is highly susceptible to post-event suggestion, leading to the formation of "
        "false memories that feel indistinguishable from genuine recollections. The 'misinformation effect' "
        "— where exposure to incorrect information after an event contaminates the original memory — has "
        "profound implications for the legal system, where eyewitness testimony has historically carried "
        "enormous weight. DNA exonerations through the Innocence Project have confirmed that faulty eyewitness "
        "identification was a contributing factor in roughly 70% of wrongful conviction cases. Meanwhile, "
        "working memory research shows that humans can hold only approximately four chunks of information "
        "in active consciousness at once, far fewer than the 'seven plus or minus two' figure that appeared "
        "in George Miller's classic 1956 paper. Long-term memory consolidation occurs primarily during sleep, "
        "particularly during slow-wave and REM phases, which explains why sleep deprivation so severely "
        "impairs learning retention. The phenomenon of the 'spacing effect' — whereby distributed practice "
        "over time produces far better retention than massed practice (cramming) — is one of the most "
        "robust findings in educational psychology, yet it remains underutilized in school curricula globally."
    ),
    (
        "The development of quantum computing has shifted from a purely theoretical exercise to an engineering "
        "race among governments, technology companies, and research institutions worldwide. Unlike classical "
        "bits that hold a value of 0 or 1, quantum bits (qubits) exploit superposition to exist in both "
        "states simultaneously, and entanglement to correlate qubit states across physical distances. These "
        "properties theoretically allow quantum computers to solve certain classes of problems exponentially "
        "faster than any classical machine. The most widely anticipated application is the factoring of large "
        "integers, which would undermine current RSA cryptographic infrastructure — motivating urgent research "
        "into post-quantum cryptography. Other promising applications include drug discovery through molecular "
        "simulation, optimization problems in logistics and finance, and machine learning acceleration. "
        "However, today's 'noisy intermediate-scale quantum' (NISQ) devices are fragile: qubits decohere "
        "in microseconds, error rates are high, and most algorithms require error correction that demands "
        "hundreds or thousands of physical qubits per logical qubit. IBM, Google, IonQ, and others are "
        "pursuing different qubit architectures — superconducting, trapped ion, photonic, topological — each "
        "with distinct tradeoff profiles. Achieving fault-tolerant quantum computing at scale is now "
        "estimated to require a decade or more of additional engineering progress."
    ),
    (
        "Ocean acidification represents one of the less-discussed but potentially catastrophic consequences "
        "of rising atmospheric CO2 concentrations. When CO2 dissolves in seawater it forms carbonic acid, "
        "lowering ocean pH. Since pre-industrial times, average ocean pH has dropped from approximately 8.2 "
        "to 8.1 — a 26% increase in hydrogen ion concentration due to the logarithmic nature of the pH scale. "
        "Projections suggest pH could reach 7.95 by 2100 under high-emission scenarios. This seemingly small "
        "change has large biological consequences: many marine organisms — corals, mollusks, echinoderms, "
        "some plankton species — build shells and skeletons from calcium carbonate, which dissolves in more "
        "acidic water. Coral bleaching events, driven by warming combined with acidification, have devastated "
        "reefs that support roughly 25% of all marine species despite covering less than 1% of the ocean "
        "floor. The economic consequences extend to fisheries, tourism, and coastal protection — tropical "
        "reefs provide storm buffering estimated at $9 billion annually. Deep-sea ecosystems, once considered "
        "refuges, are now showing acidification effects at depths below 200 meters. Unlike temperature, "
        "which can theoretically be addressed through solar geoengineering, ocean acidification has no "
        "identified technical solution short of reducing CO2 emissions at the source."
    ),
]

DISTRACTORS = [
    "Analysts estimated the value at approximately 42 units.",
    "The committee recorded a preliminary count of 17 instances.",
    "Early surveys suggested the figure was closer to 200.",
    "Standard protocol requires a minimum threshold of 75.",
    "The reference baseline was established at 1,000 units.",
    "Initial projections placed the number at roughly 350.",
    "Historical records indicate a previous peak of 88.",
    "Preliminary data suggested a range of 30-40.",
    "The average across comparable cases was 56.",
    "Experts initially estimated the total at 500.",
]

def make_item(seed: int, needle_fact: str, needle_answer: str, question: str, n_filler: int = 6):
    rng = random.Random(seed)
    filler_pool = FILLERS[:]
    rng.shuffle(filler_pool)
    selected = filler_pool[:n_filler]
    distractor_pool = DISTRACTORS[:]
    rng.shuffle(distractor_pool)
    # Insert distractors into fillers
    enhanced = []
    for i, f in enumerate(selected):
        d = distractor_pool[i % len(distractor_pool)]
        enhanced.append(f + " " + d)
    # Place needle at a random middle position
    insert_pos = rng.randint(1, max(1, n_filler - 1))
    paragraphs = enhanced[:insert_pos] + [needle_fact] + enhanced[insert_pos:]
    context = "\n\n".join(f"[Section {i+1}] {p}" for i, p in enumerate(paragraphs))
    return {"context": context, "question": question, "answer": needle_answer}

DATASET = [
    make_item(
        seed=1,
        needle_fact="CRITICAL FACT: The vault combination is 7-3-19.",
        needle_answer="7-3-19",
        question="What is the vault combination mentioned in the document?",
        n_filler=7,
    ),
    make_item(
        seed=2,
        needle_fact="NOTE: The experiment was conducted on March 4th, 2019.",
        needle_answer="march 4 2019",
        question="On what date was the experiment conducted?",
        n_filler=6,
    ),
    make_item(
        seed=3,
        needle_fact="EMBEDDED FACT: The witness's phone number was 555-0147.",
        needle_answer="555-0147",
        question="What was the witness's phone number?",
        n_filler=8,
    ),
    make_item(
        seed=4,
        needle_fact="HIDDEN DETAIL: The password for the archive is COBALT-7.",
        needle_answer="cobalt-7",
        question="What is the password for the archive?",
        n_filler=5,
    ),
    make_item(
        seed=5,
        needle_fact="RECORD: Agent Kowalski's badge number is 4471.",
        needle_answer="4471",
        question="What is Agent Kowalski's badge number?",
        n_filler=7,
    ),
    make_item(
        seed=6,
        needle_fact="SPECIFICATION: The target coordinates are 48.2°N, 16.3°E.",
        needle_answer="48.2n 16.3e",
        question="What are the target coordinates listed in the document?",
        n_filler=8,
    ),
    make_item(
        seed=7,
        needle_fact="BURIED NOTE: The correct frequency is 137.5 MHz.",
        needle_answer="137.5",
        question="What frequency is specified in the document?",
        n_filler=6,
    ),
    make_item(
        seed=8,
        needle_fact="INTERNAL MEMO: The acquisition budget cap is $2.3 million.",
        needle_answer="2.3 million",
        question="What is the acquisition budget cap mentioned in the document?",
        n_filler=7,
    ),
    make_item(
        seed=9,
        needle_fact="ALERT: The critical threshold is 0.004 parts per million.",
        needle_answer="0.004",
        question="What is the critical threshold value mentioned?",
        n_filler=8,
    ),
    make_item(
        seed=10,
        needle_fact="MINUTES: The vote passed with exactly 11 in favor.",
        needle_answer="11",
        question="How many voted in favor according to the document?",
        n_filler=5,
    ),
    make_item(
        seed=11,
        needle_fact="FOOTNOTE: The patient's blood type on file is AB negative.",
        needle_answer="ab negative",
        question="What is the patient's blood type?",
        n_filler=7,
    ),
    make_item(
        seed=12,
        needle_fact="ANNOTATION: The treaty was signed on November 11, 1918.",
        needle_answer="november 11 1918",
        question="When was the treaty signed?",
        n_filler=6,
    ),
    make_item(
        seed=13,
        needle_fact="CORRECTION: The actual melting point of the alloy is 843°C.",
        needle_answer="843",
        question="What is the melting point of the alloy?",
        n_filler=8,
    ),
    make_item(
        seed=14,
        needle_fact="INSET: The station's call sign is WKRX-FM.",
        needle_answer="wkrx-fm",
        question="What is the station's call sign?",
        n_filler=7,
    ),
    make_item(
        seed=15,
        needle_fact="TECHNICAL NOTE: The fuel mixture ratio is 3 parts hydrogen to 1 part oxygen.",
        needle_answer="3 to 1",
        question="What is the fuel mixture ratio given in the document?",
        n_filler=6,
    ),
    make_item(
        seed=16,
        needle_fact="PRIVATE: The safety deposit box key number is 0892.",
        needle_answer="0892",
        question="What is the safety deposit box key number?",
        n_filler=8,
    ),
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
    if all(p in r_clean for p in a_parts):
        return True
    try:
        if abs(float(re.sub(r'[^0-9.\-]', '', r)) - float(re.sub(r'[^0-9.\-]', '', a))) < 0.01:
            return True
    except Exception:
        pass
    return False


@kbench.task(name="sustained_attention")
def task_func(llm, context: str, question: str, answer: str) -> float:
    prompt = (
        f"Read the following long document carefully. A single critical piece of information is "
        f"hidden somewhere within it. Find it and answer the question precisely.\n\n"
        f"DOCUMENT:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        f"Answer with ONLY the specific value or phrase requested. No explanation."
    )
    response = llm(prompt)
    correct = check_answer(response, answer)
    kbench.assertions.assert_true(correct, expectation=f"Expected '{answer}', got '{response}'")
    return 1.0 if correct else 0.0


df = pd.DataFrame(DATASET)
task_func.evaluate(llm=[kbench.llm], evaluation_data=df)
