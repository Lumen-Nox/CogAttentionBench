"""
Dataset Generator: Sustained Attention — Vigilance Over Growing Context
Tests whether model accuracy degrades as irrelevant context increases
while task difficulty stays constant.

Cognitive basis:
- Continuous Performance Task (Rosvold et al., 1956)
- Vigilance decrement (Warm et al., 2008)
- Mackworth Clock Test (Mackworth, 1948)

Each item has: text, question, answer, context_paragraphs, task_type
"""

import json
import random

random.seed(42)

# Filler paragraphs - plausible, topical, but irrelevant
FILLER_PARAGRAPHS = [
    "The history of mathematics stretches back thousands of years. Ancient civilizations developed numerical systems for trade and astronomy. The Babylonians used a base-60 system that still influences how we measure time today. Egyptian mathematicians developed practical geometry for land surveying after annual Nile floods.",
    "In recent years, advances in computing have transformed how we process information. Cloud computing and distributed systems allow for unprecedented scale of data analysis. Many industries have been revolutionized by these technologies. The shift to remote work has further accelerated digital transformation across sectors.",
    "The natural world contains countless examples of mathematical patterns. The Fibonacci sequence appears in sunflower seeds, pinecone spirals, and nautilus shells. These patterns emerge from simple rules governing growth. Fractal geometry reveals self-similar structures at every scale in nature.",
    "Climate science relies heavily on mathematical models to predict future conditions. These models incorporate data from ocean temperatures, atmospheric composition, and solar radiation. The complexity of these systems makes precise prediction challenging. Ensemble methods help quantify uncertainty in projections.",
    "Music theory is deeply connected to mathematics. Harmonic relationships between notes can be expressed as simple ratios. The octave represents a 2:1 frequency ratio. Pythagoras discovered that consonant musical intervals correspond to simple numerical relationships.",
    "Cryptography relies on number theory to secure communications. Prime factorization, modular arithmetic, and elliptic curves form the basis of modern encryption. These mathematical tools protect billions of online transactions daily. Quantum computing may eventually challenge current cryptographic methods.",
    "Game theory studies strategic interactions between rational agents. The prisoner's dilemma illustrates how individual rationality can lead to collectively suboptimal outcomes. Nash equilibrium provides a framework for analyzing such situations. Applications range from economics to evolutionary biology.",
    "Statistical mechanics bridges the microscopic world of atoms with macroscopic thermodynamic properties. The Boltzmann distribution describes how energy is distributed among particles. This connection between scales remains one of physics' greatest achievements. Modern extensions include non-equilibrium statistical mechanics.",
    "Topology studies properties preserved under continuous deformation. A coffee mug and a donut are topologically equivalent because each has exactly one hole. This branch of mathematics has found applications in data analysis, robotics, and quantum computing.",
    "Linear algebra provides the mathematical framework for machine learning. Matrices represent transformations and data, while eigenvalues reveal fundamental properties. Singular value decomposition enables dimensionality reduction. These tools are essential for modern AI systems.",
    "Probability theory formalizes reasoning under uncertainty. Bayes' theorem shows how to update beliefs with new evidence. The central limit theorem explains why normal distributions appear throughout nature. Monte Carlo methods use random sampling to approximate complex calculations.",
    "Differential equations model how quantities change over time. Newton's laws of motion, Maxwell's equations of electromagnetism, and Schrödinger's equation in quantum mechanics are all differential equations. Numerical methods allow computers to approximate solutions when analytical solutions don't exist.",
    "Graph theory studies networks of connections. Social networks, transportation systems, and the internet can all be modeled as graphs. Euler's solution to the Königsberg bridge problem in 1736 is often considered the birth of graph theory.",
    "Information theory quantifies the fundamental limits of data compression and reliable communication. Shannon entropy measures the information content of a message. These concepts underpin modern telecommunications, data storage, and machine learning.",
    "Chaos theory reveals how small changes in initial conditions can lead to vastly different outcomes. The butterfly effect illustrates this sensitive dependence. Despite apparent randomness, chaotic systems follow deterministic rules. Weather prediction is fundamentally limited by this sensitivity.",
]


def generate_arithmetic_items(n_per_level=5, max_context_level=10):
    """Generate arithmetic tasks with growing context. Difficulty is constant."""
    items = []
    
    for level in range(max_context_level + 1):
        for i in range(n_per_level):
            # Same difficulty: two-digit addition
            a = random.randint(10, 99)
            b = random.randint(10, 99)
            correct = a + b
            
            if level == 0:
                text = f"What is {a} + {b}?"
            else:
                # Insert filler paragraphs before and after
                n_before = level // 2 + 1
                n_after = level - n_before + 1
                
                filler_before = "\n\n".join(random.sample(FILLER_PARAGRAPHS, min(n_before, len(FILLER_PARAGRAPHS))))
                filler_after = "\n\n".join(random.sample(FILLER_PARAGRAPHS, min(n_after, len(FILLER_PARAGRAPHS))))
                
                text = f"""{filler_before}

[TASK] Calculate: What is {a} + {b}? [END TASK]

{filler_after}"""
            
            items.append({
                "text": text,
                "question": f"What is {a} + {b}? Answer with just the number.",
                "answer": str(correct),
                "context_paragraphs": level,
                "task_type": "arithmetic",
            })
    
    return items


def generate_factual_recall_items(n_per_level=3, max_context_level=10):
    """Generate factual recall tasks with growing context."""
    items = []
    
    facts = [
        ("What is the capital of Japan?", "Tokyo"),
        ("What is the chemical symbol for iron?", "Fe"),
        ("How many sides does a pentagon have?", "5"),
        ("What year did World War I begin?", "1914"),
        ("What is the largest mammal?", "blue whale"),
        ("What is the square root of 64?", "8"),
        ("What planet is known as the Red Planet?", "Mars"),
        ("What is the chemical formula for table salt?", "NaCl"),
        ("How many meters in a kilometer?", "1000"),
        ("What is the atomic number of hydrogen?", "1"),
    ]
    
    for level in range(max_context_level + 1):
        selected_facts = random.sample(facts, min(n_per_level, len(facts)))
        for fact_q, fact_a in selected_facts:
            if level == 0:
                text = fact_q
            else:
                n_before = level // 2 + 1
                n_after = level - n_before + 1
                
                filler_before = "\n\n".join(random.sample(FILLER_PARAGRAPHS, min(n_before, len(FILLER_PARAGRAPHS))))
                filler_after = "\n\n".join(random.sample(FILLER_PARAGRAPHS, min(n_after, len(FILLER_PARAGRAPHS))))
                
                text = f"""{filler_before}

[QUESTION] {fact_q} [END QUESTION]

{filler_after}"""
            
            items.append({
                "text": text,
                "question": fact_q + " Answer briefly.",
                "answer": fact_a,
                "context_paragraphs": level,
                "task_type": "factual_recall",
            })
    
    return items


def generate_all():
    """Generate the complete sustained attention dataset."""
    all_items = []
    all_items.extend(generate_arithmetic_items(n_per_level=5, max_context_level=10))
    all_items.extend(generate_factual_recall_items(n_per_level=3, max_context_level=10))
    
    return all_items


if __name__ == "__main__":
    items = generate_all()
    print(f"Generated {len(items)} sustained attention items")
    print(f"  Arithmetic: {sum(1 for i in items if i['task_type'] == 'arithmetic')}")
    print(f"  Factual recall: {sum(1 for i in items if i['task_type'] == 'factual_recall')}")
    
    # Show context level distribution
    from collections import Counter
    levels = Counter(i['context_paragraphs'] for i in items)
    for level in sorted(levels):
        print(f"  Context level {level}: {levels[level]} items")
    
    with open("dataset_sustained_attention.json", "w") as f:
        json.dump(items, f, indent=2)
    print("Saved to dataset_sustained_attention.json")
