"""
Generate .ipynb notebooks from .py source files.
Usage: python generate_notebooks.py
"""
import json
import os

SRC_DIR = os.path.join(os.path.dirname(__file__), "src")
OUT_DIR = os.path.dirname(__file__)

TASKS = [
    ("task1_selective_attention_v5.py", "task1_selective_attention_v5.ipynb", "selective_attention_v2"),
    ("task2_attention_shifting_v4.py",  "task2_attention_shifting_v4.ipynb",  "attention_shifting"),
    ("task3_sustained_attention_v5.py", "task3_sustained_attention_v5.ipynb", "sustained_attention"),
    ("task4_inattentional_blindness_v4.py", "task4_inattentional_blindness_v4.ipynb", "inattentional_blindness"),
    ("task5_saliency_awareness_v2.py",  "task5_saliency_awareness_v2.ipynb",  "saliency_awareness"),
]

NB_TEMPLATE = {
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

for src_name, out_name, task_name in TASKS:
    src_path = os.path.join(SRC_DIR, src_name)
    out_path = os.path.join(OUT_DIR, out_name)

    with open(src_path, "r", encoding="utf-8") as f:
        code = f.read()

    nb = dict(NB_TEMPLATE)
    nb["cells"] = [
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [code]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [f"%choose {task_name}"]
        }
    ]

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    size = os.path.getsize(out_path)
    print(f"  Written: {out_name}  ({size:,} bytes)")

print("Done.")
