# backend/ml/generate_synthetic_questions.py
import pandas as pd
import random, os

random.seed(42)
os.makedirs(os.path.join(os.path.dirname(__file__), "..", "data"), exist_ok=True)
out_dir = os.path.join(os.path.dirname(__file__), "..", "data")

subjects = ["math","physics","chemistry","cs"]
topics = {
    "math": ["calculus integral problem", "algebra equation solving", "geometry theorems proof", "differential equations solve", "linear algebra matrix"],
    "physics": ["mechanics motion problem", "optics refraction question", "thermodynamics law application", "electric circuits problem", "quantum basics question"],
    "chemistry": ["organic reaction mechanism", "stoichiometry titration problem", "chemical bonding explanation", "periodic trends question", "acid base titration"],
    "cs": ["graph theory shortest path", "dynamic programming example", "sorting algorithm explanation", "recursion tree problem", "data structures stack queue"]
}

def synth_row(i):
    subj = random.choice(subjects)
    text = random.choice(topics[subj])
    extra = " ".join(random.choices(["explain", "solve step by step", "with example", "show steps", "detailed explanation"], k=random.randint(0,2)))
    full_text = (text + " " + extra).strip()
    length = max(3, len(full_text.split()))
    complexity = round(random.uniform(0.5, 3.5), 2)
    demand = round(random.uniform(0.5, 2.5), 2)
    # keep the same price heuristic you already use
    price = round(20 + (complexity * 50) + (length * 2) + (demand * 20) + random.uniform(-10, 10), 2)
    keywords = ",".join(list(dict.fromkeys([w.strip().lower() for w in full_text.split() if len(w) > 2])))
    return {"id": i+1, "text": full_text, "subject": subj, "length": length,
            "complexity": complexity, "demand": demand, "price": price, "keywords": keywords}

def generate(n=1200):
    rows = [synth_row(i) for i in range(n)]
    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "synthetic_questions.csv")
    xlsx_path = os.path.join(out_dir, "synthetic_questions.xlsx")
    df.to_csv(csv_path, index=False)
    df.to_excel(xlsx_path, index=False)
    print("Saved", csv_path)
    print("Saved", xlsx_path)
    return df

if __name__ == "__main__":
    df = generate(1200)
    print(df.head())
