import os
import json

class ResultWriter:

    def __init__(self):
        self.results = []

    def add(self, item, result):

        steps = []

        num_steps = len(result.qa_pairs)

        for i in range(num_steps):

            qa = result.qa_pairs[i]

            # tenta pegar evidence por step (ideal)
            if isinstance(result.evidence, list) and i < len(result.evidence):
                evidence = result.evidence[i]
            else:
                evidence = result.evidence  # fallback (não ideal)

            # tenta pegar stance por step
            if isinstance(result.stances, list) and i < len(result.stances):
                stance = result.stances[i]
            else:
                stance = None

            step = {
                "question": qa.get("question"),
                "answer": qa.get("answer"),
                "evidence": evidence,
                "stance": stance
            }

            steps.append(step)

        self.results.append({
            "claim": item.get("claim"),
            "prediction": result.verdict,
            "gold_label": item.get("label"),

            "pipeline": {
                "steps": steps,
                "final_verdict": result.verdict
            }
        })

    def save(self, path):

        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)