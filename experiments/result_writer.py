import os
import json

class ResultWriter:

    def __init__(self):
        self.results = []

    def add(self, item, result):

        qa_with_evidence = []

        for qa in result.qa_pairs:

            qa_with_evidence.append({
                "question": qa["question"],
                "answer": qa["answer"],
                "evidence": result.evidence  # 🔥 aqui pode melhorar depois
            })

        self.results.append({
            "claim": item["claim"],
            "prediction": result.verdict,
            "gold_label": item.get("label"),

            "pipeline": {
                "qa": qa_with_evidence,
                "stances": result.stances
            }
        })

    def save(self, path):

        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)