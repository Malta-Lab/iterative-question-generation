import os
import json


class ResultWriter:

    def __init__(self):
        self.results = []

    def add(self, item, result):
        steps = []

        qa_pairs = result.qa_pairs or []
        stances = result.stances or []
        evidences = result.evidence or []

        # 🔥 garante lista de evidências
        if not isinstance(evidences, list):
            evidences = [evidences]

        for i, qa in enumerate(qa_pairs):

            # pega stance correspondente
            stance = None
            if isinstance(stances, list) and i < len(stances):
                stance = stances[i]

            processed_evidence = []

            for ev in evidences:

                if isinstance(ev, dict):
                    processed_evidence.append({
                        "text": ev.get("text"),
                        "bm25_score": ev.get("bm25_score"),
                        "rerank_score": ev.get("rerank_score"),
                        "label": stance.get("label") if isinstance(stance, dict) else stance
                    })
                else:
                    processed_evidence.append({
                        "text": str(ev),
                        "label": stance
                    })

            step = {
                "question": qa.get("question"),
                "answer": qa.get("answer"),
                "evidence": processed_evidence
            }

            steps.append(step)

        self.results.append({
            "claim": item.get("claim"),
            "prediction": result.verdict,
            "gold_label": item.get("label"),
            "speaker": item.get("speaker"),
            "pipeline": {
                "steps": steps,
                "final_verdict": result.verdict
            }
        })

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)