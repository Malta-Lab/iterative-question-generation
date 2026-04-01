import os
import json


class ResultWriter:

    def __init__(self, path):
        self.path = path

        os.makedirs(os.path.dirname(path), exist_ok=True)

        if not os.path.exists(self.path):
            open(self.path, "w", encoding="utf-8").close()

    def _build_entry(self, item, result, claim_id):

        steps = []

        qa_pairs = result.qa_pairs or []
        stances = result.stances or []
        evidences = result.evidence or []

        if not isinstance(evidences, list):
            evidences = [evidences]

        for i, qa in enumerate(qa_pairs):

            stance = None
            if isinstance(stances, list) and i < len(stances):
                stance = stances[i]

            processed_evidence = []

            for ev in evidences:
                if isinstance(ev, dict):
                    processed_evidence.append({
                        "text": ev.get("text"),
                        "url": ev.get("url"),
                        "rerank_score": ev.get("rerank_score"),
                    })
                else:
                    processed_evidence.append({
                        "text": str(ev),
                        "url": None,
                    })

            steps.append({
                "question": qa.get("question"),
                "answer": qa.get("answer"),
                "evidence": processed_evidence
            })

        return {
            "claim_id": claim_id,  # 🔥 ESSENCIAL
            "claim": item.get("claim"),
            "claim_date": item.get("claim_date"),
            "prediction": result.verdict,
            "gold_label": item.get("label"),
            "speaker": item.get("speaker"),
            "pipeline": {
                "steps": steps
            }
        }

    def append(self, item, result, claim_id):

        entry = self._build_entry(item, result, claim_id)

        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
    