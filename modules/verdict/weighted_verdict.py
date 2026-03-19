from collections import defaultdict
from .base_verdict import BaseVerdict


class WeightedVerdict(BaseVerdict):

    def run(self, context):

        scores = defaultdict(float)

        for stance in context.stances:

            label = stance["label"].upper()

            # 🔧 normalização
            if label in ["SUPPORT", "SUPPORTED"]:
                label = "SUPPORT"
            elif label in ["REFUTE", "REFUTED"]:
                label = "REFUTE"
            elif "NOT ENOUGH" in label:
                label = "NOT ENOUGH EVIDENCE"
            elif "CONFLICTING" in label:
                label = "CONFLICTING EVIDENCE/CHERRYPICKING"

            # 🔥 peso (prioriza reranker)
            weight = stance.get("rerank_score")

            if weight is None:
                weight = stance.get("bm25_score", 1.0)

            if weight is None:
                weight = 1.0

            # 🔥 opcional: penalizar NEI (recomendado)
            if label == "NOT ENOUGH EVIDENCE":
                weight *= 0.5

            scores[label] += weight

        if not scores:
            context.verdict = "NOT ENOUGH EVIDENCE"
            return context

        verdict = max(scores, key=scores.get)

        context.verdict = verdict

        return context