from collections import defaultdict
from .base_verdict import BaseVerdict


class WeightedVerdict(BaseVerdict):

    def run(self, context):

        scores = defaultdict(int)

        for stance in context.stances:

            label = stance["label"].lower()

            scores[label] += 1


        if not scores:
            context.verdict = "not enough evidence"
            return context


        verdict = max(scores, key=scores.get)

        context.verdict = verdict

        return context