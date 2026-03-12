from collections import Counter
from .base_verdict import BaseVerdict


class MajorityVerdict(BaseVerdict):

    def run(self, context):

        stances = [s["label"].lower() for s in context.stances]

        if not stances:
            context.verdict = "not enough evidence"
            return context

        counts = Counter(stances)

        context.verdict = counts.most_common(1)[0][0]

        return context