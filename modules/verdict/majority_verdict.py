from collections import Counter
from .base_verdict import BaseVerdict


class MajorityVerdict(BaseVerdict):

    def run(self, context):

        stances = []
        for s in context.stances:
            if isinstance(s, tuple):
                # assume (evidence, label)
                stances.append(s[1].lower())
            elif isinstance(s, dict):
                stances.append(s["label"].lower())
            else:
                raise ValueError(f"Formato inesperado: {s}")

        if not stances:
            context.verdict = "not enough evidence"
            return context

        counts = Counter(stances)

        context.verdict = counts.most_common(1)[0][0]

        return context