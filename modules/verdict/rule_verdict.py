from collections import Counter
from .base_verdict import BaseVerdict


class RuleVerdict(BaseVerdict):

    def run(self, context):

        stances = [s["label"].lower() for s in context.stances]

        counts = Counter(stances)

        support = counts.get("supported", 0)
        refute = counts.get("refuted", 0)
        nee = counts.get("not enough evidence", 0)


        if support > refute and support > nee:
            verdict = "supported"

        elif refute > support and refute > nee:
            verdict = "refuted"

        elif support == 0 and refute == 0:
            verdict = "not enough evidence"

        else:
            verdict = "conflicting"


        context.verdict = verdict

        return context