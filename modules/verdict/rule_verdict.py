from collections import Counter
from .base_verdict import BaseVerdict


class RuleVerdict(BaseVerdict):

    def run(self, context):

        stances = []

        for s in context.stances:
            if isinstance(s, dict):
                label = s["label"]
            elif isinstance(s, tuple):
                label = s[1]
            else:
                continue

            label = label.upper()

            # 🔧 normalização
            if label in ["SUPPORT", "SUPPORTED"]:
                label = "SUPPORT"
            elif label in ["REFUTE", "REFUTED"]:
                label = "REFUTE"
            elif "NOT ENOUGH" in label:
                label = "NOT ENOUGH EVIDENCE"
            elif "CONFLICTING" in label:
                label = "CONFLICTING EVIDENCE/CHERRYPICKING"

            stances.append(label)

        if not stances:
            context.verdict = "NOT ENOUGH EVIDENCE"
            return context

        counts = Counter(stances)

        support = counts.get("SUPPORT", 0)
        refute = counts.get("REFUTE", 0)
        nee = counts.get("NOT ENOUGH EVIDENCE", 0)

        if support > refute and support > nee:
            verdict = "SUPPORT"

        elif refute > support and refute > nee:
            verdict = "REFUTE"

        elif support == 0 and refute == 0:
            verdict = "NOT ENOUGH EVIDENCE"

        else:
            verdict = "CONFLICTING EVIDENCE/CHERRYPICKING"

        context.verdict = verdict

        return context