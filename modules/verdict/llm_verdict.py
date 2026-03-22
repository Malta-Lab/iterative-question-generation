from .base_verdict import BaseVerdict


class LLMVerdict(BaseVerdict):

    SUPPORTED = "SUPPORTED"
    REFUTED = "REFUTED"
    NEE = "NOT ENOUGH EVIDENCE"
    CONFLICT = "CONFLICTING"

    def __init__(self, llm):
        self.llm = llm

    def build_prompt(self, context):

        support_text = "\n\n".join(
            e["text"] if isinstance(e, dict) else str(e)
            for e in context.support_evidence[:5]
        )

        refute_text = "\n\n".join(
            e["text"] if isinstance(e, dict) else str(e)
            for e in context.refute_evidence[:5]
        )

        return f"""
Task: Determine the final verdict of a claim based on evidence.

Claim:
{context.claim}

---

Supporting evidence:
{support_text}

---

Refuting evidence:
{refute_text}

---

Instructions:

- Compare supporting and refuting evidence carefully
- If strong contradiction exists → REFUTED
- If evidence clearly supports → SUPPORTED
- If evidence is weak or insufficient → NOT ENOUGH EVIDENCE
- If both strong support and refutation exist → CONFLICTING

Be conservative:
- Prefer REFUTED over SUPPORTED when conflict exists

---

Output:
Return ONLY one label:

SUPPORTED
REFUTED
NOT ENOUGH EVIDENCE
CONFLICTING
"""

    def normalize(self, response: str):

        if not response:
            return self.NEE

        r = response.strip().upper()

        if "REFUTE" in r or "FALSE" in r:
            return self.REFUTED

        if "SUPPORT" in r:
            return self.SUPPORTED

        if "CONFLICT" in r:
            return self.CONFLICT

        if "NOT ENOUGH" in r or "INSUFFICIENT" in r:
            return self.NEE

        return self.NEE

    def run(self, context):

        prompt = self.build_prompt(context)

        response = self.llm.generate(prompt)

        context.verdict = self.normalize(response)

        return context