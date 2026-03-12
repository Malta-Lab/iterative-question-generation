from .base_verdict import BaseVerdict


class LLMVerdict(BaseVerdict):

    def __init__(self, llm):

        self.llm = llm


    def run(self, context):

        evidence = "\n\n".join(context.evidence)


        prompt = f"""
        Claim:
        {context.claim}

        Evidence:
        {evidence}

        Based on the evidence, classify the claim as one of:

        SUPPORTED
        REFUTED
        NOT ENOUGH EVIDENCE
        CONFLICTING

        Answer with only the label.
        """


        response = self.llm.generate(prompt)

        context.verdict = response.strip().lower()

        return context