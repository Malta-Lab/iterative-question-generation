class LLMStanceDetector:

    def __init__(self, llm):
        self.llm = llm

    def classify(self, claim, evidence_text, claim_date):

        prompt = f"""
        Task: Determine the relationship between the claim and the evidence.

        Claim:
        {claim}

        Claim date:
        {claim_date}

        Evidence:
        {evidence_text}

        Labels:

        SUPPORT:
        The evidence clearly confirms the claim.

        REFUTE:
        The evidence clearly contradicts the claim OR explicitly states that it is false, fake, a hoax, or did not happen.

        NOT ENOUGH EVIDENCE:
        The evidence does not provide enough information to verify the claim.

        CONFLICTING EVIDENCE/CHERRYPICKING:
        The evidence contains both supporting and contradicting information.

        Decision rules:
        - If the evidence says "fake", "false", "hoax", or "did not happen" → REFUTE
        - If the evidence denies the claim → REFUTE
        - Only choose NOT ENOUGH EVIDENCE if there is truly no clear conclusion
        - Do NOT be overly cautious

        Output:
        Respond with ONLY one label.
        """

        response = self.llm.generate(prompt).strip().upper()

        if "SUPPORT" in response:
            return "SUPPORT"
        elif "REFUTE" in response:
            return "REFUTE"
        elif "CONFLICTING" in response:
            return "CONFLICTING EVIDENCE/CHERRYPICKING"
        elif "NOT ENOUGH" in response:
            return "NOT ENOUGH EVIDENCE"
        else:
            return "..."

    def run(self, context):

        stances = []

        for e in context.evidence:
            text = e["text"]

            stance = self.classify(context.claim, text, context.claim_date)

            stances.append({
                "text": text,
                "label": stance.lower(),
                "bm25_score": e.get("bm25_score"),
                "rerank_score": e.get("rerank_score")
            })

        context.stances = stances

        return context