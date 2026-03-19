class LLMStanceDetector:

    def __init__(self, llm):
        self.llm = llm

    def classify(self, claim, evidence_text, claim_date, speaker):

        prompt = f"""
        Task: Determine the relationship between the claim and the evidence.

        Claim:
        {claim}

        Claim date:
        {claim_date}

        Speaker:
        {speaker}

        Evidence:
        {evidence_text}

        Labels:

        SUPPORTED:
        The evidence clearly supports the claim.

        REFUTED:
        The evidence clearly contradicts the claim OR shows that it is false, fake, misleading, or did not happen.

        NOT ENOUGH EVIDENCE:
        There is insufficient relevant evidence to make a decision.

        CONFLICTING EVIDENCE/CHERRYPICKING:
        There is strong evidence both supporting and refuting the claim, and it is genuinely unclear which is correct.

        Decision rules:

        - If the evidence explicitly says "false", "fake", "hoax", or "did not happen" → REFUTED
        - If the evidence contradicts the claim → REFUTED
        - If the evidence supports the claim → SUPPORTED

        CRITICAL RULES:

        - Do NOT use CONFLICTING as a default.
        - Use CONFLICTING ONLY if there is clear, strong evidence on BOTH sides.

        - If the claim is specific and strong (numbers, events, accusations) and there is NO solid supporting evidence → REFUTED

        - If evidence is weak, partial, or slightly unclear, choose the MOST LIKELY label (SUPPORTED or REFUTED)

        - Only choose NOT ENOUGH EVIDENCE when there is almost no relevant information.

        - You MUST make a decisive judgment. Do NOT be overly cautious.

        Output:
        Respond with ONLY one label.
        """

        response = self.llm.generate(prompt).strip().upper()

        if "SUPPORTED" in response:
            return "SUPPORTED"
        elif "REFUTED" in response:
            return "REFUTED"
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

            stance = self.classify(context.claim, text, context.claim_date, context.speaker)

            stances.append({
                "text": text,
                "label": stance.lower(),
                "bm25_score": e.get("bm25_score"),
                "rerank_score": e.get("rerank_score")
            })

        context.stances = stances

        return context