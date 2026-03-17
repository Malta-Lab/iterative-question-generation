class LLMStanceDetector:

    def __init__(self, llm):

        self.llm = llm

    def classify(self, claim, evidence):

        prompt = f"""
            You are a fact-checking assistant.

            Claim:
            {claim}

            Evidence:
            {evidence}

            Does the evidence SUPPORT, REFUTE, or is it IRRELEVANT to the claim?

            Respond with exactly one word:
            SUPPORT
            REFUTE
            IRRELEVANT
        """

        response = self.llm.generate(prompt).strip().upper()

        if "SUPPORT" in response:
            return "SUPPORT"

        if "REFUTE" in response:
            return "REFUTE"

        return "IRRELEVANT"

    def run(self, context):

        stances = []

        for evidence in context.evidence:

            stance = self.classify(context.claim, evidence)

            stances.append((evidence, stance))

        context.stances = stances

        return context