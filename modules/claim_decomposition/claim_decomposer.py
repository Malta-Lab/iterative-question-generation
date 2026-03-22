class ClaimDecomposer:

    def __init__(self, llm, max_subclaims=3):
        self.llm = llm
        self.max_subclaims = max_subclaims

    def run(self, context):

        prompt = f"""
Task: Break the claim into atomic, verifiable subclaims.

Claim:
{context.claim}

Instructions:
- Decompose into independent factual statements
- Each subclaim must be verifiable on its own
- Keep them concise
- Avoid redundancy

Output:
Provide a list of subclaims (max {self.max_subclaims})
"""

        output = self.llm.generate(prompt)

        subclaims = []
        for line in output.split("\n"):
            line = line.strip()
            if not line:
                continue
            subclaims.append(line)

        context.subclaims = subclaims[:self.max_subclaims]

        return context