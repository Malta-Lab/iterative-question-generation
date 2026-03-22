class CounterfactualGenerator:

    def __init__(self, llm):
        self.llm = llm

    def run(self, context):

        prompt = f"""
Task: Generate a counterfactual version of the claim.

Claim:
{context.claim}

Instructions:
- Negate the main assertion
- Keep entities and context the same
- Make it a realistic alternative

Output:
Provide a single counterfactual claim.
"""

        cf = self.llm.generate(prompt).strip()

        context.counterfactual = cf

        return context