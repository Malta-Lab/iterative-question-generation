class QAGenerator:

    def __init__(self, llm, max_evidence_per_question=5):
        self.llm = llm
        self.max_evidence_per_question = max_evidence_per_question

    def run(self, context):

        # 🆕 inicializar sem sobrescrever
        if not hasattr(context, "qa_pairs") or context.qa_pairs is None:
            context.qa_pairs = []

        # 🆕 pegar apenas a última pergunta (iterativo)
        if not context.questions:
            return context

        question = context.questions[-1]

        # 🆕 selecionar evidências mais relevantes
        evidence_subset = context.evidence[:self.max_evidence_per_question]

        evidence_text = "\n\n".join(
            e["text"] if isinstance(e, dict) else e
            for e in evidence_subset
        )

        prompt = f"""
Task: Answer the question using only the provided evidence.

Question:
{question}

Claim:
{context.claim}

Claim date:
{context.claim_date}

Evidence:
{evidence_text}

Instructions:
- Use only the given evidence
- Do not use external knowledge
- If evidence is insufficient, say "INSUFFICIENT EVIDENCE"
- Be concise and factual

Output:
Provide a short answer.
"""

        answer = self.llm.generate(prompt).strip()

        # 🆕 salvar com rastreabilidade
        context.qa_pairs.append({
            "question": question,
            "answer": answer,
            "evidence_used": evidence_subset
        })

        return context