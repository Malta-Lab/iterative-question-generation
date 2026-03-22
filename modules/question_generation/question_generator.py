from config import Config


class QuestionGenerator:

    def __init__(self, llm):
        self.llm = llm

    def run(self, context):

        # 🛑 se atingiu limite, não gera mais
        if len(context.questions) >= Config.MAX_QUESTIONS:
            return context

        previous_questions = context.questions

        # 🧠 usar evidência e stances como contexto
        evidence_snippets = [
            e["text"][:200] for e in context.evidence[:5]
        ] if hasattr(context, "evidence") else []

        stance_summary = [
            f'{s["label"]}: {s["text"][:100]}'
            for s in context.stances[:5]
        ] if hasattr(context, "stances") else []

        prompt = f"""
Task: Generate ONE new question to help verify the claim.

Claim:
{context.claim}

Claim date:
{context.claim_date}

Speaker:
{context.speaker}

---

Previous questions:
{previous_questions}

---

Known evidence (partial):
{evidence_snippets}

---

Current stance signals:
{stance_summary}

---

Instructions:
- Generate ONLY ONE question
- Do NOT repeat previous questions
- Focus on missing or uncertain aspects
- Prefer specific, verifiable facts
- If claim involves time, check chronology
- If claim involves attribution, verify speaker/source
- If claim involves numbers, verify quantities

Output:
Provide only the question.
"""

        question = self.llm.generate(prompt).strip()

        # 🆕 evitar duplicação simples
        if question and question not in context.questions:
            context.questions.append(question)

        return context