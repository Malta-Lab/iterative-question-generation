from typing import Tuple


class LLMStanceDetector:

    SUPPORTED = "SUPPORTED"
    REFUTED = "REFUTED"
    NEE = "NOT ENOUGH EVIDENCE"
    CONFLICT = "CONFLICTING EVIDENCE/CHERRYPICKING"

    NEGATION_CUES = [
        "false", "incorrect", "misattributed", "no evidence", "did not",
        "not true", "never", "wrongly", "debunked", "no proof",
        "without evidence", "fabricated", "fake", "hoax"
    ]

    def __init__(self, llm):
        self.llm = llm

    # -----------------------------
    # PROMPT
    # -----------------------------
    def build_prompt(self, claim, evidence_text, qa_answer, claim_date, speaker):
        return f"""
Task: Determine how the evidence relates to the claim.

Claim:
{claim}

Claim date:
{claim_date}

Speaker:
{speaker}

Evidence:
{evidence_text}

Answer derived from evidence:
{qa_answer}

---

Classify as:
SUPPORTED, REFUTED, NOT ENOUGH EVIDENCE, or CONFLICTING EVIDENCE/CHERRYPICKING.

Rules:
- Use the answer AND evidence
- If answer contradicts claim → REFUTED
- If answer supports claim → SUPPORTED
- If unclear → NOT ENOUGH EVIDENCE

IMPORTANT:

If the evidence explicitly states that something is false, fake, fabricated, or debunked,
you MUST classify it as REFUTED.

Lack of evidence can also imply REFUTATION if the claim asserts existence.

If the claim asserts that something happened,
and the evidence states it did NOT happen,
this is REFUTED — not NOT ENOUGH EVIDENCE.

Output ONLY label.
"""

    # -----------------------------
    # NORMALIZAÇÃO
    # -----------------------------
    def normalize_label(self, response: str) -> str:
        if not response:
            return self.NEE

        response = response.strip().upper()

        if response in [self.SUPPORTED, self.REFUTED, self.NEE, self.CONFLICT]:
            return response

        if "SUPPORT" in response:
            return self.SUPPORTED

        if "REFUTE" in response or "FALSE" in response:
            return self.REFUTED

        if "CONFLICT" in response:
            return self.CONFLICT

        if "NOT ENOUGH" in response or "INSUFFICIENT" in response:
            return self.NEE

        return self.NEE

    # -----------------------------
    # HEURÍSTICA
    # -----------------------------
    def apply_heuristic(self, text: str, label: str) -> str:
        text_lower = text.lower()

        if label == self.SUPPORTED:
            if any(cue in text_lower for cue in self.NEGATION_CUES):
                return self.REFUTED

        return label

    # -----------------------------
    # CLASSIFICAÇÃO
    # -----------------------------
    def classify(self, claim, evidence_text, qa_answer, claim_date, speaker):

        prompt = self.build_prompt(
            claim,
            evidence_text,
            qa_answer,
            claim_date,
            speaker
        )

        response = self.llm.generate(prompt)

        label = self.normalize_label(response)
        label = self.apply_heuristic(evidence_text, label)

        return label

    # -----------------------------
    # PIPELINE
    # -----------------------------
    def run(self, context):

        # 🆕 inicializar sem sobrescrever
        if not hasattr(context, "stances") or context.stances is None:
            context.stances = []

        # 🆕 mapear QA por evidência
        qa_map = {}
        for qa in context.qa_pairs:
            for ev in qa.get("evidence_used", []):
                qa_map[ev["text"]] = qa["answer"]

        existing_texts = set(s["text"] for s in context.stances)

        for e in context.evidence:
            text = e["text"]

            # 🆕 evitar reprocessamento
            if text in existing_texts:
                continue

            qa_answer = qa_map.get(text, "")

            stance = self.classify(
                context.claim,
                text,
                qa_answer,
                context.claim_date,
                context.speaker
            )

            context.stances.append({
                "text": text,
                "label": stance,
                "qa_answer": qa_answer,
                "rerank_score": e.get("rerank_score")
            })

        return context