from typing import Tuple


class LLMStanceDetector:

    SUPPORTED = "SUPPORTED"
    REFUTED = "REFUTED"
    NEE = "NOT ENOUGH EVIDENCE"
    CONFLICT = "CONFLICTING EVIDENCE/CHERRYPICKING"

    NEGATION_CUES = [
        "false",
        "incorrect",
        "misattributed",
        "no evidence",
        "did not",
        "not true",
        "never",
        "wrongly",
        "debunked",
        "no proof",
        "without evidence",
        "fabricated",
        "fake",
        "hoax"
    ]

    def __init__(self, llm):
        self.llm = llm

    # -----------------------------
    # 🧠 PROMPT (corrigido)
    # -----------------------------
    def build_prompt(self, claim, evidence_text, claim_date, speaker):
        return f"""
Task: Determine how the evidence relates to the claim.

You must classify the relationship as:
SUPPORTED, REFUTED, NOT ENOUGH EVIDENCE, or CONFLICTING EVIDENCE/CHERRYPICKING.

---

Claim:
{claim}

Claim date:
{claim_date}

Speaker:
{speaker}

---

Evidence:
{evidence_text}

---

DEFINITIONS:

SUPPORTED:
The evidence clearly confirms the claim.

REFUTED:
The evidence contradicts the claim OR shows that the claim is false, incorrect, misleading, or misattributed.

NOT ENOUGH EVIDENCE:
The evidence does not provide enough information to verify or refute the claim.

CONFLICTING EVIDENCE:
There is strong evidence both supporting and refuting the claim.

---

CRITICAL RULES:

1. Compare MEANING, not just keywords.

2. If the evidence says the claim is false, incorrect, misleading, misattributed, debunked → REFUTED

3. If a quote is wrongly attributed or reversed → REFUTED

4. Pay attention to WHO said WHAT:
   - If attribution differs → REFUTED

5. Similar wording ≠ support

6. If unclear → NOT ENOUGH EVIDENCE

7. Be conservative:
   - Prefer REFUTED over SUPPORTED if contradiction exists

---

Output:
Return ONLY one label:
SUPPORTED
REFUTED
NOT ENOUGH EVIDENCE
CONFLICTING EVIDENCE/CHERRYPICKING
"""

    # -----------------------------
    # 🔤 NORMALIZAÇÃO
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
    # 🛠️ HEURÍSTICA ANTI-ERRO
    # -----------------------------
    def apply_heuristic(self, text: str, label: str) -> str:
        text_lower = text.lower()

        # 🔥 regra mais importante: evitar falso SUPPORT
        if label == self.SUPPORTED:
            if any(cue in text_lower for cue in self.NEGATION_CUES):
                return self.REFUTED

        return label

    # -----------------------------
    # 🎯 CLASSIFICAÇÃO
    # -----------------------------
    def classify(self, claim, evidence_text, claim_date, speaker) -> str:

        prompt = self.build_prompt(claim, evidence_text, claim_date, speaker)

        response = self.llm.generate(prompt)

        label = self.normalize_label(response)

        # 🔧 correção pós-LLM (crítica)
        label = self.apply_heuristic(evidence_text, label)

        return label

    # -----------------------------
    # 🔄 PIPELINE
    # -----------------------------
    def run(self, context):

        stances = []

        for e in context.evidence:
            text = e["text"]

            stance = self.classify(
                context.claim,
                text,
                context.claim_date,
                context.speaker
            )

            stances.append({
                "text": text,
                "label": stance,
                "bm25_score": e.get("bm25_score"),
                "rerank_score": e.get("rerank_score")
            })

        context.stances = stances

        return context