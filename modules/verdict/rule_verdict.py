from collections import Counter
from .base_verdict import BaseVerdict


class RuleVerdict(BaseVerdict):

    # 🔒 Constantes (evita string hardcoded espalhada)
    SUPPORTED = "SUPPORTED"
    REFUTED = "REFUTED"
    NEE = "NOT ENOUGH EVIDENCE"
    CONFLICT = "CONFLICTING EVIDENCE/CHERRYPICKING"

    def normalize_label(self, label: str) -> str:
        label = label.upper()

        if label in ["SUPPORT", "SUPPORTED"]:
            return self.SUPPORTED

        elif label in ["REFUTE", "REFUTED"]:
            return self.REFUTED

        elif "NOT ENOUGH" in label:
            return self.NEE

        elif "CONFLICTING" in label:
            return self.CONFLICT

        return label

    def extract_label_and_weight(self, s):
        """
        Extrai label e peso opcional.
        Compatível com:
        - dict: {"label": "...", "score": float}
        - tuple: (..., label)
        """
        label = None
        weight = 1.0  # default

        if isinstance(s, dict):
            label = s.get("label")
            weight = float(s.get("score", 1.0))

        elif isinstance(s, tuple):
            if len(s) >= 2:
                label = s[1]

        if label is None:
            return None, 0.0

        return self.normalize_label(label), weight

    def run(self, context):

        weighted_counts = Counter()

        # 🔍 agrega evidências
        for s in context.stances:
            label, weight = self.extract_label_and_weight(s)

            if label is None:
                continue

            weighted_counts[label] += weight

        # 🧱 caso vazio
        if not weighted_counts:
            context.verdict = self.NEE
            context.verdict_scores = {}
            context.confidence = 0.0
            return context

        # 📊 extrai contagens
        support = weighted_counts.get(self.SUPPORTED, 0.0)
        refute = weighted_counts.get(self.REFUTED, 0.0)
        nee = weighted_counts.get(self.NEE, 0.0)
        conflicting = weighted_counts.get(self.CONFLICT, 0.0)

        total = support + refute + nee + conflicting

        # 🧠 lógica de decisão (corrigida)
        if support > refute:
            verdict = self.SUPPORTED

        elif refute > support:
            verdict = self.REFUTED

        elif support == 0 and refute == 0:
            if conflicting > 0:
                verdict = self.CONFLICT
            else:
                verdict = self.NEE

        else:
            verdict = self.CONFLICT

        # 📈 confiança simples (pode evoluir depois)
        if total > 0:
            confidence = abs(support - refute) / total
        else:
            confidence = 0.0

        # 🧾 salva no contexto
        context.verdict = verdict
        context.verdict_scores = {
            self.SUPPORTED: support,
            self.REFUTED: refute,
            self.NEE: nee,
            self.CONFLICT: conflicting,
            "total": total,
        }
        context.confidence = confidence

        return context