from sentence_transformers import CrossEncoder
from config import Config


class CrossEncoderReranker:

    def __init__(self, model_name=None, top_k=None, threshold=None, max_evidence=None):
        self.model = CrossEncoder(model_name or Config.RERANKER_MODEL)
        self.top_k = top_k or Config.RERANKER_TOP_K
        self.threshold = threshold or Config.RERANKER_THRESHOLD
        self.max_evidence = max_evidence or getattr(Config, "MAX_EVIDENCE", 20)

    def run(self, context):
        # 🚫 NÃO apagar evidência acumulada
        if not hasattr(context, "passages") or not context.passages:
            return context

        # 🔍 query (claim ou perguntas)
        if (
            Config.USE_QUESTION_FOR_RETRIEVAL
            and hasattr(context, "questions")
            and context.questions
        ):
            query = context.questions[-1]  # 🔥 só a última pergunta
        else:
            query = context.claim

        # 🧠 preparar pares
        pairs = [(query, p["text"]) for p in context.passages]

        # ⚡ scoring
        scores = self.model.predict(pairs)

        # 📊 ordenar por score
        scored = list(zip(context.passages, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        # 🎯 aplicar threshold
        filtered = [(p, s) for p, s in scored if s >= self.threshold]

        # fallback se nada passar no threshold
        if not filtered:
            filtered = scored[: self.top_k]

        # pegar top_k
        top_passages = filtered[: self.top_k]

        # 🆕 nova evidência
        new_evidence = [
            {
                "text": p["text"],
                "rerank_score": float(s)
            }
            for p, s in top_passages
        ]

        # 🆕 inicializar evidência acumulada
        if not hasattr(context, "evidence") or context.evidence is None:
            context.evidence = []

        # 🆕 deduplicação
        existing_texts = set(e["text"] for e in context.evidence)

        for ev in new_evidence:
            if ev["text"] not in existing_texts:
                context.evidence.append(ev)

        # 🆕 controle de crescimento (muito importante)
        if len(context.evidence) > self.max_evidence:
            context.evidence = sorted(
                context.evidence,
                key=lambda x: x.get("rerank_score", 0),
                reverse=True
            )[: self.max_evidence]

        return context