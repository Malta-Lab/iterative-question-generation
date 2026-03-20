from sentence_transformers import CrossEncoder
from config import Config


class CrossEncoderReranker:

    def __init__(self, model_name=None, top_k=None, threshold=None):
        self.model = CrossEncoder(model_name or Config.RERANKER_MODEL)
        self.top_k = top_k or Config.RERANKER_TOP_K
        self.threshold = threshold or Config.RERANKER_THRESHOLD

    def run(self, context):
        if not hasattr(context, "passages") or not context.passages:
            context.evidence = []
            return context

        if Config.USE_QUESTION_FOR_RETRIEVAL and hasattr(context, "questions") and context.questions:
            query = " ".join(context.questions)
        else:
            query = context.claim

        pairs = [(query, p["text"]) for p in context.passages]
        scores = self.model.predict(pairs)

        scored = list(zip(context.passages, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        filtered = [(p, s) for p, s in scored if s >= self.threshold]

        if not filtered:
            filtered = scored[:self.top_k]

        top_passages = filtered[:self.top_k]

        context.evidence = [
            {
                "text": p["text"],
                "rerank_score": float(s)
            }
            for p, s in top_passages
        ]

        return context