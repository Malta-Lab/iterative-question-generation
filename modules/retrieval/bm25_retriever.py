from rank_bm25 import BM25Okapi
from config import Config


class BM25Retriever:

    def __init__(self, top_k=None):
        self.top_k = top_k or Config.BM25_TOP_K

    def run(self, context):
        if not context.documents:
            return context

        if Config.USE_QUESTION_FOR_RETRIEVAL and hasattr(context, "questions") and context.questions:
            query = " ".join(context.questions)
        else:
            query = context.claim

        tokenized_docs = [doc.split() for doc in context.documents]
        tokenized_query = query.split()

        bm25 = BM25Okapi(tokenized_docs)
        scores = bm25.get_scores(tokenized_query)

        ranked = sorted(
            zip(context.documents, scores),
            key=lambda x: x[1],
            reverse=True
        )

        context.documents = [doc for doc, _ in ranked[:self.top_k]]
        return context