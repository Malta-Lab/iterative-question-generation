from config import Config


class Pipeline:

    def __init__(self,
                 question_generator,
                 searcher,
                 parser,
                 segmenter,
                 retriever,
                 reranker,
                 qa_generator,
                 stance_detector,
                 verdict_predictor,
                 justification_generator,
                 ):

        self.question_generator = question_generator
        self.searcher = searcher
        self.parser = parser
        self.segmenter = segmenter
        self.retriever = retriever
        self.reranker = reranker
        self.qa_generator = qa_generator
        self.stance_detector = stance_detector
        self.verdict_predictor = verdict_predictor
        self.justification_generator = justification_generator
        

    def run(self, context):

        for i in range(context.max_iterations):
            context.iteration = i

            # 🧠 nova pergunta baseada no estado atual
            context = self.question_generator.run(context)

            # 🔍 busca
            context = self.searcher.run(context)

            # 📄 parsing
            context = self.parser.run(context)

            # 📚 retrieval
            context = self.retriever.run(context)

            # ✂️ segmentação
            context = self.segmenter.run(context)

            if Config.USE_RERANKER:
                context = self.reranker.run(context)

            # 🤖 QA
            context = self.qa_generator.run(context)

            # ⚖️ stance
            context = self.stance_detector.run(context)

            # 🆕 atualização de crença
            context = self._update_belief(context)

            # 🛑 parada
            if context.confidence > 0.8:
                break

        # decisão final
        context = self.verdict_predictor.run(context)
        context = self.justification_generator.run(context)

        return context
    

    def _update_belief(self, context):

        support = 0
        refute = 0

        context.support_evidence = []
        context.refute_evidence = []
        context.neutral_evidence = []

        for s in context.stances:
            label = s.get("label", "").upper()

            if label == "SUPPORTED":
                support += 1
                context.support_evidence.append(s)

            elif label == "REFUTED":
                refute += 1
                context.refute_evidence.append(s)

            else:
                context.neutral_evidence.append(s)

        total = max(1, support + refute)

        context.support_score = support / total
        context.refute_score = refute / total
        context.confidence = max(context.support_score, context.refute_score)

        return context    