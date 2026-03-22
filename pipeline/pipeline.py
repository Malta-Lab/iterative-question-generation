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

                 # 🆕 módulos plugáveis
                 claim_decomposer=None,
                 counterfactual_module=None,
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

        # 🆕 plugáveis
        self.claim_decomposer = claim_decomposer
        self.counterfactual_module = counterfactual_module

    def run(self, context):

        # 🆕 1. decomposição de claim (opcional)
        if self.claim_decomposer:
            context = self.claim_decomposer.run(context)

            # fallback: se não gerar subclaims
            subclaims = getattr(context, "subclaims", [])
            if subclaims:
                return self._run_atomic(context)

        # 🔁 pipeline normal
        return self._run_single(context)

    # -----------------------------
    # 🔁 PIPELINE NORMAL
    # -----------------------------
    def _run_single(self, context):

        for i in range(context.max_iterations):
            context.iteration = i

            context = self.question_generator.run(context)
            context = self.searcher.run(context)
            context = self.parser.run(context)

            # 🔥 ordem corrigida
            context = self.retriever.run(context)
            context = self.segmenter.run(context)

            if Config.USE_RERANKER:
                context = self.reranker.run(context)

            # 🆕 contrafactual
            if self.counterfactual_module:
                context = self.counterfactual_module.run(context)

            context = self.qa_generator.run(context)
            context = self.stance_detector.run(context)
            context = self._update_belief(context)

            if context.confidence > 0.8 and i > 0:
                break

        context = self.verdict_predictor.run(context)
        context = self.justification_generator.run(context)

        return context

    # -----------------------------
    # 🧠 PIPELINE ATÔMICO
    # -----------------------------
    def _run_atomic(self, context):

        results = []

        for subclaim in context.subclaims:

            sub_context = self._clone_context(context, subclaim)

            sub_context = self._run_single(sub_context)

            results.append({
                "subclaim": subclaim,
                "verdict": sub_context.verdict,
                "confidence": sub_context.confidence
            })

        context.atomic_results = results

        # agregação simples
        support = sum(1 for r in results if r["verdict"] == "SUPPORTED")
        refute = sum(1 for r in results if r["verdict"] == "REFUTED")

        if refute > support:
            context.verdict = "REFUTED"
        elif support > refute:
            context.verdict = "SUPPORTED"
        else:
            context.verdict = "NOT ENOUGH EVIDENCE"

        return context

    def _clone_context(self, context, claim):
        from pipeline.context import ClaimContext

        return ClaimContext(
            context.id,
            claim,
            context.claim_date,
            context.speaker
        )

    # -----------------------------
    # 📊 BELIEF
    # -----------------------------
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