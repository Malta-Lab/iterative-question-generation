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
        context = self.question_generator.run(context)
        context = self.searcher.run(context)
        context = self.parser.run(context)
        context = self.segmenter.run(context)
        context = self.retriever.run(context)
        context = self.reranker.run(context)
        context = self.qa_generator.run(context)
        context = self.stance_detector.run(context)
        context = self.verdict_predictor.run(context)
        context = self.justification_generator.run(context)
        return context