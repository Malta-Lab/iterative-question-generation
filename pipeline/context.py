class ClaimContext:

    def __init__(self, claim_id, claim_text, claim_date=None, speaker=None):
        self.id = claim_id
        self.claim = claim_text
        self.claim_date = claim_date or "Unknown"
        self.speaker = speaker or "Unknown"

        # pipeline original
        self.questions = []
        self.search_results = []
        self.documents = []
        self.passages = []
        self.qa_pairs = []
        self.stances = []
        self.evidence = []

        # 🆕 controle iterativo
        self.iteration = 0
        self.max_iterations = 3

        # 🆕 belief state (ESSENCIAL)
        self.support_score = 0.0
        self.refute_score = 0.0
        self.confidence = 0.0

        self.support_evidence = []
        self.refute_evidence = []
        self.neutral_evidence = []

        # saída
        self.verdict = None
        self.justification = None