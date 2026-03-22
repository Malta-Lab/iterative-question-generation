from config import Config

from modules.search.search_factory import get_searcher
from modules.verdict.llm_verdict import LLMVerdict
from pipeline.pipeline import Pipeline
from modules.llm.ollama_interface import OllamaLLM
from modules.question_generation.question_generator import QuestionGenerator
from modules.parsing.document_parser import DocumentParser
from modules.segmentation.passage_extractor import PassageExtractor
from modules.retrieval.bm25_retriever import BM25Retriever
from modules.reranking.cross_encoder_reranker import CrossEncoderReranker
from modules.stance.llm_stance_detector import LLMStanceDetector
from modules.qa.qa_generator import QAGenerator
from modules.verdict.rule_verdict import RuleVerdict
from modules.verdict.majority_verdict import MajorityVerdict
from modules.justification.justification_generator import JustificationGenerator


def averitec_pipeline():
    llm = OllamaLLM(Config.OLLAMA_MODEL)

    pipeline = Pipeline(
        question_generator=QuestionGenerator(llm),
        searcher=get_searcher(),
        parser=DocumentParser(),

        retriever=BM25Retriever(
            top_k=Config.BM25_TOP_K
        ),

        segmenter=PassageExtractor(
            chunk_size=Config.CHUNK_SIZE
        ),

        reranker=CrossEncoderReranker(
            model_name=Config.RERANKER_MODEL,
            top_k=Config.RERANKER_TOP_K,
            threshold=Config.RERANKER_THRESHOLD
        ),

        qa_generator=QAGenerator(llm),
        stance_detector=LLMStanceDetector(llm),

        verdict_predictor=LLMVerdict(llm),
        justification_generator=JustificationGenerator(llm),
    )

    return pipeline

def averitec_pipeline_with_rule_veredict():
    llm = OllamaLLM(Config.OLLAMA_MODEL)

    pipeline = Pipeline(
        question_generator=QuestionGenerator(llm),
        searcher=get_searcher(),
        parser=DocumentParser(),

        retriever=BM25Retriever(
            top_k=Config.BM25_TOP_K
        ),

        segmenter=PassageExtractor(
            chunk_size=Config.CHUNK_SIZE
        ),

        reranker=CrossEncoderReranker(
            model_name=Config.RERANKER_MODEL,
            top_k=Config.RERANKER_TOP_K,
            threshold=Config.RERANKER_THRESHOLD
        ),

        qa_generator=QAGenerator(llm),
        stance_detector=LLMStanceDetector(llm),

        verdict_predictor=RuleVerdict(),
        justification_generator=JustificationGenerator(llm),
    )

    return pipeline