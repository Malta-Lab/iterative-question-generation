from dotenv import load_dotenv
import os

load_dotenv()


class Config:
    # LLM
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.0))
    LANGUAGE = os.getenv("LANGUAGE", "en")

    # Search
    SEARCH_ENGINE = os.getenv("SEARCH_ENGINE", "brave")
    SEARCH_MAX_WORKERS = int(os.getenv("SEARCH_MAX_WORKERS", 5))
    SEARCH_MAX_RESULTS = int(os.getenv("SEARCH_MAX_RESULTS", 10))
    SEARCH_MAX_URLS = int(os.getenv("SEARCH_MAX_URLS", 50))

    # Search Engine API Keys
    BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")
    SERPAPI_KEY = os.getenv("SERPAPI_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

    # Retrieval
    BM25_TOP_K = int(os.getenv("BM25_TOP_K", 5))

    # Reranking
    RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    RERANKER_TOP_K = int(os.getenv("RERANKER_TOP_K", 3))
    RERANKER_THRESHOLD = float(os.getenv("RERANKER_THRESHOLD", 0.3))
    

    # Passages
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 300))

    # Pipeline
    MAX_QUESTIONS = int(os.getenv("MAX_QUESTIONS", 5))
    MAX_EVIDENCE = int(os.getenv("MAX_EVIDENCE", 5))

    # Debug
    #SAVE_INTERMEDIATE = os.getenv("SAVE_INTERMEDIATE", "true").lower() == "true"
    #VERBOSE = os.getenv("VERBOSE", "false").lower() == "true"




    # ===== FLAGS =====
    USE_RERANKER = os.getenv("USE_RERANKER", "true").lower() == "true"
    USE_QUESTION_FOR_RETRIEVAL = os.getenv("USE_QUESTION_FOR_RETRIEVAL", "true").lower() == "true"