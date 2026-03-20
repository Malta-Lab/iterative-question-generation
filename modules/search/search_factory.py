from config import Config

from modules.search.brave_search import BraveSearch
from modules.search.duckduckgo_search import DuckDuckGoSearch
from modules.search.google_cse_search import GoogleCSESearch
from modules.search.serpapi_search import SerpAPISearch


def get_searcher():

    engine = Config.SEARCH_ENGINE

    if engine == "duckduckgo":
        return DuckDuckGoSearch(
            max_results=Config.SEARCH_MAX_RESULTS,
            max_urls=Config.SEARCH_MAX_URLS,
            max_workers=Config.SEARCH_MAX_WORKERS
        )

    if engine == "brave":
        return BraveSearch(
            api_key=Config.BRAVE_API_KEY,
            max_results=Config.SEARCH_MAX_RESULTS,
            max_urls=Config.SEARCH_MAX_URLS,
        )

    if engine == "serpapi":
        return SerpAPISearch(
            api_key=Config.SERPAPI_API_KEY,
            max_results=Config.SEARCH_MAX_RESULTS,
            max_urls=Config.SEARCH_MAX_URLS,
        )

    if engine == "google_cse":
        return GoogleCSESearch(
            api_key=Config.GOOGLE_API_KEY,
            cse_id=Config.GOOGLE_CSE_ID,
            max_results=Config.SEARCH_MAX_RESULTS,
            max_urls=Config.SEARCH_MAX_URLS,
        )

    # fallback
    return DuckDuckGoSearch(
        max_results=Config.SEARCH_MAX_RESULTS,
        max_urls=Config.SEARCH_MAX_URLS,
    )