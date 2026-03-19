from config import Config
from modules.search.brave_search import BraveSearch
from modules.search.duckduckgo_search import DuckDuckGoSearch
from modules.search.google_cse_search import GoogleCSESearch
from modules.search.serpapi_search import SerpAPISearch

def get_searcher():

    if Config.SEARCH_ENGINE == "duckduckgo":
        return DuckDuckGoSearch()

    if Config.SEARCH_ENGINE == "brave":
        return BraveSearch()

    if Config.SEARCH_ENGINE == "serpapi":
        return SerpAPISearch()


    if Config.SEARCH_ENGINE == "google_cse":
        return GoogleCSESearch()


    return DuckDuckGoSearch()