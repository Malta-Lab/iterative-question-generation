import requests
from config import Config
from utils.cache_utils import load_cache, save_cache


class BraveSearch:

    def __init__(self):
        self.api_key = Config.BRAVE_API_KEY
        self.cache = load_cache()

    def search(self, query):

        # 1️⃣ cache
        if query in self.cache:
            return self.cache[query]

        url = "https://api.search.brave.com/res/v1/web/search"

        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.api_key
        }

        params = {
            "q": query,
            "count": 5
        }

        try:
            response = requests.get(url, headers=headers, params=params)
            data = response.json()
        except Exception:
            return []

        results = []

        for r in data.get("web", {}).get("results", []):
            if "url" in r:
                results.append(r["url"])

        # 2️⃣ cache
        self.cache[query] = results
        save_cache(self.cache)

        return results

    def run(self, context):

        queries = [context.claim] + getattr(context, "questions", [])

        urls = []

        for q in queries:
            urls.extend(self.search(q))

        # remove duplicados
        context.search_results = list(set(urls))[:10]

        return context