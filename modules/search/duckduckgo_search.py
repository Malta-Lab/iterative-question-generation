from ddgs import DDGS
from concurrent.futures import ThreadPoolExecutor, as_completed


class DuckDuckGoSearch:

    def __init__(self, max_results, max_urls, max_workers):
        self.max_results = max_results
        self.max_urls = max_urls
        self.max_workers = max_workers

    def search(self, ddgs, query: str):
        results = []

        try:
            for r in ddgs.text(query, max_results=self.max_results):
                href = r.get("href")
                if href:
                    results.append(href)
        except Exception:
            return []

        return results

    def run(self, context):

        # 🔥 MELHORIA 1: reduzir queries
        queries = [context.claim]

        # opcional: usar só top 2 perguntas
        questions = getattr(context, "questions", [])
        queries.extend(questions[:2])

        urls = []

        # 🔥 MELHORIA 2: UMA sessão DDGS
        with DDGS() as ddgs:

            # 🔥 MELHORIA 3: paralelismo
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:

                futures = [
                    executor.submit(self.search, ddgs, q)
                    for q in queries
                ]

                for future in as_completed(futures):
                    try:
                        urls.extend(future.result())
                    except Exception:
                        pass

        # 🔥 MELHORIA 4: deduplicação melhor
        seen = set()
        unique_urls = []
        for url in urls:
            if url not in seen:
                seen.add(url)
                unique_urls.append(url)

        context.search_results = unique_urls[:self.max_urls]

        return context