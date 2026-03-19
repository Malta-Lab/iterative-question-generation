from ddgs import DDGS

class DuckDuckGoSearch:

    def search(self, query):

        results = []

        try:
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=5):
                    if "href" in r:
                        results.append(r["href"])
        except Exception:
            return []

        return results

    def run(self, context):

        queries = [context.claim] + getattr(context, "questions", [])

        urls = []

        for q in queries:
            urls.extend(self.search(q))

        context.search_results = list(set(urls))[:10]

        return context