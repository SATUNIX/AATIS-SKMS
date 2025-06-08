import requests

class SearxngSearchTool:
    def __init__(self, endpoint="http://localhost:8888"):
        self.endpoint = endpoint.rstrip("/") + "/search"

    def search(self, query, max_results=5):
        params = {"q": query, "format": "json"}
        resp = requests.get(self.endpoint, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        hits = data.get("results", [])[:max_results]
        return [
            {"title": h.get("title"), "link": h.get("url"), "snippet": h.get("content")}
            for h in hits
        ]
