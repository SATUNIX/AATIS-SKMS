# tools/searxng_search.py

# from pyserxng import SearXNGClient
# from pyserxng.models import SearchConfig, SafeSearchLevel, SearchCategory

class SearxngSearchTool:
    """
    Placeholder for SearXNG search integration.
    Currently disabled; returns empty results to trigger fallback mechanisms.
    """

    def __init__(self, base_url: str = "http://localhost:8888"):
        # Initialization logic is currently disabled.
        # self.client = SearXNGClient(base_url)
        pass

    def search(
        self,
        query: str,
        max_results: int = 5,
        categories=None,
        safe_search=None,
        page: int = 1,
        language: str = "en"
    ) -> list[dict]:
        """
        Simulates a failed search operation to allow agent fallback.
        :param query: Search query string.
        :param max_results: Maximum number of results to return.
        :param categories: Search categories (currently unused).
        :param safe_search: Safe search level (currently unused).
        :param page: Page number for pagination (currently unused).
        :param language: Language code (currently unused).
        :return: Empty list indicating no results found.
        """
        print(f"⚠️ SearXNG search is currently disabled. Query: '{query}'")
        return []

    def __del__(self):
        # Cleanup logic is currently disabled.
        # if hasattr(self, 'client'):
        #     self.client.close()
        pass
