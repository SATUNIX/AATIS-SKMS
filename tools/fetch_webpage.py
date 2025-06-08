import requests
from bs4 import BeautifulSoup

class FetchWebpageTool:
    def __init__(self, timeout=10):
        self.timeout = timeout

    def fetch(self, url):
        resp = requests.get(url, timeout=self.timeout)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(lines)
