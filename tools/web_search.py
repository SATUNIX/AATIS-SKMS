import os
import time
import requests
from pathlib import Path
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
from readability import Document

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

console = Console()

class SimpleSearchFetcher:
    """
    1) Searches DuckDuckGo for top-N URLs on a topic, throttled.
    2) Fetches each page and extracts main text, throttled.
    3) Saves each full text to web_content/<n>.txt
    """

    def __init__(
        self,
        num_results: int = 3,
        timeout: float = 5.0,
        search_delay: float = 1.0,
        fetch_delay: float = 1.0,
        output_dir: str = "web_content",
    ):
        self.num_results   = num_results
        self.timeout       = timeout
        self.search_delay  = search_delay
        self.fetch_delay   = fetch_delay
        self.headers = {
            "User-Agent": "Mozilla/5.0 (compatible; SimpleSearchFetcher/1.0)"
        }
        # ensure output directory exists
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _safe_filename(self, url: str, idx: int) -> str:
        # simple fallback: page_<idx>.txt
        # if you want to slugify the URL, do it here instead
        return f"page_{idx+1}.txt"

    def search_urls(self, query: str) -> list[str]:
        urls = []
        with console.status(f"[bold blue]Searching DuckDuckGo for:[/] {query}", spinner="dots"):
            with DDGS() as ddgs:
                for hit in ddgs.text(
                    keywords=query,
                    region="wt-wt",
                    timelimit=None,
                    max_results=self.num_results
                ):
                    href = hit.get("href") or hit.get("url")
                    if href:
                        urls.append(href)
                    if len(urls) >= self.num_results:
                        break
                    time.sleep(self.search_delay)

        console.print(
            Panel(
                "\n".join(urls),
                title=f"Top {len(urls)} URLs for '{query}'",
                expand=False,
            )
        )
        return urls

    def fetch_page_text(self, url: str) -> str:
        try:
            resp = requests.get(url, timeout=self.timeout, headers=self.headers)
            resp.raise_for_status()
        except Exception as e:
            console.log(f"[red]Error fetching {url}:[/] {e}")
            return f"[Error fetching {url}: {e}]"

        doc = Document(resp.text)
        summary_html = doc.summary()
        soup = BeautifulSoup(summary_html, "html.parser")
        text = soup.get_text(separator="\n").strip()
        if text:
            return text

        soup = BeautifulSoup(resp.text, "html.parser")
        paras = [p.get_text(strip=True) for p in soup.find_all("p")]
        return "\n\n".join(paras)

    def run(self, topic: str) -> dict[str, str]:
        output = {}
        urls = self.search_urls(topic)

        for idx, url in enumerate(urls):
            with console.status(f"[bold blue]Fetching:[/] {url}", spinner="dots"):
                text = self.fetch_page_text(url)

            snippet = text[:1000] + ("..." if len(text) > 1000 else "")
            console.print(
                Panel(
                    Markdown(snippet),
                    title=url,
                    expand=False,
                    border_style="green",
                )
            )

            # save full text to disk
            filename = self._safe_filename(url, idx)
            path = self.output_dir / filename
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(text)
                console.log(f"[green]Saved full text to[/] {path}")
            except Exception as e:
                console.log(f"[red]Failed to save {path}:[/] {e}")

            output[url] = text[:20_000]  # cap for downstream agents
            time.sleep(self.fetch_delay)

        console.print(f"[bold green]Done! Retrieved & saved {len(output)} pages.[/]")
        return output


if __name__ == "__main__":
    topic = "latest advances in transformer models"
    sf = SimpleSearchFetcher(
        num_results=3,
        search_delay=2.0,   # seconds between search hits
        fetch_delay=2.0     # seconds between page fetches
    )

    pages = sf.run(topic)
