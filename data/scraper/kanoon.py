"""Indian Kanoon (indiankanoon.org) scraper for judgment summaries.

Scrapes judgment summaries across criminal, civil, and consumer domains.
Extracts case details, holdings, and cited sections.
"""

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from config import config


@dataclass
class Judgment:
    """A judgment summary from Indian Kanoon."""

    case_name: str
    court: str
    date: str
    cited_sections: list[str] = field(default_factory=list)
    summary: str = ""
    domain: str = ""
    url: str = ""
    source: str = "indian_kanoon"


class KanoonScraper:
    """Scraper for Indian Kanoon (indiankanoon.org)."""

    BASE_URL = "https://indiankanoon.org"

    # Search queries for different legal domains
    DOMAIN_QUERIES = {
        "criminal": ["BNS murder", "IPC 302", "BNS theft", "criminal breach of trust"],
        "civil": ["consumer complaint", "property dispute", " breach of contract"],
        "consumer": ["consumer protection act", "deficiency in service", "unfair trade practice"],
    }

    def __init__(self, delay: float = None):
        self.delay = delay or config.request_delay
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "THEMIS-Legal-AI/1.0 (Academic Research)",
            "Accept": "text/html,application/xhtml+xml",
        })

    def _get(self, url: str, retries: int = None) -> requests.Response:
        """GET request with retry and rate limiting."""
        retries = retries or config.max_retries
        for attempt in range(retries):
            try:
                time.sleep(self.delay)
                resp = self.session.get(url, timeout=30)
                resp.raise_for_status()
                return resp
            except requests.RequestException as e:
                if attempt == retries - 1:
                    raise
                wait = 2 ** attempt
                print(f"  Retry {attempt + 1}/{retries} after {wait}s: {e}")
                time.sleep(wait)
        raise RuntimeError(f"Failed after {retries} retries")

    def search(self, query: str, page: int = 1) -> list[dict]:
        """Search for judgments. Returns list of result links."""
        url = f"{self.BASE_URL}/search/"
        params = {
            "q": query,
            "formInput": query,
            "pagenum": page,
        }
        resp = self._get(url, params=params)
        soup = BeautifulSoup(resp.text, "html.parser")

        results = []
        for result_div in soup.find_all("div", class_="result"):
            link = result_div.find("a", class_="result_title")
            if not link:
                continue

            title = link.get_text(strip=True)
            href = link.get("href", "")

            # Extract snippet/summary
            snippet_div = result_div.find("div", class_="docsnippet")
            snippet = snippet_div.get_text(strip=True) if snippet_div else ""

            results.append({
                "title": title,
                "url": urljoin(self.BASE_URL, href),
                "snippet": snippet,
            })

        return results

    def parse_judgment_page(self, url: str) -> dict:
        """Parse a single judgment page for details."""
        resp = self._get(url)
        soup = BeautifulSoup(resp.text, "html.parser")

        data = {
            "case_name": "",
            "court": "",
            "date": "",
            "cited_sections": [],
            "summary": "",
        }

        # Extract case name from header
        header = soup.find("div", class_="judgments")
        if header:
            h2 = header.find("h2")
            if h2:
                data["case_name"] = h2.get_text(strip=True)

        # Extract metadata
        for div in soup.find_all("div", class_="judgment_right"):
            text = div.get_text(strip=True)
            if "Court:" in text:
                data["court"] = text.split("Court:")[-1].strip()
            if "Date:" in text:
                date_match = re.search(r"Date:\s*(\d{1,2}/\d{1,2}/\d{4})", text)
                if date_match:
                    data["date"] = date_match.group(1)

        # Extract cited sections (e.g., "Section 302 IPC", "Section 118 BNS")
        body_text = soup.get_text()
        section_pattern = r"Section\s+(\d+[A-Z]?)\s+(IPC|BNS|BNSS|BSA)"
        cited = re.findall(section_pattern, body_text, re.IGNORECASE)
        data["cited_sections"] = [f"{sec} {law}" for sec, law in cited]

        # Extract judgment text (first few paragraphs)
        judgment_div = soup.find("div", class_="judgment_right")
        if judgment_div:
            paragraphs = judgment_div.find_all("p")
            summary_parts = []
            for p in paragraphs[:10]:  # First 10 paragraphs
                text = p.get_text(strip=True)
                if text and len(text) > 20:
                    summary_parts.append(text)
            data["summary"] = "\n\n".join(summary_parts)

        return data

    def scrape_domain(self, domain: str, max_results: int = 50) -> list[Judgment]:
        """Scrape judgments for a specific legal domain."""
        print(f"\nScraping domain: {domain}")
        judgments = []
        queries = self.DOMAIN_QUERIES.get(domain, [])

        for query in queries:
            print(f"  Searching: {query}")
            results = self.search(query)

            for result in results[:max_results // len(queries)]:
                try:
                    data = self.parse_judgment_page(result["url"])
                    judgment = Judgment(
                        case_name=data["case_name"],
                        court=data["court"],
                        date=data["date"],
                        cited_sections=data["cited_sections"],
                        summary=data["summary"],
                        domain=domain,
                        url=result["url"],
                    )
                    judgments.append(judgment)
                    print(f"    Scraped: {judgment.case_name[:60]}...")
                except Exception as e:
                    print(f"    Warning: Failed to scrape {result['url']}: {e}")

        return judgments

    def save_judgments(self, judgments: list[Judgment], domain: str, output_dir: Path = None):
        """Save judgments to JSON."""
        output_dir = output_dir or config.raw_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        filepath = output_dir / f"kanoon_{domain}.json"
        data = [
            {
                "source": "indian_kanoon",
                "case_name": j.case_name,
                "court": j.court,
                "date": j.date,
                "cited_sections": j.cited_sections,
                "summary": j.summary,
                "domain": j.domain,
                "url": j.url,
            }
            for j in judgments
        ]

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"  Saved {len(judgments)} judgments to {filepath}")
        return filepath


def scrape_all_domains():
    """Scrape judgments from all target domains."""
    scraper = KanoonScraper()
    all_judgments = []

    for domain in scraper.DOMAIN_QUERIES:
        judgments = scraper.scrape_domain(domain, max_results=50)
        scraper.save_judgments(judgments, domain)
        all_judgments.extend(judgments)

    print(f"\nTotal judgments scraped: {len(all_judgments)}")
    return all_judgments


if __name__ == "__main__":
    scrape_all_domains()
