"""India Code (indiacode.nic.in) scraper for Bare Act sections.

Scrapes central acts from India Code website, extracting section numbers,
titles, and full text. Downloads PDFs as fallback for parsing.
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
class Section:
    """A single section of a Bare Act."""

    section_number: str
    title: str
    text: str = ""
    chapter: str = ""
    act_name: str = ""
    replaces: str = ""  # e.g., "IPC 302"


@dataclass
class Act:
    """A complete Bare Act with all sections."""

    name: str
    year: str
    act_id: str
    handle_id: str
    enactment_date: str = ""
    ministry: str = ""
    department: str = ""
    sections: list[Section] = field(default_factory=list)


class IndiaCodeScraper:
    """Scraper for India Code (indiacode.nic.in)."""

    BASE_URL = config.india_code_base

    def __init__(self, delay: float = None):
        self.delay = delay or config.request_delay
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "THEMIS-Legal-AI/1.0 (Academic Research)",
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en-US,en;q=0.9",
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

    def search_act(self, query: str) -> list[dict]:
        """Search for an act by name. Returns list of matching acts."""
        url = f"{self.BASE_URL}/handle/123456789/1362/simple-search"
        params = {
            "query": query,
            "searchradio": "acts",
            "rpp": 10,
            "sort_by": "score",
            "order": "desc",
        }
        resp = self._get(url, params=params)
        soup = BeautifulSoup(resp.text, "html.parser")

        results = []
        table = soup.find("table", class_="table")
        if not table:
            return results

        rows = table.find_all("tr")[1:]  # Skip header
        for row in rows:
            cols = row.find_all("td")
            if len(cols) < 4:
                continue
            link = cols[3].find("a")
            if not link:
                continue
            href = link.get("href", "")
            # Extract handle ID from href
            match = re.search(r"/handle/(\d+/\d+)", href)
            if match:
                results.append({
                    "enactment_date": cols[0].text.strip(),
                    "act_number": cols[1].text.strip(),
                    "short_title": cols[2].text.strip(),
                    "handle_id": match.group(1),
                    "view_url": urljoin(self.BASE_URL, href),
                })
        return results

    def get_act_page(self, handle_id: str) -> BeautifulSoup:
        """Fetch the main page for an act."""
        url = f"{self.BASE_URL}/handle/{handle_id}"
        resp = self._get(url)
        return BeautifulSoup(resp.text, "html.parser")

    def extract_section_links(self, soup: BeautifulSoup) -> list[dict]:
        """Extract section links from act page."""
        sections = []
        # Find all section links in the accordion
        for link in soup.find_all("a", class_="title"):
            href = link.get("href", "")
            text = link.get_text(strip=True)
            # Match section number pattern
            match = re.search(r"sectionno=(\d+)", href)
            if match:
                section_no = match.group(1)
                # Clean section title
                title_match = re.search(r"Section \d+\.\s*(.*)", text)
                title = title_match.group(1).strip() if title_match else text
                sections.append({
                    "section_number": section_no,
                    "title": title,
                    "url": urljoin(self.BASE_URL, href),
                })
        return sections

    def extract_chapters(self, soup: BeautifulSoup) -> dict[str, str]:
        """Extract chapter structure mapping section ranges to chapter names."""
        chapters = {}
        # Find chapter headings
        for li in soup.find_all("li"):
            b = li.find("b")
            if b and "CHAPTER" in b.text:
                chapter_name = b.text.strip()
                # Find sub-sections under this chapter
                sub_ul = li.find("ul")
                if sub_ul:
                    for a in sub_ul.find_all("a", class_="headingtwo"):
                        heading = a.text.strip()
                        # This chapter covers sections in this range
                        chapters[heading] = chapter_name
        return chapters

    def fetch_section_text(self, section_url: str) -> str:
        """Fetch the full text of a single section."""
        try:
            resp = self._get(section_url)
            soup = BeautifulSoup(resp.text, "html.parser")
            # The section text is in the page content
            # Look for the main content area
            content = soup.find("div", class_="panel-body")
            if content:
                # Get all paragraph text
                paragraphs = content.find_all("p")
                text = "\n\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
                if text:
                    return text
            # Fallback: get all text from main content
            main = soup.find("div", id="col8") or soup.find("main")
            if main:
                return main.get_text(separator="\n", strip=True)
        except Exception as e:
            print(f"  Warning: Failed to fetch section text: {e}")
        return ""

    def scrape_act(self, handle_id: str, act_name: str = "") -> Act:
        """Scrape a complete act by its handle ID."""
        print(f"Scraping act: {handle_id}")

        # Get act page
        soup = self.get_act_page(handle_id)

        # Extract metadata
        title_tag = soup.find("td", id="short_title")
        title = title_tag.text.strip() if title_tag else act_name

        # Extract year from title
        year_match = re.search(r"(\d{4})", title)
        year = year_match.group(1) if year_match else ""

        # Extract act ID
        act_id_tag = soup.find("td", class_="metadataFieldValue", string=re.compile(r"\d{4}-\d+"))
        act_id = act_id_tag.text.strip() if act_id_tag else handle_id

        act = Act(
            name=title,
            year=year,
            act_id=act_id,
            handle_id=handle_id,
        )

        # Extract section links
        section_links = self.extract_section_links(soup)
        print(f"  Found {len(section_links)} sections")

        # Extract chapter structure
        chapters = self.extract_chapters(soup)

        # Fetch each section
        for i, sec_info in enumerate(section_links):
            print(f"  Fetching section {sec_info['section_number']}/{len(section_links)}...")
            text = self.fetch_section_text(sec_info["url"])

            section = Section(
                section_number=sec_info["section_number"],
                title=sec_info["title"],
                text=text,
                act_name=title,
            )
            act.sections.append(section)

        print(f"  Scraped {len(act.sections)} sections from {title}")
        return act

    def save_act(self, act: Act, output_dir: Path = None):
        """Save act data to JSON."""
        output_dir = output_dir or config.raw_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create filename from act name
        safe_name = re.sub(r"[^\w\s-]", "", act.name.lower())
        safe_name = re.sub(r"\s+", "_", safe_name.strip())
        filepath = output_dir / f"{safe_name}.json"

        data = {
            "source": "india_code",
            "name": act.name,
            "year": act.year,
            "act_id": act.act_id,
            "handle_id": act.handle_id,
            "enactment_date": act.enactment_date,
            "ministry": act.ministry,
            "department": act.department,
            "section_count": len(act.sections),
            "sections": [
                {
                    "section_number": s.section_number,
                    "title": s.title,
                    "text": s.text,
                    "chapter": s.chapter,
                    "act_name": s.act_name,
                    "replaces": s.replaces,
                }
                for s in act.sections
            ],
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"  Saved to {filepath}")
        return filepath


def scrape_target_laws():
    """Scrape all target laws for THEMIS training."""
    scraper = IndiaCodeScraper()

    for law_name in config.target_laws:
        print(f"\n{'='*60}")
        print(f"Searching for: {law_name}")
        print("=" * 60)

        results = scraper.search_act(law_name)
        if not results:
            print(f"  No results found for '{law_name}'")
            continue

        # Take the first (most relevant) result
        best = results[0]
        print(f"  Found: {best['short_title']} (Handle: {best['handle_id']})")

        act = scraper.scrape_act(best["handle_id"], law_name)
        scraper.save_act(act)

        print(f"  Total sections scraped: {len(act.sections)}")


if __name__ == "__main__":
    scrape_target_laws()
