"""India Code (indiacode.nic.in) scraper for Bare Act sections.

Scrapes central acts from India Code website, extracting section numbers,
titles, and full text. Handles server errors gracefully and continues
to the next law on failure.
"""

import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from ...config import config


def _print(msg: str, flush: bool = True):
    """Print with automatic flush for real-time output."""
    print(msg, flush=flush)


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
    """Scraper for India Code (indiacode.nic.in).

    Resilient design:
    - 3s delay between requests (configurable)
    - 5 retries with exponential backoff (1s, 2s, 4s, 8s, 16s)
    - 500/502/503 errors caught and retried
    - Individual section failures logged and skipped
    - Individual law failures logged and skipped (continues to next law)
    """

    BASE_URL = config.india_code_base

    def __init__(self, delay: float = None, verbose: bool = False):
        self.delay = delay or 3.0
        self.verbose = verbose
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "THEMIS-Legal-AI/1.0 (Academic Research)",
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en-US,en;q=0.9",
        })

    def _get(self, url: str, retries: int = 5, params: dict = None) -> requests.Response:
        """GET request with retry, exponential backoff, and rate limiting."""
        for attempt in range(retries):
            try:
                time.sleep(self.delay)
                resp = self.session.get(url, params=params, timeout=30)

                if resp.status_code >= 500:
                    wait = min(2 ** (attempt + 1), 30)
                    _print(f"  Retry {attempt + 1}/{retries} after {wait}s: "
                           f"HTTP {resp.status_code}")
                    time.sleep(wait)
                    continue

                resp.raise_for_status()
                return resp

            except requests.exceptions.Timeout:
                wait = min(2 ** (attempt + 1), 30)
                _print(f"  Retry {attempt + 1}/{retries} after {wait}s: Timeout")
                time.sleep(wait)

            except requests.ConnectionError:
                wait = min(2 ** (attempt + 1), 30)
                _print(f"  Retry {attempt + 1}/{retries} after {wait}s: Connection error")
                time.sleep(wait)

            except requests.RequestException as e:
                if attempt == retries - 1:
                    raise
                wait = min(2 ** (attempt + 1), 30)
                _print(f"  Retry {attempt + 1}/{retries} after {wait}s: {e}")
                time.sleep(wait)

        raise RuntimeError(f"Failed after {retries} retries for {url}")

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
        for link in soup.find_all("a", class_="title"):
            href = link.get("href", "")
            text = link.get_text(strip=True)
            match = re.search(r"sectionno=(\d+)", href)
            if match:
                section_no = match.group(1)
                act_id_match = re.search(r"actid=([^&]+)", href)
                section_id_match = re.search(r"sectionId=(\d+)", href)
                act_id = act_id_match.group(1) if act_id_match else ""
                section_id = section_id_match.group(1) if section_id_match else ""
                title_match = re.search(r"Section \d+\.\s*(.*)", text)
                title = title_match.group(1).strip() if title_match else text
                sections.append({
                    "section_number": section_no,
                    "title": title,
                    "url": urljoin(self.BASE_URL, href),
                    "act_id": act_id,
                    "section_id": section_id,
                })
        return sections

    def extract_chapters(self, soup: BeautifulSoup) -> dict[str, str]:
        """Extract chapter structure mapping section ranges to chapter names."""
        chapters = {}
        for li in soup.find_all("li"):
            b = li.find("b")
            if b and "CHAPTER" in b.text:
                chapter_name = b.text.strip()
                sub_ul = li.find("ul")
                if sub_ul:
                    for a in sub_ul.find_all("a", class_="headingtwo"):
                        heading = a.text.strip()
                        chapters[heading] = chapter_name
        return chapters

    def fetch_section_text(self, section_url: str, act_id: str = "", section_id: str = "") -> str:
        """Fetch the full text of a single section via AJAX endpoint."""
        try:
            ajax_url = f"{self.BASE_URL}/SectionPageContent"
            params = {"actid": act_id, "sectionID": section_id}
            resp = self._get(ajax_url, params=params)

            try:
                data = resp.json()
                content = data.get("content", "")
                footnote = data.get("footnote", "")
                full_text = content
                if footnote:
                    full_text += "\n\n" + footnote
                if full_text:
                    soup = BeautifulSoup(full_text, "html.parser")
                    return soup.get_text(separator="\n", strip=True)
            except Exception:
                soup = BeautifulSoup(resp.text, "html.parser")
                return soup.get_text(separator="\n", strip=True)

        except Exception as e:
            _print(f"    Warning: Failed to fetch section text: {e}")
        return ""

    def scrape_act(self, handle_id: str, act_name: str = "") -> Act | None:
        """Scrape a complete act by its handle ID.

        Returns None if the act page cannot be fetched.
        Individual section failures are logged and skipped.
        """
        _print(f"Scraping act: {handle_id}")

        try:
            soup = self.get_act_page(handle_id)
        except Exception as e:
            _print(f"  ERROR: Could not fetch act page for {handle_id}: {e}")
            _print(f"  Skipping this act.")
            return None

        title_tag = soup.find("td", id="short_title")
        title = title_tag.text.strip() if title_tag else act_name

        year_match = re.search(r"(\d{4})", title)
        year = year_match.group(1) if year_match else ""

        act_id_tag = soup.find("td", class_="metadataFieldValue", string=re.compile(r"\d{4}-\d+"))
        act_id = act_id_tag.text.strip() if act_id_tag else handle_id

        act = Act(
            name=title,
            year=year,
            act_id=act_id,
            handle_id=handle_id,
        )

        section_links = self.extract_section_links(soup)
        total = len(section_links)
        _print(f"  Found {total} sections")

        if not section_links:
            _print(f"  Warning: No sections found for {title}")
            return act

        # Fetch each section
        failed_sections = 0
        total_chars = 0
        for i, sec_info in enumerate(section_links):
            sec_num = sec_info["section_number"]
            try:
                text = self.fetch_section_text(
                    sec_info["url"],
                    act_id=sec_info.get("act_id", ""),
                    section_id=sec_info.get("section_id", ""),
                )
                section = Section(
                    section_number=sec_num,
                    title=sec_info["title"],
                    text=text,
                    act_name=title,
                )
                act.sections.append(section)
                total_chars += len(text)

                # Show snippet for first 3 sections and every 50th after
                if self.verbose and (i < 3 or (i + 1) % 50 == 0):
                    snippet = text[:120].replace("\n", " ") if text else "(empty)"
                    _print(f"    [{sec_num}] {sec_info['title'][:50]}")
                    _print(f"         {snippet}...")

            except Exception as e:
                failed_sections += 1
                _print(f"    Skipping section {sec_num}: {e}")

            # Progress every 10 sections
            done = i + 1
            if done % 10 == 0 or done == total:
                pct = (done / total) * 100
                char_k = total_chars / 1024
                _print(f"  [{done}/{total}] {pct:.0f}% | "
                       f"collected: {char_k:.1f} KB | "
                       f"failed: {failed_sections}")

        _print(f"  Done: {len(act.sections)}/{total} sections | "
               f"{total_chars/1024:.1f} KB collected"
               + (f" | {failed_sections} failed" if failed_sections else ""))
        return act

    def save_act(self, act: Act, output_dir: Path = None):
        """Save act data to JSON."""
        output_dir = output_dir or config.raw_dir
        output_dir.mkdir(parents=True, exist_ok=True)

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

        _print(f"  Saved to {filepath}")
        return filepath


def _law_to_filename(law_name: str) -> str:
    """Convert law name to JSON filename."""
    safe = re.sub(r"[^\w\s-]", "", law_name.lower())
    safe = re.sub(r"\s+", "_", safe.strip())
    return f"{safe}.json"


def _is_already_scraped(law_name: str, raw_dir: Path) -> bool:
    """Check if a law has already been scraped.

    Handles filename mismatches (e.g., 'the_' prefix from website titles).
    """
    expected = _law_to_filename(law_name)

    # Exact match
    if (raw_dir / expected).exists():
        return True

    # Match with 'the_' prefix (website titles include "The")
    the_filename = "the_" + expected
    if (raw_dir / the_filename).exists():
        return True

    # Partial match: check if any file contains key words from the law name
    words = [w for w in law_name.lower().replace(",", "").split() if len(w) > 3]
    for f in raw_dir.glob("*.json"):
        if all(w in f.name for w in words):
            return True

    return False


def scrape_target_laws(force: bool = False):
    """Scrape all target laws for THEMIS training.

    Resilient: if one law fails (server error, timeout, etc.),
    it logs the error and continues to the next law.
    Skips laws already scraped unless force=True.
    """
    scraper = IndiaCodeScraper(delay=3.0, verbose=True)

    succeeded = []
    skipped = []
    failed = []

    for law_name in config.target_laws:
        _print(f"\n{'='*60}")
        _print(f"Searching for: {law_name}")
        _print("=" * 60)

        # Check if already scraped
        if _is_already_scraped(law_name, config.raw_dir):
            _print(f"  Already scraped: {law_name}")
            _print(f"  Use --force to re-scrape")
            skipped.append(law_name)
            continue

        try:
            results = scraper.search_act(law_name)
            if not results:
                _print(f"  No results found for '{law_name}'")
                failed.append((law_name, "No search results"))
                continue

            best = results[0]
            _print(f"  Found: {best['short_title']} (Handle: {best['handle_id']})")

            act = scraper.scrape_act(best["handle_id"], law_name)
            if act is None:
                failed.append((law_name, "Could not fetch act page"))
                continue

            if not act.sections:
                failed.append((law_name, "No sections scraped"))
                continue

            scraper.save_act(act)
            succeeded.append((law_name, len(act.sections)))

        except Exception as e:
            _print(f"  ERROR scraping {law_name}: {e}")
            failed.append((law_name, str(e)))
            continue

    # Summary
    _print(f"\n{'='*60}")
    _print("SCRAPING SUMMARY")
    _print("=" * 60)
    if succeeded:
        _print(f"Scraped: {len(succeeded)}")
        for name, count in succeeded:
            _print(f"  OK  {name}: {count} sections")
    if skipped:
        _print(f"Skipped (already scraped): {len(skipped)}")
        for name in skipped:
            _print(f"  --  {name}")
    if failed:
        _print(f"Failed: {len(failed)}")
        for name, reason in failed:
            _print(f"  FAIL  {name}: {reason}")
    _print("=" * 60)


if __name__ == "__main__":
    scrape_target_laws()
