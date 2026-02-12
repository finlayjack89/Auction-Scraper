#!/usr/bin/env python
"""
Streamlit UI with live agent progress for the Fast Auction Research CrewAI workflow.

Multi-crew architecture with per-lot token isolation:
  Phase 1a: Catalog extraction (Scout outputs ALL lots as JSON)
  Python:   Deterministic keyword filtering
  Phase 1b: Risk assessment + detail extraction (on filtered lots only)
  Phase 2:  Per-lot market validation (fresh crew per lot)
  Phase 3:  Per-lot deep research (fresh crew per lot, top 40% only)
  Phase 4:  Synthesis — ranking, archive, bidding sheet

Launch with:
    uv run streamlit run app.py
"""

import sys
import os
import io
import re
import json
import math
import time
import asyncio
import threading
from datetime import datetime
from dataclasses import dataclass, field
from urllib.parse import urlparse, urlunparse, urlencode, parse_qs, urljoin

# ── Project root & src/ on sys.path ────────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.join(_PROJECT_ROOT, "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

try:
    from dotenv import load_dotenv
    from crewai import Crew  # noqa: F401
except ModuleNotFoundError as _e:
    print(
        "\n*** ModuleNotFoundError: " + str(_e) + " ***\n"
        "Please run:  uv run streamlit run app.py\n"
    )
    sys.exit(1)

load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

import streamlit as st
import pandas as pd

# ═══════════════════════════════════════════════════════════════════════════
# AGENT ROLE MAP
# ═══════════════════════════════════════════════════════════════════════════

AGENT_ROLE_MAP = {
    "Scout - Auction Catalog Extractor":                    "Scout",
    "Scout - Auction Navigator & Keyword Filter":           "Scout",
    "Risk Officer - Compliance & Filtration Specialist":    "Risk Officer",
    "Extractor - Item Detail Parser":                       "Extractor",
    "Market Validator - Rapid Assessment Specialist":       "Market Validator",
    "Quant - Financial Analysis Specialist":                "Quant",
    "Archivist - Data Curator & Storage Specialist":        "Archivist",
    "Deep Research Analyst - Comprehensive Market Intelligence": "Deep Research",
    "Mobile Report Generator - Bidding Sheet Formatter":    "Report Generator",
}


# ═══════════════════════════════════════════════════════════════════════════
# PROGRESS TRACKER (thread-safe, phase-aware, allowlist-gated lots)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class LotInfo:
    lot_num: str
    title: str = ""
    keyword: str = ""
    status: str = "found"          # found | passed | rejected | validating | researching | complete
    rejection_reason: str = ""
    stage: str = ""
    fmv: str = ""
    margin: str = ""
    recommendation: str = ""


@dataclass
class Insight:
    lot: str
    text: str
    agent: str = ""
    timestamp: str = ""


# Ordered list of pipeline phases for the stage tracker
PIPELINE_PHASES = [
    {"key": "1a", "label": "Extract Catalog", "icon": "1", "detail": "Scout scrapes all lots"},
    {"key": "filter", "label": "Keyword Filter", "icon": "2", "detail": "Python deterministic filter"},
    {"key": "1b", "label": "Risk Assessment", "icon": "3", "detail": "Red flag screening"},
    {"key": "2", "label": "Market Validation", "icon": "4", "detail": "Per-lot market checks"},
    {"key": "3", "label": "Deep Research", "icon": "5", "detail": "Top 40% deep dive"},
    {"key": "4", "label": "Synthesis", "icon": "6", "detail": "Rank, archive, report"},
]

# Map internal phase strings to pipeline keys
PHASE_KEY_MAP = {
    "Initializing": None,
    "Phase 1a: Extracting Catalog": "1a",
    "Keyword Filtering": "filter",
    "Phase 1b: Risk Assessment": "1b",
    "Phase 2: Market Validation": "2",
    "Phase 3: Deep Research": "3",
    "Phase 4: Synthesis": "4",
}


class ProgressTracker:
    """Accumulates execution state; read by the UI polling loop."""

    def __init__(self):
        self._lock = threading.Lock()

        # Phase tracking
        self.current_phase: str = "Initializing"
        self.phase_lot_current: int = 0
        self.phase_lot_total: int = 0
        self.phase_step: str = ""

        # Timing for ETA
        self.phase_start_time: float = 0.0
        self.lot_times: list[float] = []  # seconds per lot, for rolling average

        # Current agent state
        self.current_agent: str = ""
        self.current_task_desc: str = ""
        self.current_thought: str = ""
        self.current_tool: str = ""
        self.current_tool_input: str = ""

        # Lot tracking (allowlist-gated)
        self._known_lots: set[str] = set()
        self.lots: dict[str, LotInfo] = {}

        # Research insights
        self.insights: list[Insight] = []

        # Activity log
        self.log: list[tuple[str, str, str]] = []

        # Final result holder
        self.finished: bool = False
        self.result: object = None
        self.error: Exception | None = None

        # Phase completion tracking for summary stats
        self.total_catalog_lots: int = 0
        self.total_filtered_lots: int = 0
        self.total_risk_passed: int = 0
        self.total_validated: int = 0
        self.total_researched: int = 0
        self.workflow_start: float = 0.0

        # Extraction progress (Phase 1a specific)
        self.extraction_pages_scraped: int = 0
        self.extraction_total_pages: int = 0
        self.extraction_lots_found: int = 0
        self.extraction_current_page: str = ""
        self.extraction_last_lot: str = ""

    # -- helpers ----------------------------------------------------------

    def _ts(self) -> str:
        return datetime.now().strftime("%H:%M:%S")

    def add_log(self, agent: str, msg: str):
        with self._lock:
            self.log.append((self._ts(), agent, msg))

    def set_agent(self, agent_name: str, task_desc: str = ""):
        with self._lock:
            self.current_agent = agent_name
            if task_desc:
                self.current_task_desc = task_desc
            self.current_thought = ""
            self.current_tool = ""
            self.current_tool_input = ""

    def set_thought(self, thought: str):
        with self._lock:
            self.current_thought = thought

    def set_tool(self, tool: str, tool_input: str = ""):
        with self._lock:
            self.current_tool = tool
            self.current_tool_input = tool_input

    def set_phase(self, phase: str, lot_current: int = 0, lot_total: int = 0, step: str = ""):
        with self._lock:
            self.current_phase = phase
            self.phase_lot_current = lot_current
            self.phase_lot_total = lot_total
            self.phase_step = step
            if lot_current <= 1:
                self.phase_start_time = time.time()
                self.lot_times = []

    def record_lot_time(self, elapsed: float):
        """Record how long a single lot took for ETA calculations."""
        with self._lock:
            self.lot_times.append(elapsed)

    def get_eta_seconds(self) -> float | None:
        """Estimate remaining time based on rolling average of lot processing times."""
        with self._lock:
            if not self.lot_times or self.phase_lot_total == 0:
                return None
            remaining = self.phase_lot_total - self.phase_lot_current
            if remaining <= 0:
                return 0.0
            # Use recent lot times (last 5) for better accuracy
            recent = self.lot_times[-5:] if len(self.lot_times) >= 5 else self.lot_times
            avg = sum(recent) / len(recent)
            return avg * remaining

    # -- lot helpers (allowlist-gated) ------------------------------------

    def register_filtered_lots(self, lots_data: list[dict]):
        """Register the canonical set of keyword-filtered lots. Only these can appear in UI."""
        with self._lock:
            for lot in lots_data:
                lot_num = str(lot.get("lot_number", lot.get("lot_num", "")))
                if not lot_num:
                    continue
                self._known_lots.add(lot_num)
                if lot_num not in self.lots:
                    self.lots[lot_num] = LotInfo(
                        lot_num=lot_num,
                        title=lot.get("title", "")[:80],
                        status="found",
                        stage="Filtered",
                    )

    def update_lot(self, lot_num: str, **kwargs):
        """Update an existing known lot. Cannot create new lots."""
        with self._lock:
            if lot_num not in self._known_lots:
                return
            if lot_num not in self.lots:
                return
            lot = self.lots[lot_num]
            for k, v in kwargs.items():
                if v and hasattr(lot, k):
                    setattr(lot, k, v)

    def remove_lot(self, lot_num: str, reason: str = ""):
        """Mark a lot as rejected (Risk Officer removed it)."""
        with self._lock:
            if lot_num in self.lots:
                self.lots[lot_num].status = "rejected"
                self.lots[lot_num].rejection_reason = reason

    def update_extraction(self, **kwargs):
        """Update extraction progress fields (Phase 1a)."""
        with self._lock:
            for k, v in kwargs.items():
                if v is not None and hasattr(self, k):
                    setattr(self, k, v)

    def add_insight(self, lot: str, text: str, agent: str = ""):
        with self._lock:
            for existing in self.insights[-20:]:
                if existing.text[:60] == text[:60]:
                    return
            self.insights.append(Insight(lot=lot, text=text, agent=agent, timestamp=self._ts()))


# ═══════════════════════════════════════════════════════════════════════════
# STDOUT INTERCEPTOR (captures agent/tool metadata only — no lot parsing)
# ═══════════════════════════════════════════════════════════════════════════

class OutputInterceptor(io.TextIOBase):
    """Wraps sys.stdout to capture CrewAI verbose output for the tracker."""

    ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')

    def __init__(self, tracker: ProgressTracker, original):
        self.tracker = tracker
        self.original = original
        self._buffer = ""

    def write(self, text):
        if self.original:
            self.original.write(text)
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._process_line(line)
        return len(text)

    def flush(self):
        if self.original:
            self.original.flush()

    @property
    def encoding(self):
        return getattr(self.original, 'encoding', 'utf-8')

    def isatty(self):
        return False

    def _clean(self, text: str) -> str:
        return self.ANSI_RE.sub('', text).strip()

    def _process_line(self, raw_line: str):
        line = self._clean(raw_line)
        if not line:
            return

        t = self.tracker

        # Detect agent started
        m = re.match(r'Agent:\s*(.+)', line)
        if m:
            role_raw = m.group(1).strip()
            friendly = AGENT_ROLE_MAP.get(role_raw, role_raw)
            t.set_agent(friendly)
            t.add_log(friendly, "Agent activated")
            return

        # Detect task description
        m = re.match(r'Task:\s*(.+)', line)
        if m:
            desc = m.group(1).strip()
            with t._lock:
                t.current_task_desc = desc[:250]
            # During Phase 1a, update the sub-step label based on which task is active
            if t.current_phase == "Phase 1a: Extracting Catalog":
                desc_lower = desc.lower()
                if "validation" in desc_lower or "validate" in desc_lower:
                    with t._lock:
                        t.phase_step = "Validating auction URL..."
                elif "buyer" in desc_lower or "premium" in desc_lower:
                    with t._lock:
                        t.phase_step = "Discovering buyer premium..."
                elif "catalog" in desc_lower or "bulk" in desc_lower or "extraction" in desc_lower:
                    with t._lock:
                        t.phase_step = "Scraping catalog pages..."
            return

        # Detect tool started
        m = re.match(r'Tool:\s*(\S+)', line)
        if m:
            tool_name = m.group(1).strip()
            t.set_tool(tool_name)
            t.add_log(t.current_agent, f"Using tool: {tool_name}")
            return

        # Detect tool args
        m = re.match(r'Args:\s*(.+)', line)
        if m:
            args_str = m.group(1).strip()
            with t._lock:
                t.current_tool_input = args_str[:300]
            # During Phase 1a catalog extraction, detect page from URL in tool args
            if t.current_phase == "Phase 1a: Extracting Catalog" and t.current_tool in _SCRAPING_TOOLS:
                _parse_extraction_tool(t, t.current_tool, args_str)
            return

        # NO parse_content here — this was the main leak causing phantom lots


# ═══════════════════════════════════════════════════════════════════════════
# EXTRACTION PROGRESS HELPERS (Phase 1a — page & lot detection)
# ═══════════════════════════════════════════════════════════════════════════

_SCRAPING_TOOLS = frozenset({
    "FirecrawlScrapeWebsiteTool",
    "ScrapeWebsiteTool",
    "ScrapeElementFromWebsiteTool",
    "SerperScrapeWebsiteTool",
})


def _extract_page_from_url(url: str) -> int | None:
    """Extract page number from common auction pagination URL patterns."""
    # ?page=N, ?p=N, ?pageNo=N, ?page_no=N
    m = re.search(r'[?&](?:page|p|pageNo|page_no)=(\d+)', url)
    if m:
        return int(m.group(1))
    # /page/N
    m = re.search(r'/page/(\d+)', url)
    if m:
        return int(m.group(1))
    # ?offset=N or ?start=N (assume ~60 lots per page)
    m = re.search(r'[?&](?:offset|start)=(\d+)', url)
    if m:
        offset = int(m.group(1))
        if offset > 0:
            return (offset // 60) + 1
    return None


def _parse_extraction_tool(tracker: 'ProgressTracker', tool: str, tool_input: str):
    """Detect page being scraped from a tool call URL during Phase 1a."""
    if tool not in _SCRAPING_TOOLS:
        return
    url_m = re.search(r'https?://[^\s"\'}\]]+', tool_input)
    if not url_m:
        return
    url = url_m.group(0)
    page = _extract_page_from_url(url)
    if page is not None:
        tracker.update_extraction(
            extraction_current_page=f"Page {page}",
            extraction_pages_scraped=max(tracker.extraction_pages_scraped, page),
        )
        tracker.add_log("Scout", f"Scraping page {page}")
    elif tracker.extraction_pages_scraped == 0:
        # First scrape call — no page param means page 1
        tracker.update_extraction(
            extraction_current_page="Page 1",
            extraction_pages_scraped=1,
        )
        tracker.add_log("Scout", "Scraping page 1")


def _parse_extraction_thought(tracker: 'ProgressTracker', thought: str):
    """Parse agent thoughts/output during Phase 1a for page & lot counts."""
    text = thought.lower()

    # "TOTAL LOTS EXTRACTED: 245" or "extracted 120 lots"
    m = re.search(r'total\s+lots?\s+extracted\s*:\s*(\d+)', text)
    if m:
        tracker.update_extraction(extraction_lots_found=int(m.group(1)))
        return

    # "N lots on page", "N lots from page", "extracted N lots", "found N lots", "parsed N lots"
    m = re.search(r'(?:extracted|found|parsed|scraped|got)\s+(\d+)\s+lots?', text)
    if m:
        count = int(m.group(1))
        if count > tracker.extraction_lots_found:
            tracker.update_extraction(extraction_lots_found=count)

    # "N lots" at sentence level (standalone mention of lot count)
    m = re.search(r'(\d+)\s+lots?\s+(?:total|across|in total|altogether|so far)', text)
    if m:
        count = int(m.group(1))
        if count > tracker.extraction_lots_found:
            tracker.update_extraction(extraction_lots_found=count)

    # "page N of M"
    m = re.search(r'page\s+(\d+)\s+of\s+(\d+)', text)
    if m:
        page_num = int(m.group(1))
        total_pages = int(m.group(2))
        tracker.update_extraction(
            extraction_current_page=f"Page {page_num}",
            extraction_pages_scraped=max(tracker.extraction_pages_scraped, page_num),
            extraction_total_pages=max(tracker.extraction_total_pages, total_pages),
        )

    # "M pages total" or "total of M pages"
    m = re.search(r'(\d+)\s+pages?\s+(?:total|in total)', text)
    if m:
        tracker.update_extraction(extraction_total_pages=int(m.group(1)))

    # Detect last lot seen: "lot 45", "lot number 45"
    for lot_m in re.finditer(r'lot\s+(?:number\s+)?(\d+)', text):
        tracker.update_extraction(extraction_last_lot=f"Lot {lot_m.group(1)}")


# ═══════════════════════════════════════════════════════════════════════════
# CALLBACK HANDLERS
# ═══════════════════════════════════════════════════════════════════════════

def make_step_callback(tracker: ProgressTracker):
    """Returns a step_callback function for a Crew."""
    from crewai.agents.parser import AgentAction, AgentFinish

    def step_callback(step_output):
        try:
            is_extraction = (tracker.current_phase == "Phase 1a: Extracting Catalog")

            if isinstance(step_output, AgentAction):
                thought = step_output.thought or ""
                tool = step_output.tool or ""
                tool_input = step_output.tool_input or ""

                # Display thought (but suppress parse failure message)
                if thought and thought != "Failed to parse LLM response":
                    tracker.set_thought(thought[:500])
                    # During extraction, parse thoughts for page/lot info
                    if is_extraction:
                        _parse_extraction_thought(tracker, thought)
                if tool:
                    tracker.set_tool(tool, str(tool_input)[:300])
                    tracker.add_log(tracker.current_agent, f"Tool: {tool}")
                    # During extraction, parse scraping tool URLs for page info
                    if is_extraction:
                        _parse_extraction_tool(tracker, tool, str(tool_input))

                # NO parse_content — prevents phantom lots from tool results

            elif isinstance(step_output, AgentFinish):
                thought = step_output.thought or ""
                # Suppress the benign "Failed to parse LLM response" message
                if thought and thought != "Failed to parse LLM response":
                    tracker.set_thought(thought[:500])
                    # During extraction, parse final answer for lot/page info
                    if is_extraction:
                        _parse_extraction_thought(tracker, thought)
                tracker.add_log(tracker.current_agent, "Reached final answer")

        except Exception:
            pass

    return step_callback


def make_task_callback(tracker: ProgressTracker):
    """Returns a task_callback function for a Crew."""

    def task_callback(task_output):
        try:
            agent_role = getattr(task_output, 'agent', '') or ''
            task_name = getattr(task_output, 'name', '') or ''
            friendly = AGENT_ROLE_MAP.get(agent_role, agent_role)
            tracker.add_log(
                friendly or tracker.current_agent,
                f"Task completed: {task_name[:80]}"
            )

            # During Phase 1a, extract pagination info from validation task output
            if tracker.current_phase == "Phase 1a: Extracting Catalog":
                raw = getattr(task_output, 'raw', '') or ''
                # estimated_total_pages from the validation task
                m = re.search(r'estimated_total_pages\s*:\s*(\d+)', raw)
                if m:
                    tracker.update_extraction(extraction_total_pages=int(m.group(1)))
                # estimated total lots
                m = re.search(r'estimated_lots\s*:\s*(\d+)', raw)
                if m:
                    tracker.add_log("Scout", f"Auction reports ~{m.group(1)} total lots")
        except Exception:
            pass

    return task_callback


# ═══════════════════════════════════════════════════════════════════════════
# PROGRAMMATIC KEYWORD FILTER (deterministic Python — no LLM)
# ═══════════════════════════════════════════════════════════════════════════

def _phrase_matches(phrase: str, text: str) -> bool:
    """Check if a phrase matches text. All words in the phrase must appear."""
    words = phrase.lower().split()
    return all(word in text for word in words)


def apply_keyword_filter(
    lots: list[dict],
    phrases: list[str],
) -> list[dict]:
    """
    Deterministic keyword filter applied in Python.

    A lot PASSES if it matches ANY one of the phrases.
    A phrase matches if ALL words in the phrase appear in the lot's title+description.

    Examples:
      phrases = ["Chanel", "Bottega Veneta", "Dior"]
      → lot passes if it contains "Chanel"
        OR (contains "Bottega" AND "Veneta")
        OR contains "Dior"
    """
    if not phrases:
        return lots

    filtered = []
    for lot in lots:
        text = f"{lot.get('title', '')} {lot.get('description', '')}".lower()

        if any(_phrase_matches(phrase, text) for phrase in phrases):
            filtered.append(lot)

    return filtered


# ═══════════════════════════════════════════════════════════════════════════
# JSON PARSING HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _repair_truncated_json_array(text: str) -> list | None:
    """Attempt to recover a truncated JSON array by finding the last complete object.

    If the LLM output was cut off mid-JSON (e.g. the closing ] is missing, or an
    object is only half-written), this finds the last complete {...} and closes the
    array.  Returns a list of dicts, or None if nothing useful could be recovered.
    """
    # Find the opening bracket
    start = text.find("[")
    if start == -1:
        return None

    # Try to find the last complete JSON object (ends with })
    last_brace = text.rfind("}")
    if last_brace == -1 or last_brace <= start:
        return None

    candidate = text[start:last_brace + 1].rstrip().rstrip(",") + "]"
    try:
        result = json.loads(candidate)
        if isinstance(result, list) and len(result) > 0:
            print(f"[JSON-REPAIR] Recovered {len(result)} items from truncated JSON array")
            return result
    except (json.JSONDecodeError, TypeError):
        pass

    # More aggressive: iterate backwards to find a valid cut point
    search_from = last_brace
    for _ in range(50):  # try up to 50 positions backwards
        prev_brace = text.rfind("}", start, search_from)
        if prev_brace == -1 or prev_brace <= start:
            break
        candidate = text[start:prev_brace + 1].rstrip().rstrip(",") + "]"
        try:
            result = json.loads(candidate)
            if isinstance(result, list) and len(result) > 0:
                print(f"[JSON-REPAIR] Recovered {len(result)} items (aggressive) from truncated JSON")
                return result
        except (json.JSONDecodeError, TypeError):
            pass
        search_from = prev_brace

    return None


def _extract_json_from_output(raw: str) -> list | dict | None:
    """Extract JSON from a crew output, handling ```json code fences and truncation."""
    if not raw:
        return None

    # Try to find JSON inside ```json ... ``` code fence
    m = re.search(r'```json\s*\n?(.*?)```', raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except (json.JSONDecodeError, TypeError):
            # Fence found but JSON broken — try repair
            repaired = _repair_truncated_json_array(m.group(1))
            if repaired:
                return repaired

    # Handle ```json fence without closing ``` (truncation mid-fence)
    m = re.search(r'```json\s*\n?(.*)', raw, re.DOTALL)
    if m:
        repaired = _repair_truncated_json_array(m.group(1))
        if repaired:
            return repaired

    # Try to find a bare JSON array or object
    for pattern in [r'(\[.*\])', r'(\{.*\})']:
        m = re.search(pattern, raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except (json.JSONDecodeError, TypeError):
                pass

    # Try the whole string
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        pass

    # Final fallback: try repair on the raw string
    repaired = _repair_truncated_json_array(raw)
    if repaired:
        return repaired

    return None


def parse_lots_from_page_output(result) -> list[dict]:
    """Extract lot array from a single-page extraction crew output."""
    raw = getattr(result, "raw", str(result))

    # Try the last task output (there's only one task in PageExtractionCrew)
    task_outputs = getattr(result, "tasks_output", None)
    if task_outputs:
        for t_out in reversed(task_outputs):
            t_raw = getattr(t_out, "raw", "")
            parsed = _extract_json_from_output(t_raw)
            if isinstance(parsed, list) and len(parsed) > 0:
                return parsed

    # Fall back to main output
    parsed = _extract_json_from_output(raw)
    if isinstance(parsed, list):
        return parsed

    print(f"[WARN] parse_lots_from_page_output: could not parse lots. Raw output length={len(raw)}")
    print(f"[DEBUG] First 500 chars: {raw[:500]}")
    return []


def extract_buyer_premium(result) -> str:
    """Extract buyer premium data string from CatalogSetupCrew output (task index 1)."""
    task_outputs = getattr(result, "tasks_output", None)
    if task_outputs and len(task_outputs) >= 2:
        return getattr(task_outputs[1], "raw", "")
    # Fallback: try last task
    if task_outputs:
        return getattr(task_outputs[-1], "raw", "")
    return ""


# ═══════════════════════════════════════════════════════════════════════════
# PAGINATION URL CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════

def _parse_pagination_info(validation_raw: str) -> dict:
    """Parse pagination fields from the validation task output.

    Returns dict with keys: estimated_lots, lots_per_page, pagination_detected,
    pagination_url_pattern, page_2_url, total_pages.
    """
    info: dict = {
        "estimated_lots": 0,
        "lots_per_page": 0,
        "pagination_detected": False,
        "pagination_url_pattern": "",
        "page_2_url": "",
        "total_pages": 1,
    }
    if not validation_raw:
        return info

    text = validation_raw

    # estimated_lots
    m = re.search(r'estimated_lots\s*[:=]\s*(\d+)', text, re.IGNORECASE)
    if m:
        info["estimated_lots"] = int(m.group(1))

    # lots_per_page
    m = re.search(r'lots_per_page\s*[:=]\s*(\d+)', text, re.IGNORECASE)
    if m:
        info["lots_per_page"] = int(m.group(1))

    # pagination_detected
    m = re.search(r'pagination_detected\s*[:=]\s*(true|false|yes|no)', text, re.IGNORECASE)
    if m:
        info["pagination_detected"] = m.group(1).lower() in ("true", "yes")

    # total_pages
    m = re.search(r'total_pages\s*[:=]\s*(\d+)', text, re.IGNORECASE)
    if m:
        info["total_pages"] = int(m.group(1))

    # pagination_url_pattern (e.g. "?page=N", "?offset=N")
    m = re.search(r'pagination_url_pattern\s*[:=]\s*["\']?([^"\'}\n,]+)', text, re.IGNORECASE)
    if m:
        info["pagination_url_pattern"] = m.group(1).strip().strip("\"'")

    # page_2_url (full URL for page 2)
    m = re.search(r'page_2_url\s*[:=]\s*["\']?(https?://[^\s"\'}\n,]+)', text, re.IGNORECASE)
    if m:
        info["page_2_url"] = m.group(1).strip().strip("\"'")

    # Infer total_pages from estimated_lots / lots_per_page if not reported
    if info["total_pages"] <= 1 and info["estimated_lots"] > 0 and info["lots_per_page"] > 0:
        info["total_pages"] = math.ceil(info["estimated_lots"] / info["lots_per_page"])

    return info


def _detect_pattern_from_page2_url(base_url: str, page2_url: str) -> str | None:
    """Detect the pagination URL pattern by comparing page 1 and page 2 URLs."""
    if not page2_url:
        return None

    # Check for ?page=2 / &page=2
    if "page=2" in page2_url.lower():
        return "page"
    if "pageno=2" in page2_url.lower():
        return "pageNo"
    if "page_no=2" in page2_url.lower():
        return "page_no"
    if "p=2" in page2_url.lower().split("?")[-1]:
        return "p"

    # Check for offset
    m = re.search(r'[?&](?:offset|start)=(\d+)', page2_url)
    if m:
        return f"offset={m.group(1)}"

    # Check for /page/2 path pattern
    if "/page/2" in page2_url:
        return "path_page"

    return None


def construct_page_urls(base_url: str, pagination_info: dict) -> list[str]:
    """Construct a list of all page URLs to scrape, including page 1.

    Returns a list of URLs: [page1_url, page2_url, page3_url, ...]
    Uses the pagination info from the validation task to construct URLs.
    Falls back to common patterns if detection is unreliable.
    """
    total_pages = max(1, pagination_info.get("total_pages", 1))
    page2_url = pagination_info.get("page_2_url", "")
    pattern_str = pagination_info.get("pagination_url_pattern", "")

    # Page 1 is always the base URL
    urls = [base_url]

    if total_pages <= 1:
        return urls

    # Detect the pattern from page_2_url or the reported pattern string
    pattern = _detect_pattern_from_page2_url(base_url, page2_url)

    if pattern is None and pattern_str:
        # Try to infer from the pattern string
        pattern_lower = pattern_str.lower()
        if "page=n" in pattern_lower or "?page=" in pattern_lower:
            pattern = "page"
        elif "offset=" in pattern_lower:
            pattern = "offset_from_pattern"
        elif "/page/" in pattern_lower:
            pattern = "path_page"
        elif "pageno=" in pattern_lower:
            pattern = "pageNo"
        elif "p=" in pattern_lower:
            pattern = "p"

    # If we have a concrete page_2_url but couldn't detect the pattern,
    # use the page_2_url directly for page 2 and try query param substitution for rest
    if pattern is None and page2_url:
        urls.append(page2_url)
        # Try to construct subsequent pages by incrementing the number in page2_url
        for pg in range(3, total_pages + 1):
            # Replace "2" with pg in the last occurrence of "2" in query string
            candidate = re.sub(r'(\d+)(?=[^/\d]*$)', str(pg), page2_url, count=1)
            if candidate != page2_url:
                urls.append(candidate)
            else:
                break
        return urls

    # Build URLs based on the detected pattern
    parsed = urlparse(base_url)
    lots_per_page = pagination_info.get("lots_per_page", 60) or 60

    for pg in range(2, total_pages + 1):
        if pattern == "page":
            qs = parse_qs(parsed.query, keep_blank_values=True)
            qs["page"] = [str(pg)]
            new_query = urlencode(qs, doseq=True)
            url = urlunparse(parsed._replace(query=new_query))
        elif pattern == "pageNo":
            qs = parse_qs(parsed.query, keep_blank_values=True)
            qs["pageNo"] = [str(pg)]
            new_query = urlencode(qs, doseq=True)
            url = urlunparse(parsed._replace(query=new_query))
        elif pattern == "page_no":
            qs = parse_qs(parsed.query, keep_blank_values=True)
            qs["page_no"] = [str(pg)]
            new_query = urlencode(qs, doseq=True)
            url = urlunparse(parsed._replace(query=new_query))
        elif pattern == "p":
            qs = parse_qs(parsed.query, keep_blank_values=True)
            qs["p"] = [str(pg)]
            new_query = urlencode(qs, doseq=True)
            url = urlunparse(parsed._replace(query=new_query))
        elif pattern and pattern.startswith("offset"):
            offset = (pg - 1) * lots_per_page
            qs = parse_qs(parsed.query, keep_blank_values=True)
            qs["offset"] = [str(offset)]
            new_query = urlencode(qs, doseq=True)
            url = urlunparse(parsed._replace(query=new_query))
        elif pattern == "path_page":
            # /page/N pattern
            path = parsed.path.rstrip("/")
            # Remove existing /page/N if present
            path = re.sub(r'/page/\d+', '', path)
            url = urlunparse(parsed._replace(path=f"{path}/page/{pg}"))
        else:
            # Default fallback: try ?page=N
            qs = parse_qs(parsed.query, keep_blank_values=True)
            qs["page"] = [str(pg)]
            new_query = urlencode(qs, doseq=True)
            url = urlunparse(parsed._replace(query=new_query))

        urls.append(url)

    return urls


def parse_extracted_lots(result) -> list[dict]:
    """Extract lot array from Phase 1b output (risk + extraction)."""
    raw = getattr(result, "raw", str(result))

    # Try last task output (extract_item_details)
    task_outputs = getattr(result, "tasks_output", None)
    if task_outputs:
        last_raw = getattr(task_outputs[-1], "raw", "")
        parsed = _extract_json_from_output(last_raw)
        if isinstance(parsed, list):
            return parsed

    parsed = _extract_json_from_output(raw)
    if isinstance(parsed, list):
        return parsed

    return []


def parse_rejection_reasons(result) -> dict[str, str]:
    """Parse the REJECTED LOTS section from risk assessment output.
    Returns dict mapping lot_number -> rejection reason string."""
    task_outputs = getattr(result, "tasks_output", None)
    raw = ""
    if task_outputs and len(task_outputs) >= 1:
        raw = getattr(task_outputs[0], "raw", "")
    if not raw:
        raw = getattr(result, "raw", str(result))

    reasons: dict[str, str] = {}

    # Look for the REJECTED LOTS section
    m = re.search(r'REJECTED\s+LOTS:\s*\n(.*?)(?:\n\n|\Z)', raw, re.DOTALL | re.IGNORECASE)
    if m:
        block = m.group(1)
        # Parse each line: "- Lot 123: "title" — REJECTED because: reason"
        for line in block.split("\n"):
            line = line.strip()
            if not line or line.lower() == "none":
                continue
            lot_m = re.match(
                r'-?\s*Lot\s+(\d+)\s*:?\s*.*?(?:REJECTED\s+because|Reason|because)\s*:?\s*(.+)',
                line, re.IGNORECASE
            )
            if lot_m:
                lot_num = lot_m.group(1).strip()
                reason = lot_m.group(2).strip()
                reasons[lot_num] = reason
            else:
                # Fallback: try simpler pattern "- Lot 123: reason"
                lot_m2 = re.match(r'-?\s*Lot\s+(\d+)\s*:?\s*(.+)', line, re.IGNORECASE)
                if lot_m2:
                    lot_num = lot_m2.group(1).strip()
                    reason = lot_m2.group(2).strip()
                    reasons[lot_num] = reason

    return reasons


def parse_per_lot_result(result) -> dict:
    """Extract per-lot result dict from a per-lot crew output."""
    raw = getattr(result, "raw", str(result))

    # Try last task output
    task_outputs = getattr(result, "tasks_output", None)
    if task_outputs:
        last_raw = getattr(task_outputs[-1], "raw", "")
        parsed = _extract_json_from_output(last_raw)
        if isinstance(parsed, dict):
            return parsed

    parsed = _extract_json_from_output(raw)
    if isinstance(parsed, dict):
        return parsed

    # Return a minimal dict with the raw output
    return {"raw_output": raw}


def select_top_lots(validated_lots: list[dict], top_pct: float = 0.4) -> list[dict]:
    """Rank validated lots by expected profit margin, return top N%."""
    # Try to sort by expected_profit_margin_pct
    def get_margin(lot):
        for key in ["expected_profit_margin_pct", "estimated_profit_margin_percentage", "margin_pct"]:
            val = lot.get(key)
            if val is not None:
                try:
                    return float(val)
                except (ValueError, TypeError):
                    pass
        return -999

    ranked = sorted(validated_lots, key=get_margin, reverse=True)

    # Filter to BUY/MONITOR recommendations only
    viable = [
        lot for lot in ranked
        if lot.get("investment_recommendation", "").upper() in ("BUY", "MONITOR", "")
    ]

    if not viable:
        viable = ranked

    # Take top N%
    n = max(1, int(len(viable) * top_pct))
    return viable[:n]


# ═══════════════════════════════════════════════════════════════════════════
# UI RENDERING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def _fmt_eta(seconds: float | None) -> str:
    """Format ETA seconds into human-readable string."""
    if seconds is None:
        return ""
    if seconds <= 0:
        return "finishing..."
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"~{h}h {m}m remaining"
    if m > 0:
        return f"~{m}m {s}s remaining"
    return f"~{s}s remaining"


def _fmt_elapsed(start: float) -> str:
    """Format elapsed time from a start timestamp."""
    if start <= 0:
        return ""
    elapsed = time.time() - start
    m, s = divmod(int(elapsed), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def render_pipeline_stages(tracker: ProgressTracker, container):
    """Render a horizontal pipeline stage indicator showing all phases."""
    with container.container():
        current_key = PHASE_KEY_MAP.get(tracker.current_phase)
        reached = False
        phase_idx = -1
        for i, p in enumerate(PIPELINE_PHASES):
            if p["key"] == current_key:
                phase_idx = i
                break

        # Build the stage indicator HTML
        stages_html = '<div style="display:flex;gap:4px;align-items:stretch;margin:0.6rem 0 0.8rem 0;">'
        for i, p in enumerate(PIPELINE_PHASES):
            is_current = (p["key"] == current_key)
            is_done = (phase_idx >= 0 and i < phase_idx) or (tracker.finished and phase_idx >= 0 and i <= phase_idx)

            if tracker.finished and not tracker.error:
                is_done = True
                is_current = False

            if is_current:
                bg = "#1976d2"
                fg = "#fff"
                border = "2px solid #1565c0"
                opacity = "1"
                icon = "&#9654;"  # play arrow
            elif is_done:
                bg = "#e8f5e9"
                fg = "#2e7d32"
                border = "1px solid #a5d6a7"
                opacity = "1"
                icon = "&#10003;"  # checkmark
            else:
                bg = "#f5f5f5"
                fg = "#999"
                border = "1px solid #e0e0e0"
                opacity = "0.7"
                icon = p["icon"]

            stages_html += (
                f'<div style="flex:1;background:{bg};color:{fg};border:{border};'
                f'border-radius:8px;padding:8px 6px;text-align:center;opacity:{opacity};'
                f'font-size:0.78rem;line-height:1.35;min-width:0;">'
                f'<div style="font-size:1.1rem;font-weight:700;margin-bottom:2px;">{icon}</div>'
                f'<div style="font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{p["label"]}</div>'
                f'<div style="font-size:0.68rem;opacity:0.8;margin-top:1px;">{p["detail"]}</div>'
                f'</div>'
            )
        stages_html += '</div>'

        st.markdown(stages_html, unsafe_allow_html=True)


def render_phase_progress(tracker: ProgressTracker, container):
    """Render detailed progress for the current phase including lot progress and ETA."""
    with container.container():
        phase = tracker.current_phase
        lot_cur = tracker.phase_lot_current
        lot_tot = tracker.phase_lot_total
        step = tracker.phase_step

        # Elapsed time
        elapsed_str = _fmt_elapsed(tracker.workflow_start) if tracker.workflow_start else ""
        if elapsed_str:
            st.caption(f"Elapsed: {elapsed_str}")

        # Extraction-specific progress during Phase 1a
        if phase == "Phase 1a: Extracting Catalog":
            _render_extraction_progress(tracker)
        elif lot_tot > 0:
            # Show lot-level progress with progress bar
            pct = lot_cur / lot_tot
            eta = _fmt_eta(tracker.get_eta_seconds())
            progress_label = f"Lot {lot_cur} of {lot_tot}"
            if eta:
                progress_label += f"  ·  {eta}"
            st.progress(pct, text=progress_label)
        else:
            if step:
                st.markdown(f"*{step}*")

        if tracker.finished:
            if tracker.error:
                st.error("Workflow failed")
            else:
                st.success("Workflow complete")


def _render_extraction_progress(tracker: ProgressTracker):
    """Render extraction-specific progress during Phase 1a."""
    pages = tracker.extraction_pages_scraped
    total_pages = tracker.extraction_total_pages
    lots_found = tracker.extraction_lots_found
    current_page = tracker.extraction_current_page
    last_lot = tracker.extraction_last_lot
    step = tracker.phase_step

    # If we have page/lot data, show the rich extraction indicator
    if pages > 0 or lots_found > 0:
        # Build the status parts
        parts = []
        if current_page:
            if total_pages > 0:
                parts.append(f"Scraping {current_page} of ~{total_pages}")
            else:
                parts.append(f"Scraping {current_page}")
        if lots_found > 0:
            parts.append(f"{lots_found} lots found")
        if last_lot:
            parts.append(f"Last seen: {last_lot}")

        progress_text = "  \u00b7  ".join(parts) if parts else "Extracting catalog..."

        # Show a progress bar if we know total pages
        if total_pages > 0 and pages > 0:
            pct = min(pages / total_pages, 1.0)
            st.progress(pct, text=progress_text)
        else:
            # Show an indeterminate-style status
            extraction_html = (
                f'<div style="background:#e8f5e9;border-left:4px solid #43a047;'
                f'padding:10px 14px;border-radius:0 8px 8px 0;font-size:0.9rem;">'
                f'{progress_text}</div>'
            )
            st.markdown(extraction_html, unsafe_allow_html=True)
    else:
        # Before scraping starts, show the sub-step (validation, buyer premium, etc.)
        if step:
            st.markdown(f"*{step}*")


def render_agent_activity(tracker: ProgressTracker, container):
    """Render the current agent activity with visual emphasis."""
    with container.container():
        agent = tracker.current_agent or "Initializing..."

        # Agent name with highlighted style
        agent_html = (
            f'<div style="background:#f0f7ff;border-left:4px solid #1976d2;'
            f'padding:10px 14px;border-radius:0 8px 8px 0;margin-bottom:8px;">'
            f'<span style="font-weight:700;font-size:1rem;color:#1565c0;">{agent}</span>'
        )
        if tracker.current_task_desc:
            desc = tracker.current_task_desc[:180]
            agent_html += f'<br><span style="font-size:0.82rem;color:#666;margin-top:2px;">{desc}</span>'
        agent_html += '</div>'
        st.markdown(agent_html, unsafe_allow_html=True)

        if tracker.current_tool:
            tool_html = (
                f'<div style="background:#fff3e0;border-left:3px solid #f57c00;'
                f'padding:6px 12px;border-radius:0 6px 6px 0;margin-bottom:6px;font-size:0.85rem;">'
                f'<b>Tool:</b> <code>{tracker.current_tool}</code>'
            )
            if tracker.current_tool_input:
                inp = tracker.current_tool_input[:180]
                tool_html += f'<br><span style="color:#888;font-size:0.78rem;">{inp}</span>'
            tool_html += '</div>'
            st.markdown(tool_html, unsafe_allow_html=True)

        if tracker.current_thought:
            thought = tracker.current_thought[:350]
            st.info(f"**Thinking:** {thought}")


def render_summary_stats(tracker: ProgressTracker, container):
    """Render summary statistics for the workflow."""
    with container.container():
        lots = list(tracker.lots.values())
        if not lots:
            return

        total = len(lots)
        rejected = sum(1 for l in lots if l.status == "rejected")
        active = total - rejected
        validating = sum(1 for l in lots if l.status == "validating")
        researching = sum(1 for l in lots if l.status == "researching")
        complete = sum(1 for l in lots if l.status == "complete")
        buys = sum(1 for l in lots if l.recommendation == "BUY")

        cols = st.columns(6)
        cols[0].metric("Filtered", total)
        cols[1].metric("Active", active)
        cols[2].metric("Rejected", rejected)
        cols[3].metric("Validating", validating + researching)
        cols[4].metric("Complete", complete)
        cols[5].metric("BUY", buys)


def render_lots(tracker: ProgressTracker, container):
    """Render the live lot tracker with tabs for active and rejected lots."""
    with container.container():
        lots = list(tracker.lots.values())
        if not lots:
            st.caption("Waiting for catalog extraction and filtering...")
            return

        active_lots = [l for l in lots if l.status != "rejected"]
        rejected_lots = [l for l in lots if l.status == "rejected"]

        tab_active, tab_rejected = st.tabs([
            f"Active Lots ({len(active_lots)})",
            f"Rejected Lots ({len(rejected_lots)})"
        ])

        with tab_active:
            if not active_lots:
                st.caption("No active lots yet...")
            else:
                rows = []
                for lot in sorted(active_lots, key=lambda l: int(l.lot_num) if l.lot_num.isdigit() else 0):
                    if lot.recommendation == "BUY":
                        status_display = "BUY"
                    elif lot.recommendation == "AVOID":
                        status_display = "AVOID"
                    elif lot.recommendation == "MONITOR":
                        status_display = "MONITOR"
                    elif lot.status == "complete":
                        status_display = "Researched"
                    elif lot.status == "researching":
                        status_display = "Researching..."
                    elif lot.status == "validating":
                        status_display = "Validating..."
                    elif lot.status == "passed":
                        status_display = "Passed Risk"
                    else:
                        status_display = "Queued"

                    rows.append({
                        "Lot": lot.lot_num,
                        "Title": lot.title[:55] if lot.title else "-",
                        "Stage": lot.stage or "-",
                        "FMV": lot.fmv or "-",
                        "Margin": lot.margin or "-",
                        "Status": status_display,
                    })

                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True, hide_index=True,
                             height=min(400, 35 * len(rows) + 40))

        with tab_rejected:
            if not rejected_lots:
                st.caption("No lots rejected yet.")
            else:
                rows = []
                for lot in sorted(rejected_lots, key=lambda l: int(l.lot_num) if l.lot_num.isdigit() else 0):
                    rows.append({
                        "Lot": lot.lot_num,
                        "Title": lot.title[:50] if lot.title else "-",
                        "Rejection Reason": lot.rejection_reason or "No details",
                    })

                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True, hide_index=True,
                             height=min(400, 35 * len(rows) + 40))


def render_insights(tracker: ProgressTracker, container):
    """Render research insights panel."""
    with container.container():
        st.markdown("#### Research Insights")
        insights = tracker.insights
        if not insights:
            st.caption("Insights will appear as research agents analyze lots...")
            return

        for ins in reversed(insights[-30:]):
            lot_tag = f"**Lot {ins.lot}**" if ins.lot and ins.lot != "-" else ""
            agent_tag = f"`{ins.agent}`" if ins.agent else ""
            prefix = " - ".join(filter(None, [lot_tag, agent_tag]))
            st.markdown(f"- {prefix}: {ins.text}")


def render_log(tracker: ProgressTracker, container):
    """Render the scrollable activity log."""
    with container.container():
        st.markdown("#### Activity Log")
        entries = tracker.log
        if not entries:
            st.caption("Waiting for activity...")
            return

        lines = []
        for ts, agent, msg in entries[-50:]:
            lines.append(f"`{ts}` **{agent}**: {msg}")
        st.markdown("\n\n".join(reversed(lines)))


# ═══════════════════════════════════════════════════════════════════════════
# SESSION STATE & SEARCH HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _init_session_state():
    if "phrases" not in st.session_state:
        st.session_state.phrases = []


def _add_phrases(input_key: str):
    """Parse comma-separated input into individual phrases and add them."""
    raw = st.session_state.get(input_key, "").strip()
    if not raw:
        return
    # Split by comma, strip whitespace, deduplicate
    for part in raw.split(","):
        phrase = part.strip()
        if phrase and phrase.lower() not in [p.lower() for p in st.session_state.phrases]:
            st.session_state.phrases.append(phrase)
    st.session_state[input_key] = ""


def _remove_phrase(phrase: str):
    try:
        st.session_state.phrases.remove(phrase)
    except ValueError:
        pass


def _clear_phrases():
    st.session_state.phrases = []


def _render_chips_html(terms: list, bg: str, fg: str) -> str:
    if not terms:
        return ""
    chips = []
    for t in terms:
        chips.append(
            f'<span style="background:{bg};color:{fg};padding:5px 14px;'
            f'border-radius:20px;margin:3px 4px 3px 0;display:inline-block;'
            f'font-size:0.9em;font-weight:500;letter-spacing:0.01em;">{t}</span>'
        )
    return "".join(chips)


# ═══════════════════════════════════════════════════════════════════════════
# FINAL OUTPUT RENDERING
# ═══════════════════════════════════════════════════════════════════════════

def _try_parse_json(text: str):
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


def _extract_sources_json(raw_output: str):
    """Extract the SOURCES_JSON block from the bidding sheet output."""
    m = re.search(
        r'<!--\s*SOURCES_JSON_START\s*-->\s*(\[.*?\])\s*<!--\s*SOURCES_JSON_END\s*-->',
        raw_output,
        re.DOTALL,
    )
    if m:
        try:
            return json.loads(m.group(1))
        except (json.JSONDecodeError, TypeError):
            pass
    return None


def render_final_sources(raw_output: str):
    """Render per-lot, per-platform source dropdowns from the crew output."""
    sources_data = _extract_sources_json(raw_output)
    if not sources_data:
        return

    st.divider()
    st.subheader("Research Sources by Lot")
    st.caption(
        "Expand each lot to see the individual listings that contributed to the "
        "market value estimate, organised by platform."
    )

    try:
        sources_data = sorted(
            sources_data,
            key=lambda x: float(x.get("margin_pct", 0) or 0),
            reverse=True,
        )
    except (ValueError, TypeError):
        pass

    for lot_data in sources_data:
        lot_num = lot_data.get("lot", "?")
        lot_title = lot_data.get("title", "Unknown")
        margin = lot_data.get("margin_pct", "")
        rec = lot_data.get("recommendation", "")
        sources = lot_data.get("sources", {})

        if not sources:
            continue

        margin_str = f" - {margin}% margin" if margin else ""
        rec_icon = {"BUY": "BUY", "AVOID": "AVOID", "MONITOR": "MONITOR"}.get(str(rec).upper(), "")
        label = f"Lot {lot_num}: {lot_title}{margin_str} [{rec_icon}]"

        with st.expander(label, expanded=False):
            platform_names = list(sources.keys())
            if not platform_names:
                st.caption("No source data available.")
                continue

            tabs = st.tabs(platform_names)
            for tab, platform in zip(tabs, platform_names):
                with tab:
                    listings = sources[platform]
                    if not listings:
                        st.caption(f"No listings found on {platform}.")
                        continue

                    rows = []
                    for item in listings:
                        row = {
                            "Title": item.get("title", "-"),
                            "Sold Price": item.get("sold_price", "-"),
                        }
                        if "seller_net" in item and item["seller_net"]:
                            row["Seller Net"] = item["seller_net"]
                        if "date" in item and item["date"]:
                            row["Date"] = item["date"]
                        if "url" in item and item["url"]:
                            row["Link"] = item["url"]
                        rows.append(row)

                    df = pd.DataFrame(rows)

                    if "Link" in df.columns:
                        df["Link"] = df["Link"].apply(
                            lambda u: f'<a href="{u}" target="_blank">View</a>'
                            if u and u != "-" else "-"
                        )
                        st.markdown(
                            df.to_html(escape=False, index=False),
                            unsafe_allow_html=True,
                        )
                    else:
                        st.dataframe(df, use_container_width=True, hide_index=True)

                    st.caption(f"{len(listings)} listing(s) on {platform}")


def render_final_result(raw_output: str):
    """Show the final crew output in the best format."""
    parsed = _try_parse_json(raw_output)
    if parsed is not None:
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
            st.subheader("Results (Table)")
            st.dataframe(pd.DataFrame(parsed), use_container_width=True)
        else:
            st.subheader("Results (JSON)")
            st.json(parsed)
    else:
        st.subheader("Results")
        st.markdown(raw_output)

    with st.expander("Raw output", expanded=False):
        st.code(raw_output, language="markdown")


# ═══════════════════════════════════════════════════════════════════════════
# STREAMLIT PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Fast Auction Research",
    page_icon="🔨",
    layout="wide",
    initial_sidebar_state="expanded",
)

_init_session_state()

# ── Custom CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
    .hero-title {
        font-size: 2.4rem; font-weight: 700; line-height: 1.15;
        margin-bottom: 0.15rem;
    }
    .hero-sub {
        font-size: 1.05rem; color: #777; margin-bottom: 1.2rem; line-height: 1.5;
    }
    .search-preview {
        background: #f0f7ff; border-left: 4px solid #1976d2;
        padding: 0.9rem 1.2rem; border-radius: 0 8px 8px 0;
        margin: 0.8rem 0 0.4rem 0; font-size: 0.93rem; line-height: 1.6;
    }
    .search-preview code {
        background: #d6e8fa; padding: 2px 6px; border-radius: 4px;
        font-size: 0.88em;
    }
    /* Sidebar How-It-Works steps */
    .sidebar-step {
        display: flex; align-items: flex-start; gap: 0.7rem;
        margin-bottom: 0.75rem;
    }
    .sidebar-step-num {
        display: inline-flex; align-items: center; justify-content: center;
        background: #1976d2; color: #fff; min-width: 24px; height: 24px;
        border-radius: 50%; font-weight: 700; font-size: 0.78rem;
        flex-shrink: 0; margin-top: 1px;
    }
    .sidebar-step-label {
        font-weight: 600; font-size: 0.9rem; display: block;
    }
    .sidebar-step-detail {
        font-size: 0.76rem; color: #888; display: block; margin-top: 1px;
    }
    /* Metric styling for summary stats */
    [data-testid="stMetric"] {
        background: #f8f9fa;
        border: 1px solid #e8e8e8;
        border-radius: 8px;
        padding: 8px 12px;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.6rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("#### How It Works")
    st.markdown(
        '<div class="sidebar-step">'
        '<span class="sidebar-step-num">1</span>'
        '<div><span class="sidebar-step-label">Extract</span>'
        '<span class="sidebar-step-detail">Scout scrapes the full catalog; Python filters by your exact keywords</span></div>'
        '</div>'
        '<div class="sidebar-step">'
        '<span class="sidebar-step-num">2</span>'
        '<div><span class="sidebar-step-label">Validate</span>'
        '<span class="sidebar-step-detail">Per-lot market validation with fresh token window for each lot</span></div>'
        '</div>'
        '<div class="sidebar-step">'
        '<span class="sidebar-step-num">3</span>'
        '<div><span class="sidebar-step-label">Research</span>'
        '<span class="sidebar-step-detail">Deep research on top 40% of lots, each with its own context</span></div>'
        '</div>'
        '<div class="sidebar-step">'
        '<span class="sidebar-step-num">4</span>'
        '<div><span class="sidebar-step-label">Report</span>'
        '<span class="sidebar-step-detail">Ranked by profit margin; mobile bidding sheet saved to Box</span></div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown("#### API Key Status")
    for key in ["GEMINI_API_KEY", "OPENAI_API_KEY", "SERPER_API_KEY",
                 "FIRECRAWL_API_KEY", "BROWSERBASE_API_KEY"]:
        icon = "Y" if os.environ.get(key) else "N"
        st.text(f"{icon}  {key}")


# ═══════════════════════════════════════════════════════════════════════════
# LANDING PAGE
# ═══════════════════════════════════════════════════════════════════════════

st.markdown(
    '<p class="hero-title">Fast Auction Research</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="hero-sub">'
    "Paste an auction catalog URL, add your search filters, and let the AI crew "
    "validate market prices and rank profit potential in one click."
    "</p>",
    unsafe_allow_html=True,
)

# ── Auction URL ──────────────────────────────────────────────────────────
auction_url = st.text_input(
    "Auction Catalog URL",
    placeholder="https://www.example-auction.com/sale/12345",
    help="Full URL of the auction catalog page to scan.",
)

# ── Search Filters ───────────────────────────────────────────────────────
st.caption(
    "Add brands, designers, or item types. A lot matches if it contains **any** phrase. "
    "Multi-word phrases require **all** words to appear."
)

inp_col, btn_col = st.columns([5, 1])
with inp_col:
    st.text_input(
        "phrase_input_label",
        key="phrase_input",
        placeholder="e.g. Chanel, Bottega Veneta, Dior, Hermes Birkin ...",
        label_visibility="collapsed",
    )
with btn_col:
    st.button("Add", key="add_phrase_btn", on_click=_add_phrases,
               args=("phrase_input",), use_container_width=True)

if st.session_state.phrases:
    st.markdown(
        _render_chips_html(st.session_state.phrases, "#e3f2fd", "#1565c0"),
        unsafe_allow_html=True,
    )
    n = len(st.session_state.phrases)
    cols_per_row = min(n, 8)
    rm_cols = st.columns(cols_per_row)
    for i, phrase in enumerate(st.session_state.phrases):
        rm_cols[i % cols_per_row].button(
            f"x {phrase}", key=f"rm_phrase_{i}",
            on_click=_remove_phrase, args=(phrase,),
        )
    if n > 1:
        st.button("Clear all", key="clear_phrases", on_click=_clear_phrases)
else:
    st.caption("No phrases added yet. Type above and click Add, or enter multiple separated by commas.")

# ── Live search preview (directly under chips) ──────────────────────────
phrases = st.session_state.phrases
has_terms = bool(phrases)

if has_terms:
    # Build human-readable logic string
    logic_parts = []
    for phrase in phrases:
        words = phrase.split()
        if len(words) > 1:
            logic_parts.append(f"({' AND '.join(words)})")
        else:
            logic_parts.append(phrase)
    logic = " OR ".join(logic_parts)

    st.markdown(
        f'<div class="search-preview">'
        f"<b>Filter logic:</b> <code>{logic}</code><br><br>"
        f"A lot is included if it matches <b>any one</b> of the above phrases."
        f"</div>",
        unsafe_allow_html=True,
    )

# ── Bidding Fee (inline) ────────────────────────────────────────────────
st.markdown("")
platform_fee_paid = st.toggle(
    "I will pay the flat registration fee to avoid the 3% online bidding surcharge",
    value=False,
    help="Most platforms charge a 3% internet surcharge. Enable this if you plan to "
         "pay the flat registration fee instead, removing the per-lot surcharge from "
         "cost calculations.",
)

# ── Start button ─────────────────────────────────────────────────────────
st.markdown("")
run_button = st.button(
    "Start Research Workflow",
    type="primary",
    use_container_width=True,
    disabled=(not auction_url or not has_terms),
)

if not auction_url and not has_terms:
    st.caption("Enter an auction URL and add at least one search term to get started.")
elif not auction_url:
    st.caption("Enter an auction catalog URL above to enable the workflow.")
elif not has_terms:
    st.caption("Add at least one search term to enable the workflow.")


# ═══════════════════════════════════════════════════════════════════════════
# WORKFLOW EXECUTION — 4-PHASE MULTI-CREW ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════

if run_button:
    platform_fee_choice = "flat_fee_paid" if platform_fee_paid else "3_percent_surcharge"
    fee_label = "Flat fee paid (no 3% surcharge)" if platform_fee_paid else "3% internet surcharge applies"

    inputs = {
        "auction_url": auction_url.strip(),
        "platform_fee_choice": platform_fee_choice,
    }

    st.info(
        f"**Auction URL:** {inputs['auction_url']}  \n"
        f"**Bidding fee:** {fee_label}  \n"
        f"**Search phrases:** {', '.join(phrases)}"
    )

    # ── Import crews ─────────────────────────────────────────────────────
    from fast_auction_research___speed_optimized.crew import (
        CatalogSetupCrew,
        PageExtractionCrew,
        ScreeningCrewPartB,
        PerLotValidationCrew,
        PerLotDeepResearchCrew,
        SynthesisCrew,
    )

    # ── Create tracker & UI containers ───────────────────────────────────
    tracker = ProgressTracker()
    tracker.workflow_start = time.time()
    tracker.add_log("System", "Starting multi-phase workflow")

    st.divider()

    # Pipeline stages bar (full width)
    stages_ph = st.empty()

    # Summary stats row (full width)
    stats_ph = st.empty()

    # Phase progress + ETA (full width)
    progress_ph = st.empty()

    # Tabbed dashboard: Agent Activity | Lot Tracker | Insights | Activity Log
    tab_agent, tab_lots, tab_insights, tab_log = st.tabs([
        "Agent Activity", "Lot Tracker", "Insights", "Activity Log"
    ])
    with tab_agent:
        activity_ph = st.empty()
    with tab_lots:
        lots_ph = st.empty()
    with tab_insights:
        insights_ph = st.empty()
    with tab_log:
        log_ph = st.empty()

    result_ph = st.empty()

    def _update_ui():
        render_pipeline_stages(tracker, stages_ph)
        render_summary_stats(tracker, stats_ph)
        render_phase_progress(tracker, progress_ph)
        render_agent_activity(tracker, activity_ph)
        render_lots(tracker, lots_ph)
        render_insights(tracker, insights_ph)
        render_log(tracker, log_ph)

    # ── Stdout interceptor ───────────────────────────────────────────────
    original_stdout = sys.stdout
    interceptor = OutputInterceptor(tracker, original_stdout)
    sys.stdout = interceptor

    start_time = time.time()

    # #region agent log — CANARY: confirm instrumented code is running
    _DBG_LOG_PATH = os.path.join(_PROJECT_ROOT, ".cursor", "debug.log")
    os.makedirs(os.path.dirname(_DBG_LOG_PATH), exist_ok=True)
    def _dbg(msg, data, hyp="X"):
        try:
            with open(_DBG_LOG_PATH, "a") as _f:
                _f.write(json.dumps({"timestamp": int(time.time()*1000), "location": "app.py", "message": msg, "data": data, "hypothesisId": hyp}) + "\n")
        except Exception:
            pass
    _dbg("CANARY_workflow_start", {"auction_url": inputs.get("auction_url", "")[:100], "phrases": phrases}, "CANARY")
    # #endregion

    try:
        # ═════════════════════════════════════════════════════════════════
        # PHASE 1a — STEP 1: Catalog Setup (validation + buyer premium)
        # ═════════════════════════════════════════════════════════════════
        tracker.set_phase("Phase 1a: Extracting Catalog", step="Validating auction URL & discovering buyer premium...")
        tracker.add_log("System", "Phase 1a-setup: Validating URL and discovering buyer premium")
        _update_ui()

        crew_setup = CatalogSetupCrew().crew()
        crew_setup.step_callback = make_step_callback(tracker)
        crew_setup.task_callback = make_task_callback(tracker)

        result_setup = crew_setup.kickoff(inputs=inputs)

        # #region agent log — Hypothesis E: CatalogSetupCrew output
        _dbg("setup_crew_result_type", {"type": str(type(result_setup)), "has_raw": hasattr(result_setup, "raw"), "raw_len": len(getattr(result_setup, "raw", "")), "raw_first_500": getattr(result_setup, "raw", "")[:500]}, "E")
        _setup_tasks = getattr(result_setup, "tasks_output", None)
        _dbg("setup_tasks_output", {"count": len(_setup_tasks) if _setup_tasks else 0, "task_names": [getattr(t, "name", getattr(t, "description", "?"))[:80] for t in (_setup_tasks or [])]}, "E")
        if _setup_tasks:
            for _i, _t in enumerate(_setup_tasks):
                _dbg(f"setup_task_{_i}_raw", {"raw_len": len(getattr(_t, "raw", "")), "raw_first_800": getattr(_t, "raw", "")[:800]}, "E")
        # #endregion

        buyer_premium_data = extract_buyer_premium(result_setup)

        # Parse pagination info from the validation task output
        validation_raw = ""
        setup_task_outputs = getattr(result_setup, "tasks_output", None)
        if setup_task_outputs and len(setup_task_outputs) >= 1:
            validation_raw = getattr(setup_task_outputs[0], "raw", "")

        pagination_info = _parse_pagination_info(validation_raw)
        page_urls = construct_page_urls(inputs["auction_url"], pagination_info)

        # #region agent log — Hypothesis A: Pagination parsing
        _dbg("validation_raw_length", {"len": len(validation_raw), "first_600": validation_raw[:600]}, "A")
        _dbg("pagination_info_parsed", {"info": pagination_info}, "A")
        _dbg("page_urls_constructed", {"count": len(page_urls), "urls": page_urls[:10]}, "A")
        # #endregion

        tracker.add_log("System",
            f"Pagination detected: {pagination_info['pagination_detected']}, "
            f"~{pagination_info['estimated_lots']} lots across ~{len(page_urls)} pages"
        )
        tracker.update_extraction(
            extraction_total_pages=len(page_urls),
        )
        _update_ui()

        # ═════════════════════════════════════════════════════════════════
        # PHASE 1a — STEP 2: Per-Page Extraction (concurrent, capped)
        #
        # Runs page extraction crews concurrently with a semaphore to
        # respect Firecrawl free-plan limits (2 concurrent, 10 RPM).
        # Each crew gets its own token window so output is never truncated.
        # Adjust MAX_CONCURRENT_PAGES if you upgrade your Firecrawl plan.
        # ═════════════════════════════════════════════════════════════════
        MAX_CONCURRENT_PAGES = 2  # safe for Firecrawl free plan (2 concurrent, 10 RPM)

        tracker.add_log("System",
            f"Phase 1a-pages: Extracting lots from {len(page_urls)} page(s) "
            f"({MAX_CONCURRENT_PAGES} concurrent)"
        )
        _update_ui()

        # -- result container (thread-safe via the GIL for simple list ops) --
        page_results: dict[int, list[dict]] = {}  # page_num -> lots

        async def _extract_one_page(
            semaphore: asyncio.Semaphore,
            page_idx: int,
            page_url: str,
        ):
            """Extract lots from a single page, respecting the semaphore."""
            page_num = page_idx + 1
            async with semaphore:
                tracker.add_log("Scout", f"Scraping page {page_num}: {page_url[:120]}")
                tracker.update_extraction(
                    extraction_current_page=f"Pages {page_num}/{len(page_urls)}",
                    extraction_pages_scraped=page_num,
                )
                tracker.set_phase(
                    "Phase 1a: Extracting Catalog",
                    step=f"Scraping page {page_num} of {len(page_urls)} "
                         f"({MAX_CONCURRENT_PAGES} concurrent)..."
                )

                try:
                    # #region agent log — Hypothesis B: crew task list
                    crew_page = PageExtractionCrew().crew()
                    _dbg(f"page_{page_num}_crew_tasks", {"task_count": len(crew_page.tasks), "task_names": [getattr(t, "name", getattr(t, "description", "?"))[:80] for t in crew_page.tasks], "agent_count": len(crew_page.agents)}, "B")
                    # #endregion
                    crew_page.step_callback = make_step_callback(tracker)
                    crew_page.task_callback = make_task_callback(tracker)

                    # #region agent log — Hypothesis C: akickoff execution
                    _dbg(f"page_{page_num}_akickoff_start", {"page_num": page_num, "page_url": page_url[:150]}, "C")
                    # #endregion
                    # akickoff = native async CrewAI execution
                    result_page = await crew_page.akickoff(inputs={
                        **inputs,
                        "page_url": page_url,
                    })

                    # #region agent log — Hypothesis C+D: result inspection
                    _dbg(f"page_{page_num}_akickoff_result", {"type": str(type(result_page)), "has_raw": hasattr(result_page, "raw"), "raw_len": len(getattr(result_page, "raw", "")), "raw_first_500": getattr(result_page, "raw", "")[:500]}, "CD")
                    _pg_tasks = getattr(result_page, "tasks_output", None)
                    _dbg(f"page_{page_num}_tasks_output", {"count": len(_pg_tasks) if _pg_tasks else 0}, "CD")
                    if _pg_tasks:
                        for _pi, _pt in enumerate(_pg_tasks):
                            _dbg(f"page_{page_num}_task_{_pi}_raw", {"raw_len": len(getattr(_pt, "raw", "")), "raw_first_500": getattr(_pt, "raw", "")[:500]}, "CD")
                    # #endregion

                    page_lots = parse_lots_from_page_output(result_page)

                    # #region agent log — Hypothesis D: parse result
                    _dbg(f"page_{page_num}_parsed_lots", {"count": len(page_lots), "first_2": page_lots[:2] if page_lots else []}, "D")
                    # #endregion

                    if page_lots:
                        page_results[page_num] = page_lots
                        # Update running total
                        running_total = sum(len(v) for v in page_results.values())
                        tracker.update_extraction(extraction_lots_found=running_total)
                        tracker.add_log("Scout",
                            f"Page {page_num}: extracted {len(page_lots)} lots "
                            f"(running total: {running_total})"
                        )
                    else:
                        page_results[page_num] = []
                        tracker.add_log("Scout", f"Page {page_num}: 0 lots found")

                except Exception as exc:
                    # #region agent log — Hypothesis C: exception details
                    import traceback as _tb
                    _dbg(f"page_{page_num}_exception", {"page_num": page_num, "error": str(exc), "traceback": _tb.format_exc()}, "C")
                    # #endregion
                    page_results[page_num] = []
                    tracker.add_log("System", f"Page {page_num} extraction failed: {exc}")

        async def _extract_all_pages():
            """Run all page extractions with bounded concurrency."""
            sem = asyncio.Semaphore(MAX_CONCURRENT_PAGES)
            tasks = [
                _extract_one_page(sem, idx, url)
                for idx, url in enumerate(page_urls)
            ]
            await asyncio.gather(*tasks)

        # Run the async extraction — Streamlit runs sync, so use asyncio.run()
        # in a thread to avoid event-loop conflicts with Streamlit's own loop.
        def _run_async_extraction():
            try:
                asyncio.run(_extract_all_pages())
                # #region agent log — thread completion
                _dbg("async_extraction_complete", {"page_results_keys": list(page_results.keys()), "total_lots": sum(len(v) for v in page_results.values())}, "C")
                # #endregion
            except Exception as _thr_exc:
                # #region agent log — thread-level error
                import traceback as _tb2
                _dbg("async_thread_exception", {"error": str(_thr_exc), "traceback": _tb2.format_exc()}, "C")
                # #endregion

        extraction_thread = threading.Thread(target=_run_async_extraction, daemon=True)
        extraction_thread.start()

        # Poll for completion while keeping the Streamlit UI responsive
        while extraction_thread.is_alive():
            extraction_thread.join(timeout=2.0)
            _update_ui()

        # Assemble all_lots in page order
        all_lots: list[dict] = []
        for pg in sorted(page_results.keys()):
            all_lots.extend(page_results[pg])

        tracker.add_log("System",
            f"All pages complete: {len(all_lots)} lots from {len(page_urls)} pages"
        )

        # Deduplicate lots by lot_number (in case pages overlap)
        seen_lot_nums: set[str] = set()
        deduped_lots: list[dict] = []
        for lot in all_lots:
            lot_num = str(lot.get("lot_number", ""))
            if lot_num and lot_num in seen_lot_nums:
                continue
            if lot_num:
                seen_lot_nums.add(lot_num)
            deduped_lots.append(lot)
        all_lots = deduped_lots

        tracker.total_catalog_lots = len(all_lots)
        tracker.add_log("System", f"Catalog extracted: {len(all_lots)} total lots")

        # #region agent log — final lot count before keyword filter
        _dbg("all_lots_final", {"count": len(all_lots), "first_3": all_lots[:3] if all_lots else [], "page_results_summary": {str(k): len(v) for k, v in page_results.items()}}, "FINAL")
        # #endregion

        # ═════════════════════════════════════════════════════════════════
        # PYTHON KEYWORD FILTER (deterministic)
        # ═════════════════════════════════════════════════════════════════
        tracker.set_phase("Keyword Filtering", step="Applying Python filter...")
        _update_ui()

        filtered_lots = apply_keyword_filter(all_lots, phrases)

        tracker.total_filtered_lots = len(filtered_lots)
        tracker.add_log(
            "System",
            f"Python keyword filter: {len(all_lots)} -> {len(filtered_lots)} lots "
            f"({len(all_lots) - len(filtered_lots)} discarded)"
        )

        # Log per-phrase match breakdown so the user can see which terms hit
        for phrase in phrases:
            count = sum(
                1 for lot in all_lots
                if _phrase_matches(phrase, f"{lot.get('title', '')} {lot.get('description', '')}".lower())
            )
            tracker.add_log("Filter", f'"{phrase}" matched {count} of {len(all_lots)} lots')

        # Register filtered lots in the tracker (this is the canonical lot list)
        tracker.register_filtered_lots(filtered_lots)
        _update_ui()

        if not filtered_lots:
            tracker.finished = True
            tracker.error = Exception(
                f"No lots matched your search phrases. The catalog had {len(all_lots)} lots "
                f"but none matched: {', '.join(phrases)}"
            )
            _update_ui()
            sys.stdout = original_stdout
            elapsed = time.time() - start_time
            st.error(f"**No matching lots found after {elapsed:.1f}s.**\n\n"
                     f"The catalog had {len(all_lots)} lots but none matched your keywords.")
            st.stop()

        # ═════════════════════════════════════════════════════════════════
        # PHASE 1b — Risk Assessment + Detail Extraction
        # ═════════════════════════════════════════════════════════════════
        tracker.set_phase("Phase 1b: Risk Assessment", step=f"Assessing {len(filtered_lots)} lots...")
        tracker.add_log("System", f"Phase 1b: Risk assessment on {len(filtered_lots)} filtered lots")
        _update_ui()

        crew_1b = ScreeningCrewPartB().crew()
        crew_1b.step_callback = make_step_callback(tracker)
        crew_1b.task_callback = make_task_callback(tracker)

        result_1b = crew_1b.kickoff(inputs={
            **inputs,
            "filtered_lots_json": json.dumps(filtered_lots, indent=2),
        })

        extracted_lots = parse_extracted_lots(result_1b)
        if not extracted_lots:
            # Fallback: use filtered_lots directly if parsing failed
            extracted_lots = filtered_lots

        # Parse rejection reasons from risk assessment output
        rejection_reasons = parse_rejection_reasons(result_1b)

        tracker.total_risk_passed = len(extracted_lots)
        tracker.add_log("System", f"Phase 1b complete: {len(extracted_lots)} lots passed risk assessment")
        if rejection_reasons:
            tracker.add_log("Risk Officer", f"Rejected {len(rejection_reasons)} lots with justification")

        # Update tracker — mark risk-removed lots with specific reasons
        extracted_nums = {str(l.get("lot_number", l.get("lot_num", ""))) for l in extracted_lots}
        for lot_num in list(tracker._known_lots):
            if lot_num not in extracted_nums:
                reason = rejection_reasons.get(lot_num, "Risk flag (no details)")
                tracker.remove_lot(lot_num, reason)
            else:
                tracker.update_lot(lot_num, status="passed", stage="Risk+Extract")

        _update_ui()

        # ═════════════════════════════════════════════════════════════════
        # PHASE 2 — Per-Lot Market Validation
        # ═════════════════════════════════════════════════════════════════
        tracker.set_phase("Phase 2: Market Validation", lot_total=len(extracted_lots))
        tracker.add_log("System", f"Phase 2: Validating {len(extracted_lots)} lots (one crew per lot)")
        _update_ui()

        validated_lots = []
        for i, lot in enumerate(extracted_lots):
            lot_num = str(lot.get("lot_number", lot.get("lot_num", i + 1)))
            tracker.set_phase("Phase 2: Market Validation", lot_current=i + 1, lot_total=len(extracted_lots))
            tracker.update_lot(lot_num, status="validating", stage="Market Validation")
            tracker.add_log("System", f"Validating lot {lot_num} ({i + 1}/{len(extracted_lots)})")
            _update_ui()

            lot_start = time.time()

            try:
                crew_v = PerLotValidationCrew().crew()
                crew_v.step_callback = make_step_callback(tracker)
                crew_v.task_callback = make_task_callback(tracker)

                result_v = crew_v.kickoff(inputs={
                    "lot_data": json.dumps(lot, indent=2),
                    "buyer_premium_data": buyer_premium_data,
                    "platform_fee_choice": platform_fee_choice,
                })

                lot_result = parse_per_lot_result(result_v)
                # Ensure lot identifiers are preserved
                lot_result["lot_number"] = lot_num
                lot_result["title"] = lot.get("title", "")
                lot_result["estimate_low"] = lot.get("estimate_low")
                lot_result["estimate_high"] = lot.get("estimate_high")
                validated_lots.append(lot_result)

                # Update tracker with results
                margin = lot_result.get("expected_profit_margin_pct", lot_result.get("estimated_profit_margin_percentage", ""))
                fmv = lot_result.get("initial_fmv_estimate", "")
                rec = lot_result.get("investment_recommendation", "")
                tracker.update_lot(lot_num, status="passed", stage="Validated",
                                   margin=f"{margin}%" if margin else "",
                                   fmv=f"£{fmv}" if fmv else "",
                                   recommendation=rec)
                if margin:
                    tracker.add_insight(lot_num, f"Validation margin: {margin}%", "Market Validator")

            except Exception as exc:
                tracker.add_log("System", f"Lot {lot_num} validation failed: {exc}")
                tracker.update_lot(lot_num, status="rejected", rejection_reason="Validation failed")

            tracker.record_lot_time(time.time() - lot_start)
            _update_ui()

        tracker.total_validated = len(validated_lots)
        tracker.add_log("System", f"Phase 2 complete: {len(validated_lots)} lots validated")

        # ═════════════════════════════════════════════════════════════════
        # SELECT TOP 40%
        # ═════════════════════════════════════════════════════════════════
        top_lots = select_top_lots(validated_lots, top_pct=0.4)
        tracker.add_log("System", f"Selected top {len(top_lots)} lots for deep research (40% of {len(validated_lots)})")
        _update_ui()

        # ═════════════════════════════════════════════════════════════════
        # PHASE 3 — Per-Lot Deep Research
        # ═════════════════════════════════════════════════════════════════
        tracker.set_phase("Phase 3: Deep Research", lot_total=len(top_lots))
        tracker.add_log("System", f"Phase 3: Deep researching {len(top_lots)} lots (one crew per lot)")
        _update_ui()

        researched_lots = []
        for i, lot in enumerate(top_lots):
            lot_num = str(lot.get("lot_number", lot.get("lot_num", i + 1)))
            tracker.set_phase("Phase 3: Deep Research", lot_current=i + 1, lot_total=len(top_lots))
            tracker.update_lot(lot_num, status="researching", stage="Deep Research")
            tracker.add_log("System", f"Deep researching lot {lot_num} ({i + 1}/{len(top_lots)})")
            _update_ui()

            lot_start = time.time()

            try:
                crew_r = PerLotDeepResearchCrew().crew()
                crew_r.step_callback = make_step_callback(tracker)
                crew_r.task_callback = make_task_callback(tracker)

                result_r = crew_r.kickoff(inputs={
                    "lot_data": json.dumps(lot, indent=2),
                    "buyer_premium_data": buyer_premium_data,
                    "platform_fee_choice": platform_fee_choice,
                })

                lot_result = parse_per_lot_result(result_r)
                lot_result["lot_number"] = lot_num
                lot_result["title"] = lot.get("title", "")
                researched_lots.append(lot_result)

                # Update tracker
                margin = lot_result.get("expected_profit_margin_pct", "")
                fmv = lot_result.get("comprehensive_fmv_used", lot_result.get("seller_net_fmv", ""))
                rec = lot_result.get("final_investment_recommendation", "")
                tracker.update_lot(lot_num, status="complete", stage="Researched",
                                   margin=f"{margin}%" if margin else "",
                                   fmv=f"£{fmv}" if fmv else "",
                                   recommendation=rec)
                if margin:
                    tracker.add_insight(lot_num, f"Deep research margin: {margin}%", "Deep Research")

            except Exception as exc:
                tracker.add_log("System", f"Lot {lot_num} deep research failed: {exc}")
                tracker.update_lot(lot_num, status="rejected", rejection_reason="Research failed")

            tracker.record_lot_time(time.time() - lot_start)
            _update_ui()

        tracker.total_researched = len(researched_lots)
        tracker.add_log("System", f"Phase 3 complete: {len(researched_lots)} lots researched")

        # ═════════════════════════════════════════════════════════════════
        # PHASE 4 — Synthesis
        # ═════════════════════════════════════════════════════════════════
        tracker.set_phase("Phase 4: Synthesis", step="Ranking, archiving, generating bidding sheet...")
        tracker.add_log("System", f"Phase 4: Synthesizing {len(researched_lots)} researched lots")
        _update_ui()

        crew_s = SynthesisCrew().crew()
        crew_s.step_callback = make_step_callback(tracker)
        crew_s.task_callback = make_task_callback(tracker)

        final_result = crew_s.kickoff(inputs={
            "all_researched_lots": json.dumps(researched_lots, indent=2),
            "buyer_premium_data": buyer_premium_data,
            "platform_fee_choice": platform_fee_choice,
        })

        tracker.finished = True
        tracker.result = final_result
        tracker.add_log("System", "Workflow complete!")

    except Exception as exc:
        # #region agent log — outer exception
        import traceback as _tb3
        _dbg("OUTER_EXCEPTION", {"error": str(exc), "type": str(type(exc)), "traceback": _tb3.format_exc()}, "OUTER")
        # #endregion
        tracker.finished = True
        tracker.error = exc
        tracker.add_log("System", f"Workflow failed: {exc}")

    finally:
        sys.stdout = original_stdout

    elapsed = time.time() - start_time

    # ── Final UI render ──────────────────────────────────────────────────
    _update_ui()

    st.divider()
    if tracker.error:
        st.error(f"**Workflow failed after {elapsed:.1f}s:**\n\n```\n{tracker.error}\n```")
    else:
        st.success(f"**Workflow finished in {elapsed:.1f} seconds.**")
        result = tracker.result
        raw_output = getattr(result, "raw", str(result))
        render_final_result(raw_output)
        render_final_sources(raw_output)

        task_outputs = getattr(result, "tasks_output", None)
        if task_outputs:
            st.divider()
            st.subheader("Individual Task Outputs")
            for i, t_out in enumerate(task_outputs):
                name = getattr(t_out, "name", None) or getattr(t_out, "description", f"Task {i+1}")
                label = (str(name)[:100] + "...") if len(str(name)) > 100 else str(name)
                with st.expander(f"Task {i+1}: {label}", expanded=False):
                    t_raw = getattr(t_out, "raw", str(t_out))
                    render_final_result(t_raw)
