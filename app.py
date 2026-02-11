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
import time
import threading
from datetime import datetime
from dataclasses import dataclass, field

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
            with t._lock:
                t.current_task_desc = m.group(1).strip()[:250]
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
            with t._lock:
                t.current_tool_input = m.group(1).strip()[:300]
            return

        # NO parse_content here — this was the main leak causing phantom lots


# ═══════════════════════════════════════════════════════════════════════════
# CALLBACK HANDLERS
# ═══════════════════════════════════════════════════════════════════════════

def make_step_callback(tracker: ProgressTracker):
    """Returns a step_callback function for a Crew."""
    from crewai.agents.parser import AgentAction, AgentFinish

    def step_callback(step_output):
        try:
            if isinstance(step_output, AgentAction):
                thought = step_output.thought or ""
                tool = step_output.tool or ""
                tool_input = step_output.tool_input or ""

                # Display thought (but suppress parse failure message)
                if thought and thought != "Failed to parse LLM response":
                    tracker.set_thought(thought[:500])
                if tool:
                    tracker.set_tool(tool, str(tool_input)[:300])
                    tracker.add_log(tracker.current_agent, f"Tool: {tool}")

                # NO parse_content — prevents phantom lots from tool results

            elif isinstance(step_output, AgentFinish):
                thought = step_output.thought or ""
                # Suppress the benign "Failed to parse LLM response" message
                if thought and thought != "Failed to parse LLM response":
                    tracker.set_thought(thought[:500])
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

def _extract_json_from_output(raw: str) -> list | dict | None:
    """Extract JSON from a crew output, handling ```json code fences."""
    if not raw:
        return None

    # Try to find JSON inside ```json ... ``` code fence
    m = re.search(r'```json\s*\n?(.*?)```', raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except (json.JSONDecodeError, TypeError):
            pass

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

    return None


def parse_lots_from_output(result) -> list[dict]:
    """Extract lot array from Scout crew output (Phase 1a)."""
    raw = getattr(result, "raw", str(result))

    # Try task outputs — the scout catalog is the last task (index 2)
    task_outputs = getattr(result, "tasks_output", None)
    if task_outputs and len(task_outputs) >= 3:
        scout_raw = getattr(task_outputs[2], "raw", "")
        parsed = _extract_json_from_output(scout_raw)
        if isinstance(parsed, list):
            return parsed

    # Fall back to main output
    parsed = _extract_json_from_output(raw)
    if isinstance(parsed, list):
        return parsed

    return []


def extract_buyer_premium(result) -> str:
    """Extract buyer premium data string from Phase 1a output."""
    task_outputs = getattr(result, "tasks_output", None)
    if task_outputs and len(task_outputs) >= 2:
        return getattr(task_outputs[1], "raw", "")
    return ""


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

        if lot_tot > 0:
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
    initial_sidebar_state="collapsed",
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
        font-size: 1.1rem; color: #777; margin-bottom: 1.6rem; line-height: 1.5;
    }
    .filter-card {
        background: #f8f9fa; border-radius: 12px; padding: 1.25rem 1.4rem;
        border: 1px solid #e0e0e0;
    }
    .filter-card h4 { margin: 0 0 0.25rem 0; font-size: 1.05rem; }
    .filter-card .desc { font-size: 0.82rem; color: #888; margin-bottom: 0.9rem; }
    .search-preview {
        background: #f0f7ff; border-left: 4px solid #1976d2;
        padding: 0.9rem 1.2rem; border-radius: 0 8px 8px 0;
        margin: 0.8rem 0 0.4rem 0; font-size: 0.93rem; line-height: 1.6;
    }
    .search-preview code {
        background: #d6e8fa; padding: 2px 6px; border-radius: 4px;
        font-size: 0.88em;
    }
    .step-row { display: flex; gap: 1.2rem; margin: 1.2rem 0 0.4rem 0; }
    .step-card {
        flex: 1; background: #fafafa; border: 1px solid #eee;
        border-radius: 10px; padding: 1rem 1.1rem; text-align: center;
    }
    .step-card .num {
        display: inline-block; background: #1976d2; color: #fff;
        width: 28px; height: 28px; line-height: 28px; border-radius: 50%;
        font-weight: 700; font-size: 0.85rem; margin-bottom: 0.4rem;
    }
    .step-card .label { font-weight: 600; font-size: 0.95rem; margin-bottom: 0.2rem; }
    .step-card .detail { font-size: 0.78rem; color: #888; }
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
    st.subheader("API Key Status")
    for key in ["GEMINI_API_KEY", "OPENAI_API_KEY", "SERPER_API_KEY",
                 "FIRECRAWL_API_KEY", "BROWSERBASE_API_KEY"]:
        icon = "Y" if os.environ.get(key) else "N"
        st.text(f"{icon}  {key}")
    st.divider()
    st.caption("Expand the sidebar to check API key connectivity.")


# ═══════════════════════════════════════════════════════════════════════════
# LANDING PAGE
# ═══════════════════════════════════════════════════════════════════════════

st.markdown(
    '<p class="hero-title">Fast Auction Research</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="hero-sub">'
    "Paste an auction catalog URL, build your search filters below, and let the "
    "AI crew scan every lot, validate market prices, and rank profit potential — "
    "all in one click."
    "</p>",
    unsafe_allow_html=True,
)

st.markdown("""
<div class="step-row">
  <div class="step-card">
    <div class="num">1</div>
    <div class="label">Extract</div>
    <div class="detail">Scout scrapes the full catalog; Python filters by your exact keywords</div>
  </div>
  <div class="step-card">
    <div class="num">2</div>
    <div class="label">Validate</div>
    <div class="detail">Per-lot market validation with fresh token window for each lot</div>
  </div>
  <div class="step-card">
    <div class="num">3</div>
    <div class="label">Research</div>
    <div class="detail">Deep research on top 40% of lots, each with its own context</div>
  </div>
  <div class="step-card">
    <div class="num">4</div>
    <div class="label">Report</div>
    <div class="detail">Ranked by profit margin; mobile bidding sheet saved to Box</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("")

# ── Auction URL ──────────────────────────────────────────────────────────
auction_url = st.text_input(
    "Auction Catalog URL",
    placeholder="https://www.example-auction.com/sale/12345",
    help="Full URL of the auction catalog page to scan.",
)

st.markdown("---")
st.markdown("### Search Filters")
st.markdown(
    '<div class="filter-card">'
    "<h4>Search Phrases</h4>"
    '<div class="desc">Enter brands, designers, or item types separated by commas. '
    "A lot is included if it matches <b>any one</b> of your phrases. "
    "Multi-word phrases match only if <b>all words</b> appear in the lot "
    '(e.g. <code>Bottega Veneta</code> matches lots containing both "Bottega" and "Veneta").</div>'
    "</div>",
    unsafe_allow_html=True,
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
    cols_per_row = min(n, 6)
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

# ── Bidding Fee Option ───────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Bidding Fee")
st.caption(
    "Most online bidding platforms charge a **3% internet surcharge** on top of the hammer price. "
    "Some allow you to pay a **flat registration fee** instead to avoid the surcharge."
)
platform_fee_paid = st.toggle(
    "I will pay the flat registration fee to avoid the 3% online bidding surcharge",
    value=False,
    help="Enable this if you plan to pay the flat fee to register at the auction, "
         "which removes the per-lot 3% internet surcharge from your cost calculations.",
)

# ── Live search preview ──────────────────────────────────────────────────
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
        ScreeningCrewPartA,
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

    # Summary stats row
    stats_ph = st.empty()

    # Two-column layout: phase progress + agent activity
    col_progress, col_agent = st.columns([2, 3])
    with col_progress:
        progress_ph = st.empty()
    with col_agent:
        activity_ph = st.empty()

    # Lot tracker (full width)
    lots_ph = st.empty()

    # Insights and log side by side
    col_insights, col_log = st.columns([1, 1])
    with col_insights:
        insights_ph = st.empty()
    with col_log:
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

    try:
        # ═════════════════════════════════════════════════════════════════
        # PHASE 1a — Catalog Extraction
        # ═════════════════════════════════════════════════════════════════
        tracker.set_phase("Phase 1a: Extracting Catalog", step="Scout scraping all lots...")
        tracker.add_log("System", "Phase 1a: Starting catalog extraction")
        _update_ui()

        crew_1a = ScreeningCrewPartA().crew()
        crew_1a.step_callback = make_step_callback(tracker)
        crew_1a.task_callback = make_task_callback(tracker)

        result_1a = crew_1a.kickoff(inputs=inputs)

        all_lots = parse_lots_from_output(result_1a)
        buyer_premium_data = extract_buyer_premium(result_1a)

        tracker.total_catalog_lots = len(all_lots)
        tracker.add_log("System", f"Catalog extracted: {len(all_lots)} total lots")

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
