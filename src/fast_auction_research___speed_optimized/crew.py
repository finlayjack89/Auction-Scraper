"""
Multi-crew architecture for Fast Auction Research.

5 crew classes, each with its own token window:
  - ScreeningCrewPartA: URL validation, buyer premium, catalog extraction
  - ScreeningCrewPartB: Risk assessment + detail extraction (receives filtered lots)
  - PerLotValidationCrew: Market validation + profit calc for ONE lot
  - PerLotDeepResearchCrew: Deep research + profit analysis for ONE lot
  - SynthesisCrew: Final ranking, archive, bidding sheet
"""

import os

from crewai import LLM
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import (
    SerperDevTool,
    ScrapeWebsiteTool,
    FileReadTool,
    FirecrawlScrapeWebsiteTool,
    SerperScrapeWebsiteTool,
    ScrapeElementFromWebsiteTool,
)


# ═══════════════════════════════════════════════════════════════════════════
# SHARED AGENT FACTORIES  (used by multiple crews)
# ═══════════════════════════════════════════════════════════════════════════

def _make_scout() -> Agent:
    return Agent(
        config="scout___auction_navigator_keyword_filter",
        tools=[
            ScrapeWebsiteTool(),
            SerperScrapeWebsiteTool(),
            ScrapeElementFromWebsiteTool(),
        ],
        reasoning=False,
        inject_date=True,
        allow_delegation=False,
        max_iter=25,
        llm=LLM(model="gemini/gemini-3-flash-preview", temperature=0.4),
    )


def _make_risk_officer() -> Agent:
    return Agent(
        config="risk_officer___compliance_filtration_specialist",
        tools=[],
        reasoning=False,
        inject_date=True,
        allow_delegation=False,
        max_iter=25,
        llm=LLM(model="gemini/gemini-3-flash-preview", temperature=0.1),
    )


def _make_extractor() -> Agent:
    return Agent(
        config="extractor___item_detail_parser",
        tools=[],
        reasoning=False,
        inject_date=True,
        allow_delegation=False,
        max_iter=25,
        llm=LLM(model="gemini/gemini-3-flash-preview", temperature=0.1),
    )


def _make_market_validator() -> Agent:
    return Agent(
        config="market_validator___rapid_assessment_specialist",
        tools=[SerperDevTool(), FirecrawlScrapeWebsiteTool()],
        reasoning=False,
        inject_date=True,
        allow_delegation=False,
        max_iter=25,
        llm=LLM(model="gemini/gemini-3-pro-preview", temperature=0.4),
    )


def _make_quant() -> Agent:
    return Agent(
        config="quant___financial_analysis_specialist",
        tools=[ScrapeWebsiteTool()],
        reasoning=False,
        inject_date=True,
        allow_delegation=False,
        max_iter=25,
        llm=LLM(model="gemini/gemini-3-flash-preview", temperature=0.1),
    )


def _make_deep_researcher() -> Agent:
    return Agent(
        config="deep_research_analyst___comprehensive_market_intelligence",
        tools=[SerperDevTool(), FirecrawlScrapeWebsiteTool(), ScrapeWebsiteTool()],
        reasoning=False,
        inject_date=True,
        allow_delegation=False,
        max_iter=25,
        llm=LLM(model="gemini/gemini-3-pro-preview", temperature=0.5),
    )


def _make_archivist() -> Agent:
    return Agent(
        config="archivist___data_curator_storage_specialist",
        tools=[FileReadTool()],
        reasoning=False,
        inject_date=True,
        allow_delegation=False,
        max_iter=25,
        apps=["box/save_file_from_object"],
        llm=LLM(model="gemini/gemini-3-flash-preview", temperature=0.3),
    )


def _make_report_generator() -> Agent:
    return Agent(
        config="mobile_report_generator___bidding_sheet_formatter",
        tools=[],
        reasoning=False,
        inject_date=True,
        allow_delegation=False,
        max_iter=25,
        apps=["box/save_file_from_object"],
        llm=LLM(model="gemini/gemini-3-flash-preview", temperature=0.3),
    )


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 1a — SCREENING CREW PART A
# ═══════════════════════════════════════════════════════════════════════════

@CrewBase
class ScreeningCrewPartA:
    """Extracts catalog + discovers buyer premium. Outputs ALL lots as JSON."""

    @agent
    def scout___auction_navigator_keyword_filter(self) -> Agent:
        return Agent(
            config=self.agents_config["scout___auction_navigator_keyword_filter"],
            tools=[
                ScrapeWebsiteTool(),
                SerperScrapeWebsiteTool(),
                ScrapeElementFromWebsiteTool(),
            ],
            reasoning=False,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            llm=LLM(model="gemini/gemini-3-flash-preview", temperature=0.4),
        )

    @task
    def input_validation_error_handling(self) -> Task:
        return Task(
            config=self.tasks_config["input_validation_error_handling"],
            markdown=False,
        )

    @task
    def discover_buyer_premium(self) -> Task:
        return Task(
            config=self.tasks_config["discover_buyer_premium"],
            markdown=False,
        )

    @task
    def scout_auction_catalog(self) -> Task:
        return Task(
            config=self.tasks_config["scout_auction_catalog"],
            markdown=False,
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            chat_llm=LLM(model="gemini/gemini-3-flash-preview"),
        )


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 1b — SCREENING CREW PART B
# ═══════════════════════════════════════════════════════════════════════════

@CrewBase
class ScreeningCrewPartB:
    """Risk assessment + detail extraction on pre-filtered lots."""

    @agent
    def risk_officer___compliance_filtration_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config["risk_officer___compliance_filtration_specialist"],
            tools=[],
            reasoning=False,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            llm=LLM(model="gemini/gemini-3-flash-preview", temperature=0.1),
        )

    @agent
    def extractor___item_detail_parser(self) -> Agent:
        return Agent(
            config=self.agents_config["extractor___item_detail_parser"],
            tools=[],
            reasoning=False,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            llm=LLM(model="gemini/gemini-3-flash-preview", temperature=0.1),
        )

    @task
    def risk_assessment_filtering(self) -> Task:
        return Task(
            config=self.tasks_config["risk_assessment_filtering"],
            markdown=False,
        )

    @task
    def extract_item_details(self) -> Task:
        return Task(
            config=self.tasks_config["extract_item_details"],
            markdown=False,
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            chat_llm=LLM(model="gemini/gemini-3-flash-preview"),
        )


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 2 — PER-LOT VALIDATION CREW
# ═══════════════════════════════════════════════════════════════════════════

@CrewBase
class PerLotValidationCrew:
    """Market validation + profit calculation for a single lot."""

    @agent
    def market_validator___rapid_assessment_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config["market_validator___rapid_assessment_specialist"],
            tools=[SerperDevTool(), FirecrawlScrapeWebsiteTool()],
            reasoning=False,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            llm=LLM(model="gemini/gemini-3-pro-preview", temperature=0.4),
        )

    @agent
    def quant___financial_analysis_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config["quant___financial_analysis_specialist"],
            tools=[ScrapeWebsiteTool()],
            reasoning=False,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            llm=LLM(model="gemini/gemini-3-flash-preview", temperature=0.1),
        )

    @task
    def per_lot_market_validation(self) -> Task:
        return Task(
            config=self.tasks_config["per_lot_market_validation"],
            markdown=False,
        )

    @task
    def per_lot_profit_calculation(self) -> Task:
        return Task(
            config=self.tasks_config["per_lot_profit_calculation"],
            markdown=False,
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            chat_llm=LLM(model="gemini/gemini-3-flash-preview"),
        )


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 3 — PER-LOT DEEP RESEARCH CREW
# ═══════════════════════════════════════════════════════════════════════════

@CrewBase
class PerLotDeepResearchCrew:
    """Deep market research + profit analysis for a single lot."""

    @agent
    def deep_research_analyst___comprehensive_market_intelligence(self) -> Agent:
        return Agent(
            config=self.agents_config["deep_research_analyst___comprehensive_market_intelligence"],
            tools=[SerperDevTool(), FirecrawlScrapeWebsiteTool(), ScrapeWebsiteTool()],
            reasoning=False,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            llm=LLM(model="gemini/gemini-3-pro-preview", temperature=0.5),
        )

    @agent
    def quant___financial_analysis_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config["quant___financial_analysis_specialist"],
            tools=[ScrapeWebsiteTool()],
            reasoning=False,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            llm=LLM(model="gemini/gemini-3-flash-preview", temperature=0.1),
        )

    @task
    def per_lot_deep_research(self) -> Task:
        return Task(
            config=self.tasks_config["per_lot_deep_research"],
            markdown=False,
        )

    @task
    def per_lot_deep_profit(self) -> Task:
        return Task(
            config=self.tasks_config["per_lot_deep_profit"],
            markdown=False,
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            chat_llm=LLM(model="gemini/gemini-3-flash-preview"),
        )


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 4 — SYNTHESIS CREW
# ═══════════════════════════════════════════════════════════════════════════

@CrewBase
class SynthesisCrew:
    """Final ranking, archive, and bidding sheet generation."""

    @agent
    def quant___financial_analysis_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config["quant___financial_analysis_specialist"],
            tools=[ScrapeWebsiteTool()],
            reasoning=False,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            llm=LLM(model="gemini/gemini-3-flash-preview", temperature=0.1),
        )

    @agent
    def archivist___data_curator_storage_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config["archivist___data_curator_storage_specialist"],
            tools=[FileReadTool()],
            reasoning=False,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            apps=["box/save_file_from_object"],
            llm=LLM(model="gemini/gemini-3-flash-preview", temperature=0.3),
        )

    @agent
    def mobile_report_generator___bidding_sheet_formatter(self) -> Agent:
        return Agent(
            config=self.agents_config["mobile_report_generator___bidding_sheet_formatter"],
            tools=[],
            reasoning=False,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            apps=["box/save_file_from_object"],
            llm=LLM(model="gemini/gemini-3-flash-preview", temperature=0.3),
        )

    @task
    def synthesis_final_ranking(self) -> Task:
        return Task(
            config=self.tasks_config["synthesis_final_ranking"],
            markdown=False,
        )

    @task
    def synthesis_archive(self) -> Task:
        return Task(
            config=self.tasks_config["synthesis_archive"],
            markdown=False,
        )

    @task
    def synthesis_bidding_sheet(self) -> Task:
        return Task(
            config=self.tasks_config["synthesis_bidding_sheet"],
            markdown=False,
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            chat_llm=LLM(model="gemini/gemini-3-flash-preview"),
        )
