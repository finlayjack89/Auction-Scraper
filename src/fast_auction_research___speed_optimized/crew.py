"""
Multi-crew architecture for Fast Auction Research.

6 crew classes, each with its own token window:
  - CatalogSetupCrew: URL validation + buyer premium discovery (Phase 1a setup)
  - PageExtractionCrew: Extract lots from ONE catalog page (called per-page)
  - ScreeningCrewPartB: Risk assessment + detail extraction (receives filtered lots)
  - PerLotValidationCrew: Market validation + profit calc for ONE lot
  - PerLotDeepResearchCrew: Deep research + profit analysis for ONE lot
  - SynthesisCrew: Final ranking, archive, bidding sheet
"""

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
# SHARED LLM CONFIGS — max output tokens set to model ceiling everywhere
# ═══════════════════════════════════════════════════════════════════════════
_GEMINI_MAX_TOKENS = 65536  # absolute max for gemini-3-flash/pro-preview

def _flash(temperature: float = 0.3) -> LLM:
    """Gemini 3 Flash with max output tokens."""
    return LLM(
        model="gemini/gemini-3-flash-preview",
        temperature=temperature,
        max_tokens=_GEMINI_MAX_TOKENS,
    )

def _pro(temperature: float = 0.5) -> LLM:
    """Gemini 3 Pro with max output tokens."""
    return LLM(
        model="gemini/gemini-3-pro-preview",
        temperature=temperature,
        max_tokens=_GEMINI_MAX_TOKENS,
    )

_MAX_ITER = 100


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 1a — CATALOG SETUP CREW (validation + buyer premium only)
# ═══════════════════════════════════════════════════════════════════════════

@CrewBase
class CatalogSetupCrew:
    """Validates the auction URL and discovers buyer premium. No lot extraction."""
    tasks_config = "config/tasks_1a.yaml"

    @agent
    def scout___auction_navigator_keyword_filter(self) -> Agent:
        return Agent(
            config=self.agents_config["scout___auction_navigator_keyword_filter"],
            tools=[
                FirecrawlScrapeWebsiteTool(),
                ScrapeWebsiteTool(),
                SerperScrapeWebsiteTool(),
                ScrapeElementFromWebsiteTool(),
            ],
            reasoning=False,
            inject_date=True,
            allow_delegation=False,
            max_iter=_MAX_ITER,
            llm=_flash(temperature=0.4),
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

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            chat_llm=_flash(),
        )


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 1a — PER-PAGE EXTRACTION CREW (one page at a time)
# ═══════════════════════════════════════════════════════════════════════════

@CrewBase
class PageExtractionCrew:
    """Extracts lots from a SINGLE catalog page. Called once per page."""
    tasks_config = "config/tasks_1a.yaml"

    @agent
    def scout___auction_navigator_keyword_filter(self) -> Agent:
        return Agent(
            config=self.agents_config["scout___auction_navigator_keyword_filter"],
            tools=[
                FirecrawlScrapeWebsiteTool(),
                ScrapeWebsiteTool(),
                SerperScrapeWebsiteTool(),
                ScrapeElementFromWebsiteTool(),
            ],
            reasoning=False,
            inject_date=True,
            allow_delegation=False,
            max_iter=_MAX_ITER,
            llm=_flash(temperature=0.4),
        )

    @task
    def extract_single_page(self) -> Task:
        return Task(
            config=self.tasks_config["extract_single_page"],
            markdown=False,
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            chat_llm=_flash(),
        )


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 1b — SCREENING CREW PART B
# ═══════════════════════════════════════════════════════════════════════════

@CrewBase
class ScreeningCrewPartB:
    """Risk assessment + detail extraction on pre-filtered lots."""
    tasks_config = "config/tasks_1b.yaml"

    @agent
    def risk_officer___compliance_filtration_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config["risk_officer___compliance_filtration_specialist"],
            tools=[],
            reasoning=False,
            inject_date=True,
            allow_delegation=False,
            max_iter=_MAX_ITER,
            llm=_flash(temperature=0.1),
        )

    @agent
    def extractor___item_detail_parser(self) -> Agent:
        return Agent(
            config=self.agents_config["extractor___item_detail_parser"],
            tools=[],
            reasoning=False,
            inject_date=True,
            allow_delegation=False,
            max_iter=_MAX_ITER,
            llm=_flash(temperature=0.1),
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
            chat_llm=_flash(),
        )


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 2 — PER-LOT VALIDATION CREW
# ═══════════════════════════════════════════════════════════════════════════

@CrewBase
class PerLotValidationCrew:
    """Market validation + profit calculation for a single lot."""
    tasks_config = "config/tasks_validation.yaml"

    @agent
    def market_validator___rapid_assessment_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config["market_validator___rapid_assessment_specialist"],
            tools=[SerperDevTool(), SerperScrapeWebsiteTool(), FirecrawlScrapeWebsiteTool()],
            reasoning=False,
            inject_date=True,
            allow_delegation=False,
            max_iter=40,  # tighter — this is a rapid screening agent
            llm=_flash(temperature=0.15),
        )

    @agent
    def quant___financial_analysis_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config["quant___financial_analysis_specialist"],
            tools=[ScrapeWebsiteTool()],
            reasoning=False,
            inject_date=True,
            allow_delegation=False,
            max_iter=_MAX_ITER,
            llm=_flash(temperature=0.1),
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
            chat_llm=_flash(),
        )


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 3 — PER-LOT DEEP RESEARCH CREW
# ═══════════════════════════════════════════════════════════════════════════

@CrewBase
class PerLotDeepResearchCrew:
    """Deep market research + profit analysis for a single lot."""
    tasks_config = "config/tasks_research.yaml"

    @agent
    def deep_research_analyst___comprehensive_market_intelligence(self) -> Agent:
        return Agent(
            config=self.agents_config["deep_research_analyst___comprehensive_market_intelligence"],
            tools=[SerperDevTool(), FirecrawlScrapeWebsiteTool(), ScrapeWebsiteTool()],
            reasoning=False,
            inject_date=True,
            allow_delegation=False,
            max_iter=_MAX_ITER,
            llm=_pro(temperature=0.5),
        )

    @agent
    def quant___financial_analysis_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config["quant___financial_analysis_specialist"],
            tools=[ScrapeWebsiteTool()],
            reasoning=False,
            inject_date=True,
            allow_delegation=False,
            max_iter=_MAX_ITER,
            llm=_flash(temperature=0.1),
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
            chat_llm=_flash(),
        )


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 4 — SYNTHESIS CREW
# ═══════════════════════════════════════════════════════════════════════════

@CrewBase
class SynthesisCrew:
    """Final ranking, archive, and bidding sheet generation."""
    tasks_config = "config/tasks_synthesis.yaml"

    @agent
    def quant___financial_analysis_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config["quant___financial_analysis_specialist"],
            tools=[ScrapeWebsiteTool()],
            reasoning=False,
            inject_date=True,
            allow_delegation=False,
            max_iter=_MAX_ITER,
            llm=_flash(temperature=0.1),
        )

    @agent
    def archivist___data_curator_storage_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config["archivist___data_curator_storage_specialist"],
            tools=[FileReadTool()],
            reasoning=False,
            inject_date=True,
            allow_delegation=False,
            max_iter=_MAX_ITER,
            apps=["box/save_file_from_object"],
            llm=_flash(temperature=0.3),
        )

    @agent
    def mobile_report_generator___bidding_sheet_formatter(self) -> Agent:
        return Agent(
            config=self.agents_config["mobile_report_generator___bidding_sheet_formatter"],
            tools=[],
            reasoning=False,
            inject_date=True,
            allow_delegation=False,
            max_iter=_MAX_ITER,
            apps=["box/save_file_from_object"],
            llm=_flash(temperature=0.3),
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
            chat_llm=_flash(),
        )
