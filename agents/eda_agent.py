"""
EDA Agent — Orchestrates insight detection and story generation.

Takes the cleaned DataFrame from Sprint 1 and produces
a full EDA narrative with structured insights.
"""

import pandas as pd
from dataclasses import dataclass
from core.insight_engine import InsightEngine, Insight
from core.story_generator import StoryGenerator


@dataclass
class EDAResult:
    """Container for EDA pipeline output."""
    insights: list[Insight]
    executive_summary: str
    findings: list[dict]
    recommendations: list[dict]
    markdown_report: str


class EDAAgent:
    """
    Sprint 2 orchestrator: Detect → Analyze → Narrate.
    
    Usage:
        agent = EDAAgent()
        result = agent.run(cleaned_df, raw_df, filename)
    """

    def run(
        self,
        cleaned_df: pd.DataFrame,
        raw_df: pd.DataFrame = None,
        filename: str = "dataset",
    ) -> EDAResult:
        """
        Run the full EDA pipeline.
        
        Args:
            cleaned_df: Cleaned DataFrame from Sprint 1
            raw_df: Original raw DataFrame (optional, for comparison)
            filename: Original filename for the report
            
        Returns:
            EDAResult with insights, narrative, and report
        """
        # Step 1: Detect insights
        engine = InsightEngine(cleaned_df, raw_df)
        insights = engine.run()

        # Step 2: Generate narrative
        story = StoryGenerator(insights, filename)
        executive_summary = story.generate_executive_summary()
        findings = story.generate_findings()
        recommendations = story.generate_recommendations()
        markdown_report = story.generate_markdown_report()

        return EDAResult(
            insights=insights,
            executive_summary=executive_summary,
            findings=findings,
            recommendations=recommendations,
            markdown_report=markdown_report,
        )