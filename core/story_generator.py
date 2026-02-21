"""
Story Generator — Converts structured insights into a stakeholder-ready narrative.

Takes the list of Insight objects from InsightEngine and produces:
    1. An executive summary
    2. Detailed findings organized by category
    3. Prioritized recommendations
    4. A downloadable markdown report
"""

from datetime import datetime
from core.insight_engine import Insight, InsightType, Severity


class StoryGenerator:
    """Generates natural language EDA narrative from insights."""

    SECTION_ORDER = [
        InsightType.SUMMARY,
        InsightType.OUTLIER,
        InsightType.CORRELATION,
        InsightType.SKEWNESS,
        InsightType.IMBALANCE,
        InsightType.HIGH_CARDINALITY,
        InsightType.LOW_VARIANCE,
        InsightType.DISTRIBUTION,
    ]

    SECTION_TITLES = {
        InsightType.SUMMARY: "Dataset Overview",
        InsightType.OUTLIER: "Outlier Analysis",
        InsightType.CORRELATION: "Feature Correlations",
        InsightType.SKEWNESS: "Distribution Skewness",
        InsightType.IMBALANCE: "Class Balance",
        InsightType.HIGH_CARDINALITY: "Cardinality Check",
        InsightType.LOW_VARIANCE: "Variance Analysis",
        InsightType.DISTRIBUTION: "Distribution Profiles",
    }

    SEVERITY_EMOJI = {
        Severity.CRITICAL: "🔴",
        Severity.WARNING: "🟡",
        Severity.INFO: "🔵",
    }

    def __init__(self, insights: list[Insight], filename: str = "dataset"):
        self.insights = insights
        self.filename = filename

    def generate_executive_summary(self) -> str:
        """One-paragraph executive summary."""
        critical = [i for i in self.insights if i.severity == Severity.CRITICAL]
        warnings = [i for i in self.insights if i.severity == Severity.WARNING]
        info = [i for i in self.insights if i.severity == Severity.INFO]

        summary_insight = next(
            (i for i in self.insights if i.insight_type == InsightType.SUMMARY), None
        )
        overview = summary_insight.description if summary_insight else "Dataset analyzed."

        parts = [overview]

        if critical:
            parts.append(
                f" There are {len(critical)} critical finding(s) that require immediate attention."
            )
        if warnings:
            parts.append(
                f" {len(warnings)} warning(s) were flagged for review."
            )
        parts.append(
            f" A total of {len(self.insights)} insights were discovered across "
            f"outlier detection, correlation analysis, distribution profiling, and more."
        )

        return " ".join(parts)

    def generate_findings(self) -> list[dict]:
        """
        Organize insights into presentation-ready sections.
        
        Returns:
            List of dicts with keys: section, emoji, insights (list of formatted dicts)
        """
        findings = []

        for itype in self.SECTION_ORDER:
            section_insights = [i for i in self.insights if i.insight_type == itype]
            if not section_insights:
                continue

            formatted = []
            for insight in section_insights:
                formatted.append({
                    "severity": insight.severity,
                    "severity_emoji": self.SEVERITY_EMOJI[insight.severity],
                    "title": insight.title,
                    "description": insight.description,
                    "recommendation": insight.recommendation,
                    "affected_columns": insight.affected_columns,
                    "metrics": insight.metrics,
                })

            findings.append({
                "section": self.SECTION_TITLES.get(itype, itype.value),
                "insights": formatted,
            })

        return findings

    def generate_recommendations(self) -> list[dict]:
        """Extract and prioritize all recommendations."""
        recs = []
        for insight in self.insights:
            if insight.recommendation:
                recs.append({
                    "severity": insight.severity,
                    "emoji": self.SEVERITY_EMOJI[insight.severity],
                    "context": insight.title,
                    "recommendation": insight.recommendation,
                })

        # Sort: critical first
        severity_order = {Severity.CRITICAL: 0, Severity.WARNING: 1, Severity.INFO: 2}
        recs.sort(key=lambda x: severity_order[x["severity"]])
        return recs

    def generate_markdown_report(self) -> str:
        """Generate a full downloadable markdown report."""
        lines = []
        now = datetime.now().strftime("%Y-%m-%d %H:%M")

        lines.append(f"# EDA Report: {self.filename}")
        lines.append(f"*Generated on {now} by Dataset Curation Agent*\n")

        # Executive Summary
        lines.append("## Executive Summary\n")
        lines.append(self.generate_executive_summary() + "\n")

        # Severity counts
        critical = sum(1 for i in self.insights if i.severity == Severity.CRITICAL)
        warnings = sum(1 for i in self.insights if i.severity == Severity.WARNING)
        info = sum(1 for i in self.insights if i.severity == Severity.INFO)
        lines.append(f"| Severity | Count |")
        lines.append(f"|----------|-------|")
        lines.append(f"| 🔴 Critical | {critical} |")
        lines.append(f"| 🟡 Warning | {warnings} |")
        lines.append(f"| 🔵 Info | {info} |")
        lines.append("")

        # Findings
        findings = self.generate_findings()
        for section in findings:
            lines.append(f"## {section['section']}\n")
            for insight in section["insights"]:
                lines.append(f"### {insight['severity_emoji']} {insight['title']}\n")
                lines.append(f"{insight['description']}\n")
                if insight["affected_columns"]:
                    lines.append(f"**Affected columns:** {', '.join(insight['affected_columns'])}\n")
                if insight["recommendation"]:
                    lines.append(f"> **Recommendation:** {insight['recommendation']}\n")

        # Prioritized Recommendations
        recs = self.generate_recommendations()
        if recs:
            lines.append("## Prioritized Recommendations\n")
            for i, rec in enumerate(recs, 1):
                lines.append(f"{i}. {rec['emoji']} **{rec['context']}**: {rec['recommendation']}\n")

        lines.append("---\n")
        lines.append("*Report generated by Dataset Curation Agent — Sprint 2*")

        return "\n".join(lines)