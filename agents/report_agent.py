"""
LifeOS Agent - Report Agent
===============================
Agent 5: Intelligence Report Writer
Role: Write professional morning briefing reports.
Model: Gemini 2.5 Flash + Google ADK
Tools: ReportLab PDF, all agent outputs
"""

import logging
from datetime import datetime
from typing import Optional

from crewai import Agent, Task, Crew

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════
# LLM INITIALIZATION
# ═══════════════════════════════════════════


def _get_llm():
    """Get the best available LLM for report agent (primary: gemini)."""
    from config.llm_factory import get_llm_with_fallback
    return get_llm_with_fallback(primary="gemini")


# ═══════════════════════════════════════════
# GOOGLE ADK INTEGRATION
# ═══════════════════════════════════════════


def _enhance_with_adk(content: str, topic: str) -> str:
    """
    Optionally enhance report content using Google ADK.

    Args:
        content: The report content to enhance.
        topic: The report topic.

    Returns:
        Enhanced content string.
    """
    try:
        from google import genai
        from config.models import GEMINI_CONFIG

        api_key = GEMINI_CONFIG.get("api_key", "")
        if not api_key:
            return content

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=GEMINI_CONFIG["model_name"],
            contents=(
                f"You are a professional report editor. Polish and enhance the following "
                f"intelligence report about '{topic}'. Improve clarity, formatting, and "
                f"professional tone. Keep the structure intact but make it more impactful. "
                f"Do not add new information, only improve presentation.\n\n"
                f"Report:\n{content[:3000]}"
            ),
        )

        if response and response.text:
            logger.info("Report enhanced with Google ADK")
            return response.text
        return content

    except Exception as e:
        logger.warning(f"Google ADK enhancement failed (non-critical): {e}")
        return content


# ═══════════════════════════════════════════
# REPORT AGENT
# ═══════════════════════════════════════════


def run_report_agent(
    user_input: str,
    research_data: str,
    strategy_data: str,
    plan_data: str,
    goals_data: str,
    user_context: str = "",
) -> str:
    """
    Run the Report Agent to create a comprehensive intelligence report.

    Args:
        user_input: The user's original query/topic.
        research_data: Research findings.
        strategy_data: Strategic recommendations.
        plan_data: Action plan.
        goals_data: Goal analysis.
        user_context: User context.

    Returns:
        Complete formatted report content as a string.
    """
    try:
        llm = _get_llm()

        report_agent = Agent(
            role="Intelligence Report Writer",
            goal=(
                "Write professional, comprehensive morning briefing reports "
                "that synthesize all agent outputs into a cohesive document"
            ),
            backstory=(
                "You are an elite intelligence report writer, trained in the style of "
                "McKinsey consulting reports and presidential daily briefings. You synthesize "
                "complex information from multiple sources into clear, actionable documents. "
                "Your reports are known for their professional formatting, insightful analysis, "
                "and executive-ready presentation. You structure information hierarchically, "
                "lead with the most important insights, and always end with clear next steps. "
                "You use section headers, bullet points, and emphasis strategically."
            ),
            llm=llm,
            verbose=False,
            max_iter=3,
            allow_delegation=False,
        )

        date_str = datetime.now().strftime("%B %d, %Y")

        task_description = (
            f"Write a professional LifeOS Intelligence Report based on all the following agent outputs.\n\n"
            f"**Report Topic:** {user_input}\n"
            f"**Report Date:** {date_str}\n\n"
            f"═══ AGENT OUTPUTS ═══\n\n"
            f"**RESEARCH FINDINGS:**\n{research_data[:2000]}\n\n"
            f"**STRATEGIC ANALYSIS:**\n{strategy_data[:2000]}\n\n"
            f"**ACTION PLAN:**\n{plan_data[:2000]}\n\n"
            f"**GOALS TRACKER:**\n{goals_data[:1500]}\n\n"
            f"**USER CONTEXT:**\n{user_context[:800]}\n\n"
            f"═══ REPORT STRUCTURE ═══\n\n"
            f"Structure the report with these exact sections:\n\n"
            f"## Executive Summary\n"
            f"A 3-4 sentence overview of the key findings and recommendations.\n\n"
            f"## Research Intelligence\n"
            f"Key findings from research, organized by importance.\n"
            f"Include specific data points, trends, and citations.\n\n"
            f"## Strategic Direction\n"
            f"High-level strategy recommendations.\n"
            f"Vision, pillars, and key opportunities.\n\n"
            f"## Action Plan\n"
            f"Today's priorities, this week's tasks, and 30-day milestones.\n"
            f"Specific, time-bound, and measurable actions.\n\n"
            f"## Goals Progress\n"
            f"Current goal status, progress analysis, and recommendations.\n\n"
            f"## Key Takeaways\n"
            f"Top 5 most important insights from this report.\n\n"
            f"## Next Steps\n"
            f"The 3 most important things to do right now.\n\n"
            f"Write the report in a professional, engaging tone. "
            f"Use bullet points for lists, bold for emphasis, and ensure every section "
            f"provides specific, actionable value."
        )

        report_task = Task(
            description=task_description,
            expected_output=(
                "A complete, professionally formatted intelligence report with all "
                "sections: Executive Summary, Research Intelligence, Strategic Direction, "
                "Action Plan, Goals Progress, Key Takeaways, and Next Steps. The report "
                "should be comprehensive, well-organized, and ready for PDF generation."
            ),
            agent=report_agent,
        )

        crew = Crew(
            agents=[report_agent],
            tasks=[report_task],
            verbose=False,
        )

        result = crew.kickoff()
        report_content = str(result)

        # Optionally enhance with Google ADK
        try:
            report_content = _enhance_with_adk(report_content, user_input)
        except Exception as e:
            logger.warning(f"ADK enhancement skipped: {e}")

        logger.info(f"Report agent completed for query: {user_input[:50]}")
        return report_content

    except Exception as e:
        logger.error(f"Report agent failed: {e}")
        # Fallback: assemble raw data into a basic report
        date_str = datetime.now().strftime("%B %d, %Y")
        return (
            f"# LifeOS Intelligence Report\n"
            f"**Topic:** {user_input}\n"
            f"**Date:** {date_str}\n\n"
            f"## Research Findings\n{research_data[:1500]}\n\n"
            f"## Strategic Recommendations\n{strategy_data[:1500]}\n\n"
            f"## Action Plan\n{plan_data[:1500]}\n\n"
            f"## Goals\n{goals_data[:1000]}\n\n"
            f"---\n"
            f"*Report generated by LifeOS Agent*"
        )


def generate_pdf_report(
    report_content: str,
    topic: str,
    user_name: str = "User",
) -> str:
    """
    Generate a PDF from the report content.

    Args:
        report_content: The full report text content.
        topic: The report topic.
        user_name: The user's display name.

    Returns:
        Path to the generated PDF file.
    """
    try:
        from tools.pdf_tools import create_pdf

        pdf_path = create_pdf(
            content=report_content,
            topic=topic,
            user_name=user_name,
        )

        logger.info(f"PDF report generated: {pdf_path}")
        return pdf_path

    except Exception as e:
        logger.error(f"PDF generation failed: {e}")
        raise RuntimeError(f"Failed to generate PDF: {str(e)}")
