"""
LifeOS Agent - Research Agent
===============================
Agent 2: Senior Research Analyst
Role: Find latest information on any topic.
Model: Groq Llama 3.3 70B (fast)
Tools: Tavily Search, Wikipedia, ArXiv
"""

import logging
from typing import Optional

from crewai import Agent, Task, Crew

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════
# LLM INITIALIZATION
# ═══════════════════════════════════════════


def _get_llm():
    """Get the best available LLM for research agent (primary: groq)."""
    from config.llm_factory import get_llm_with_fallback
    return get_llm_with_fallback(primary="groq")


# ═══════════════════════════════════════════
# RESEARCH FUNCTIONS
# ═══════════════════════════════════════════


def _perform_research(query: str) -> str:
    """
    Perform comprehensive research using all available search tools.

    Args:
        query: The research query.

    Returns:
        Combined research results string.
    """
    from tools.search_tools import smart_search

    try:
        results = smart_search(query)
        return results
    except Exception as e:
        logger.error(f"Smart search failed: {e}")
        return f"Research search failed: {str(e)}"


# ═══════════════════════════════════════════
# RESEARCH AGENT
# ═══════════════════════════════════════════


def run_research_agent(query: str, user_context: str = "") -> str:
    """
    Run the Research Agent to perform deep research on a topic.

    Args:
        query: The user's research query/topic.
        user_context: Optional context about the user for personalization.

    Returns:
        Comprehensive research findings as a string.
    """
    try:
        # Gather raw research data first
        research_data = _perform_research(query)

        llm = _get_llm()

        research_agent = Agent(
            role="Senior Research Analyst",
            goal="Find the latest, most accurate information on any topic and present it clearly with citations",
            backstory=(
                "You are a world-class research analyst with expertise in finding, "
                "synthesizing, and presenting information from multiple sources. You have "
                "access to web search, Wikipedia, and academic papers. You always cite your "
                "sources and distinguish between facts and opinions. You prioritize recency "
                "and relevance, and you can distill complex topics into actionable insights. "
                "Your research reports are thorough yet concise."
            ),
            llm=llm,
            verbose=False,
            max_iter=3,
            allow_delegation=False,
        )

        task_description = (
            f"Conduct deep research on the following topic and create a comprehensive research briefing.\n\n"
            f"**Topic/Query:** {query}\n\n"
        )

        if user_context:
            task_description += f"**User Context:** {user_context}\n\n"

        task_description += (
            f"**Raw Research Data:**\n{research_data}\n\n"
            f"**Instructions:**\n"
            f"1. Analyze all the research data provided above\n"
            f"2. Identify the most important findings and trends\n"
            f"3. Organize information into clear sections\n"
            f"4. Include specific facts, statistics, and citations where available\n"
            f"5. Highlight actionable insights relevant to the user's query\n"
            f"6. Note any conflicting information or areas of uncertainty\n"
            f"7. Provide a brief executive summary at the top\n\n"
            f"Format your response as a structured research report with clear sections."
        )

        research_task = Task(
            description=task_description,
            expected_output=(
                "A structured research report with executive summary, key findings organized "
                "by section, citations/sources, and actionable insights. The report should be "
                "comprehensive yet concise."
            ),
            agent=research_agent,
        )

        crew = Crew(
            agents=[research_agent],
            tasks=[research_task],
            verbose=False,
        )

        result = crew.kickoff()
        output = str(result)

        logger.info(f"Research agent completed for query: {query[:50]}")
        return output

    except Exception as e:
        logger.error(f"Research agent failed: {e}")
        # Return raw research data as fallback
        try:
            return _perform_research(query)
        except Exception as e2:
            return f"Research failed: {str(e2)}. Please try again."
