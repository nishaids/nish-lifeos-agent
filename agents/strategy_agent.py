"""
LifeOS Agent - Strategy Agent
================================
Agent 3: Strategic Life Advisor
Role: Create high-level life strategies.
Model: Gemini 2.5 Flash (smart reasoning)
Tools: Research results, user profile
"""

import logging
from typing import Optional

from crewai import Agent, Task, Crew

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════
# LLM INITIALIZATION
# ═══════════════════════════════════════════


def _get_llm():
    """Get the best available LLM for strategy agent (primary: gemini)."""
    from config.llm_factory import get_llm_with_fallback
    return get_llm_with_fallback(primary="gemini")


# ═══════════════════════════════════════════
# STRATEGY AGENT
# ═══════════════════════════════════════════


def run_strategy_agent(
    user_input: str,
    research_data: str,
    goals_data: str,
    user_context: str = "",
) -> str:
    """
    Run the Strategy Agent to create personalized life strategies.

    Args:
        user_input: The user's original query/topic.
        research_data: Research findings from the Research Agent.
        goals_data: Goal analysis from the Goal Agent.
        user_context: User context from the Memory Agent.

    Returns:
        Strategic advice and recommendations as a string.
    """
    try:
        llm = _get_llm()

        strategy_agent = Agent(
            role="Strategic Life Advisor",
            goal=(
                "Create high-level, personalized life strategies that align with "
                "the user's goals, situation, and aspirations"
            ),
            backstory=(
                "You are a world-class strategic life advisor, combining the wisdom of "
                "executive coaches, career strategists, and personal development experts. "
                "You analyze research data, understand the user's unique situation and goals, "
                "and create comprehensive strategies that are both ambitious and achievable. "
                "You think in frameworks — considering short-term wins, medium-term milestones, "
                "and long-term vision. You balance pragmatism with inspiration, always providing "
                "actionable strategic direction. You consider multiple dimensions of life: "
                "career, learning, health, finances, relationships, and personal growth."
            ),
            llm=llm,
            verbose=False,
            max_iter=3,
            allow_delegation=False,
        )

        task_description = (
            f"Create a comprehensive, personalized life strategy based on the following inputs.\n\n"
            f"**User's Query/Topic:** {user_input}\n\n"
            f"**Research Findings:**\n{research_data[:2000]}\n\n"
            f"**User's Goals & Progress:**\n{goals_data[:1500]}\n\n"
            f"**User Context & Profile:**\n{user_context[:1000]}\n\n"
            f"**Strategy Requirements:**\n"
            f"1. **Vision Statement:** Create a compelling vision aligned with user's goals\n"
            f"2. **Strategic Pillars:** Identify 3-5 key strategic focus areas\n"
            f"3. **Competitive Advantages:** What unique strengths can the user leverage?\n"
            f"4. **Key Opportunities:** Based on research, what opportunities exist?\n"
            f"5. **Risk Mitigation:** What are the main risks and how to mitigate them?\n"
            f"6. **Resource Allocation:** How should the user allocate time and energy?\n"
            f"7. **Success Metrics:** How will the user know they're on track?\n"
            f"8. **Quick Wins:** 2-3 things the user can do immediately\n"
            f"9. **30-Day Strategy:** Key focus for the next month\n"
            f"10. **90-Day Strategy:** Medium-term strategic milestones\n\n"
            f"Make the strategy deeply personalized to this specific user's situation, "
            f"goals, and the research findings. Avoid generic advice."
        )

        strategy_task = Task(
            description=task_description,
            expected_output=(
                "A comprehensive, personalized life strategy document with vision statement, "
                "strategic pillars, opportunities, risk mitigation, resource allocation, "
                "success metrics, quick wins, 30-day plan, and 90-day milestones. "
                "The strategy should be specific, actionable, and deeply personalized."
            ),
            agent=strategy_agent,
        )

        crew = Crew(
            agents=[strategy_agent],
            tasks=[strategy_task],
            verbose=False,
        )

        result = crew.kickoff()
        output = str(result)

        logger.info(f"Strategy agent completed for query: {user_input[:50]}")
        return output

    except Exception as e:
        logger.error(f"Strategy agent failed: {e}")
        return (
            f"## Strategic Advice\n\n"
            f"Based on your query about '{user_input}', here are key strategic points:\n\n"
            f"1. **Focus on fundamentals** — Build a strong foundation before scaling\n"
            f"2. **Leverage research** — Use the latest data to inform decisions\n"
            f"3. **Set clear milestones** — Break big goals into measurable steps\n"
            f"4. **Review weekly** — Regular reviews accelerate progress\n\n"
            f"*Note: Full strategy generation encountered an issue. "
            f"The above provides general strategic guidance. Please retry for "
            f"a more personalized strategy.*"
        )
