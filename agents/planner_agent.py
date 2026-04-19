"""
LifeOS Agent - Planner Agent
================================
Agent 4: Daily Action Planner
Role: Create specific actionable daily plans.
Model: Groq Llama 3.3 (fast execution)
Tools: Goals, strategy, calendar context
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

from crewai import Agent, Task, Crew

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════
# LLM INITIALIZATION
# ═══════════════════════════════════════════


def _get_llm():
    """Get the best available LLM for planner agent (primary: groq)."""
    from config.llm_factory import get_llm_with_fallback
    return get_llm_with_fallback(primary="groq")


# ═══════════════════════════════════════════
# DATE CONTEXT HELPERS
# ═══════════════════════════════════════════


def _get_calendar_context() -> str:
    """Generate current date and calendar context."""
    now = datetime.now()
    day_name = now.strftime("%A")
    date_str = now.strftime("%B %d, %Y")
    week_num = now.isocalendar()[1]

    # Calculate days remaining in month
    if now.month == 12:
        next_month = datetime(now.year + 1, 1, 1)
    else:
        next_month = datetime(now.year, now.month + 1, 1)
    days_remaining = (next_month - now).days

    # Week context
    days_until_weekend = (5 - now.weekday()) % 7
    if days_until_weekend == 0 and now.weekday() >= 5:
        week_status = "It's the weekend"
    elif days_until_weekend <= 1:
        week_status = "End of work week"
    else:
        week_status = f"{days_until_weekend} work days until weekend"

    return (
        f"📅 **Calendar Context:**\n"
        f"Today: {day_name}, {date_str}\n"
        f"Week: #{week_num} of {now.year}\n"
        f"Days remaining in month: {days_remaining}\n"
        f"Week status: {week_status}\n"
        f"Time: {now.strftime('%I:%M %p')}"
    )


# ═══════════════════════════════════════════
# PLANNER AGENT
# ═══════════════════════════════════════════


def run_planner_agent(
    user_input: str,
    strategy_data: str,
    goals_data: str,
    user_context: str = "",
) -> str:
    """
    Run the Planner Agent to create actionable daily/weekly plans.

    Args:
        user_input: The user's original query/topic.
        strategy_data: Strategic recommendations from the Strategy Agent.
        goals_data: Goal analysis from the Goal Agent.
        user_context: User context from the Memory Agent.

    Returns:
        Detailed action plan as a string.
    """
    try:
        calendar_context = _get_calendar_context()

        llm = _get_llm()

        planner_agent = Agent(
            role="Daily Action Planner",
            goal=(
                "Create specific, actionable daily plans with clear priorities, "
                "time blocks, and measurable outcomes"
            ),
            backstory=(
                "You are an elite productivity coach and action planner. You transform "
                "high-level strategies into concrete daily actions. You understand time "
                "management, energy management, and the psychology of habit formation. "
                "You create plans that are realistic yet ambitious, always considering "
                "the user's current context, energy levels, and commitments. You prioritize "
                "ruthlessly, focusing on high-impact actions that move the needle. "
                "You use frameworks like time-blocking, the Eisenhower matrix, and the "
                "80/20 principle to optimize the user's daily output."
            ),
            llm=llm,
            verbose=False,
            max_iter=3,
            allow_delegation=False,
        )

        task_description = (
            f"Create a comprehensive, actionable plan based on the following inputs.\n\n"
            f"**User's Query/Topic:** {user_input}\n\n"
            f"**{calendar_context}**\n\n"
            f"**Strategic Direction:**\n{strategy_data[:2000]}\n\n"
            f"**Goals & Progress:**\n{goals_data[:1500]}\n\n"
            f"**User Context:**\n{user_context[:1000]}\n\n"
            f"**Create the following plan sections:**\n\n"
            f"## 🎯 Today's Top 3 Priorities\n"
            f"The three most important things to accomplish today.\n"
            f"Each priority should be specific, measurable, and achievable in one day.\n\n"
            f"## ⏰ Time-Blocked Daily Schedule\n"
            f"A realistic hour-by-hour plan for today covering:\n"
            f"- Morning power block (deep work)\n"
            f"- Afternoon execution block\n"
            f"- Evening review & learning\n"
            f"Include breaks and buffer time.\n\n"
            f"## 📋 This Week's Action Items\n"
            f"5-7 specific tasks to complete this week, each with:\n"
            f"- Clear deliverable\n"
            f"- Estimated time\n"
            f"- Priority level (P1/P2/P3)\n\n"
            f"## 🏁 30-Day Milestones\n"
            f"3-5 key milestones to hit in the next 30 days.\n"
            f"Each milestone should be measurable and linked to a goal.\n\n"
            f"## 💡 Quick Win Actions\n"
            f"2-3 things the user can do in the next 15 minutes to build momentum.\n\n"
            f"Make everything specific to the user's situation. "
            f"No generic advice — every action item should relate to their actual goals and query."
        )

        planner_task = Task(
            description=task_description,
            expected_output=(
                "A complete action plan with: Today's Top 3 Priorities, Time-Blocked "
                "Daily Schedule, This Week's Action Items (with time estimates and priority), "
                "30-Day Milestones, and Quick Win Actions. Everything should be specific, "
                "actionable, and personalized."
            ),
            agent=planner_agent,
        )

        crew = Crew(
            agents=[planner_agent],
            tasks=[planner_task],
            verbose=False,
        )

        result = crew.kickoff()
        output = str(result)

        logger.info(f"Planner agent completed for query: {user_input[:50]}")
        return output

    except Exception as e:
        logger.error(f"Planner agent failed: {e}")
        calendar = _get_calendar_context()
        return (
            f"## Action Plan\n\n"
            f"{calendar}\n\n"
            f"### Today's Focus\n"
            f"Based on your query about '{user_input}':\n\n"
            f"1. **Research Phase** — Spend 30 minutes deep diving into the topic\n"
            f"2. **Plan Phase** — Outline your next 3 concrete steps\n"
            f"3. **Execute Phase** — Take the first step immediately\n\n"
            f"*Note: Full planning encountered an issue. "
            f"Please retry for a detailed personalized plan.*"
        )
