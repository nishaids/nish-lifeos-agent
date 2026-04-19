"""
LifeOS Agent - Goal Tracker Agent
====================================
Agent 1: Personal Goal Tracker
Role: Track user goals and measure progress.
Model: Gemini 2.5 Flash
Tools: ChromaDB memory, user profile
"""

import logging
from typing import Optional

from crewai import Agent, Task, Crew

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════
# LLM INITIALIZATION
# ═══════════════════════════════════════════


def _get_llm():
    """Get the best available LLM for goal agent (primary: gemini)."""
    from config.llm_factory import get_llm_with_fallback
    return get_llm_with_fallback(primary="gemini")


# ═══════════════════════════════════════════
# GOAL TRACKING FUNCTIONS
# ═══════════════════════════════════════════


def _extract_goals_from_input(user_input: str, user_id: str) -> None:
    """
    Detect and save any new goals mentioned in user input.

    Args:
        user_input: The user's message text.
        user_id: Unique user identifier.
    """
    try:
        from memory.user_profile import get_profile_manager

        profile_mgr = get_profile_manager()

        # Simple keyword-based goal detection
        goal_keywords = [
            "i want to", "my goal is", "i aim to", "i plan to",
            "i need to", "i want", "goal:", "objective:",
            "i'm working on", "i am working on", "i'd like to",
        ]

        lower_input = user_input.lower()
        for keyword in goal_keywords:
            if keyword in lower_input:
                # Extract the goal text after the keyword
                idx = lower_input.index(keyword)
                goal_text = user_input[idx + len(keyword):].strip()
                # Clean up the goal text (take first sentence)
                for delim in [".", "!", "?", "\n"]:
                    if delim in goal_text:
                        goal_text = goal_text[:goal_text.index(delim)].strip()
                        break

                if len(goal_text) > 5:  # Minimum meaningful goal length
                    # Determine category based on keywords
                    category = _categorize_goal(goal_text)
                    profile_mgr.add_goal(user_id, goal_text, category)
                    logger.info(f"Extracted and saved goal for user {user_id}: {goal_text[:50]}")
                break  # Only extract the first goal per message
    except Exception as e:
        logger.warning(f"Goal extraction failed: {e}")


def _categorize_goal(goal_text: str) -> str:
    """Categorize a goal based on keywords."""
    lower = goal_text.lower()
    categories = {
        "career": ["job", "career", "work", "promotion", "salary", "professional", "business", "startup"],
        "learning": ["learn", "study", "course", "skill", "certification", "degree", "education", "read"],
        "health": ["health", "fitness", "weight", "exercise", "diet", "meditation", "sleep", "gym"],
        "finance": ["money", "save", "invest", "financial", "budget", "income", "debt", "wealth"],
        "personal": ["relationship", "family", "hobby", "travel", "happiness", "social", "friend"],
    }

    for category, keywords in categories.items():
        if any(kw in lower for kw in keywords):
            return category

    return "general"


# ═══════════════════════════════════════════
# GOAL AGENT
# ═══════════════════════════════════════════


def run_goal_agent(user_id: str, user_input: str, user_context: str = "") -> str:
    """
    Run the Goal Tracker Agent to analyze and track user goals.

    Args:
        user_id: Unique user identifier.
        user_input: The user's current message.
        user_context: Context from the memory agent.

    Returns:
        Goal analysis and tracking report as a string.
    """
    try:
        # Extract any new goals from input
        _extract_goals_from_input(user_input, user_id)

        # Load current goals
        from memory.user_profile import get_profile_manager

        profile_mgr = get_profile_manager()
        goals_summary = profile_mgr.get_goals_summary(user_id)
        profile = profile_mgr.load(user_id)

        # Get goal-related context from ChromaDB
        from memory.chroma_store import get_chroma_store

        chroma = get_chroma_store()
        past_goal_context = chroma.get_context_for_query(
            user_id, f"goals progress {user_input}", max_context=3
        )

        llm = _get_llm()

        goal_agent = Agent(
            role="Personal Goal Tracker",
            goal="Track user goals, measure progress, identify gaps, and provide motivational guidance",
            backstory=(
                "You are an expert personal goal tracker and accountability coach. "
                "You help users set SMART goals (Specific, Measurable, Achievable, Relevant, "
                "Time-bound), track their progress, identify gaps and obstacles, and provide "
                "motivational support. You analyze patterns in goal completion and suggest "
                "adjustments to keep users on track. You celebrate wins and gently address "
                "areas needing improvement."
            ),
            llm=llm,
            verbose=False,
            max_iter=3,
            allow_delegation=False,
        )

        goals_data = ""
        for idx, goal in enumerate(profile.get("goals", [])):
            goals_data += (
                f"\n  Goal {idx + 1}: {goal.get('goal', 'N/A')}\n"
                f"  Category: {goal.get('category', 'general')}\n"
                f"  Status: {goal.get('status', 'active')}\n"
                f"  Progress: {goal.get('progress', 0)}%\n"
                f"  Milestones: {len(goal.get('milestones', []))}\n"
            )

        if not goals_data:
            goals_data = "No goals set yet."

        task_description = (
            f"Analyze the user's goals and provide a comprehensive goal tracking report.\n\n"
            f"**User's Current Message:** {user_input}\n\n"
            f"**Current Goals:**\n{goals_data}\n\n"
            f"**User Context:** {user_context}\n\n"
            f"**Past Goal Context:**\n{past_goal_context}\n\n"
            f"**Instructions:**\n"
            f"1. Review all current goals and their progress\n"
            f"2. Identify any new goals mentioned in the user's message\n"
            f"3. Assess progress gaps and potential obstacles\n"
            f"4. Suggest specific next actions for each active goal\n"
            f"5. Provide motivational insights based on progress patterns\n"
            f"6. If no goals exist, suggest relevant goals based on the user's query\n"
            f"7. Recommend priority ordering for multiple goals"
        )

        goal_task = Task(
            description=task_description,
            expected_output=(
                "A structured goal tracking report with: current goals status, progress "
                "analysis, identified gaps, specific next actions, priority recommendations, "
                "and motivational insights."
            ),
            agent=goal_agent,
        )

        crew = Crew(
            agents=[goal_agent],
            tasks=[goal_task],
            verbose=False,
        )

        result = crew.kickoff()
        output = str(result)

        logger.info(f"Goal agent completed for user {user_id}")
        return output

    except Exception as e:
        logger.error(f"Goal agent failed: {e}")
        # Fallback: return raw goals summary
        try:
            from memory.user_profile import get_profile_manager
            return get_profile_manager().get_goals_summary(user_id)
        except Exception:
            return "Goal tracking temporarily unavailable. Please try again."
