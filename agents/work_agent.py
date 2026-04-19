"""
LifeOS Agent - Work Agent
============================
Agent 13: Professional Work and Productivity Assistant
Role: Draft emails, LinkedIn posts, presentations, meeting notes, resumes.
Model: Groq (fast, professional output)
"""

import logging
from typing import Optional

from crewai import Agent, Task, Crew

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════
# LLM INITIALIZATION
# ═══════════════════════════════════════════


def _get_llm():
    """Get the best available LLM for work agent (primary: groq for speed)."""
    from config.llm_factory import get_llm_with_fallback
    return get_llm_with_fallback(primary="groq")


# ═══════════════════════════════════════════
# WORK TASK DETECTION
# ═══════════════════════════════════════════


def _detect_work_task(user_input: str) -> str:
    """Detect the type of work/professional task."""
    input_lower = user_input.lower()

    if any(kw in input_lower for kw in ["email", "mail", "write email", "draft email", "reply"]):
        return "email"
    if any(kw in input_lower for kw in ["linkedin", "post", "social media", "professional post"]):
        return "linkedin"
    if any(kw in input_lower for kw in ["meeting", "agenda", "minutes", "meeting notes"]):
        return "meeting"
    if any(kw in input_lower for kw in ["presentation", "slides", "ppt", "slide deck"]):
        return "presentation"
    if any(kw in input_lower for kw in ["proposal", "business proposal", "project proposal"]):
        return "proposal"
    if any(kw in input_lower for kw in ["resume", "cv", "curriculum vitae"]):
        return "resume"
    if any(kw in input_lower for kw in ["cover letter", "application letter"]):
        return "cover_letter"
    if any(kw in input_lower for kw in ["task", "todo", "prioritize", "task list", "to-do"]):
        return "task_list"
    if any(kw in input_lower for kw in ["report", "business report", "project report"]):
        return "report"

    return "general"


def _detect_tone(user_input: str) -> str:
    """Detect the professional tone needed."""
    input_lower = user_input.lower()

    if any(kw in input_lower for kw in ["formal", "corporate", "official", "professional"]):
        return "formal"
    if any(kw in input_lower for kw in ["casual", "friendly", "startup", "informal"]):
        return "casual"
    if any(kw in input_lower for kw in ["apology", "sorry", "apologize"]):
        return "apologetic"
    if any(kw in input_lower for kw in ["follow up", "follow-up", "reminder"]):
        return "follow_up"

    return "professional"


# ═══════════════════════════════════════════
# WORK AGENT
# ═══════════════════════════════════════════


def run_work_agent(user_id: str, user_input: str, user_context: str = "") -> str:
    """
    Run the Work Agent for professional tasks.

    Args:
        user_id: Unique user identifier.
        user_input: The user's work-related request.
        user_context: Context from memory.

    Returns:
        Professional content response string.
    """
    try:
        task_type = _detect_work_task(user_input)
        tone = _detect_tone(user_input)

        # Get user's work context from memory
        work_context = ""
        try:
            from memory.user_profile import get_profile_manager
            profile = get_profile_manager().load(user_id)
            prefs = profile.get("preferences", {})
            work_context = (
                f"Profession: {prefs.get('profession', 'not specified')}, "
                f"Company type: {prefs.get('company_type', 'not specified')}"
            )
        except Exception:
            pass

        llm = _get_llm()

        # Task-specific configurations
        task_configs = {
            "email": {
                "instruction": (
                    "Draft a professional email with:\n"
                    "1. Subject line\n"
                    "2. Greeting\n"
                    "3. Body (clear, concise, purpose-driven)\n"
                    "4. Call to action\n"
                    "5. Professional sign-off\n\n"
                    "Make it ready to copy-paste. Adjust tone based on context."
                ),
            },
            "linkedin": {
                "instruction": (
                    "Create a LinkedIn post with:\n"
                    "1. Hook (first line that grabs attention)\n"
                    "2. Value content (insights, story, or advice)\n"
                    "3. Engagement driver (question or call to action)\n"
                    "4. Relevant hashtags (3-5)\n\n"
                    "Use line breaks for readability. Add appropriate emojis."
                ),
            },
            "meeting": {
                "instruction": (
                    "Create meeting content with:\n"
                    "1. Meeting title and objective\n"
                    "2. Agenda items (timed)\n"
                    "3. Discussion points\n"
                    "4. Action items template\n"
                    "5. Follow-up template"
                ),
            },
            "presentation": {
                "instruction": (
                    "Create a presentation outline with:\n"
                    "1. Title slide\n"
                    "2. Agenda/Overview slide\n"
                    "3. Content slides (6-10 slides with bullet points)\n"
                    "4. Key data/stats slides\n"
                    "5. Summary/Key takeaways slide\n"
                    "6. Q&A / Next Steps slide\n\n"
                    "For each slide: Title + 3-5 bullet points + speaker notes."
                ),
            },
            "proposal": {
                "instruction": (
                    "Create a business proposal with:\n"
                    "1. Executive Summary\n"
                    "2. Problem Statement\n"
                    "3. Proposed Solution\n"
                    "4. Methodology/Approach\n"
                    "5. Timeline and Milestones\n"
                    "6. Budget/Pricing (template)\n"
                    "7. Expected Outcomes\n"
                    "8. Next Steps"
                ),
            },
            "resume": {
                "instruction": (
                    "Create a resume/CV with:\n"
                    "1. Professional Summary (3-4 lines)\n"
                    "2. Skills section (categorized: Technical, Soft, Tools)\n"
                    "3. Experience section template (with STAR format bullets)\n"
                    "4. Education section\n"
                    "5. Projects section\n"
                    "6. Certifications\n\n"
                    "Use action verbs and quantifiable achievements."
                ),
            },
            "cover_letter": {
                "instruction": (
                    "Write a compelling cover letter with:\n"
                    "1. Opening hook (why this role excites you)\n"
                    "2. Value proposition (what you bring)\n"
                    "3. Specific achievements (with metrics)\n"
                    "4. Cultural fit (why this company)\n"
                    "5. Strong closing with call to action"
                ),
            },
            "task_list": {
                "instruction": (
                    "Create a prioritized task list using the Eisenhower Matrix:\n\n"
                    "🔴 **Urgent + Important** (Do First)\n"
                    "🟡 **Important, Not Urgent** (Schedule)\n"
                    "🟠 **Urgent, Not Important** (Delegate)\n"
                    "⚪ **Neither** (Eliminate)\n\n"
                    "For each task: estimate time, suggest deadlines, add context."
                ),
            },
            "report": {
                "instruction": (
                    "Create a professional report with:\n"
                    "1. Executive Summary\n"
                    "2. Background/Context\n"
                    "3. Analysis/Findings\n"
                    "4. Recommendations\n"
                    "5. Next Steps\n"
                    "6. Appendix (if needed)\n"
                    "Use professional language and data-driven insights."
                ),
            },
            "general": {
                "instruction": (
                    "Create professional content based on the user's request.\n"
                    "Ensure it's ready to use, well-formatted, and professional."
                ),
            },
        }

        config = task_configs.get(task_type, task_configs["general"])

        work_agent = Agent(
            role="Professional Work & Productivity Assistant",
            goal=f"Create perfect, ready-to-use {task_type} content",
            backstory=(
                "You are a top-tier executive assistant with experience at Fortune 500 "
                "companies. You craft professional content that's polished, persuasive, "
                "and ready to use immediately. You adapt your tone perfectly — from startup "
                "casual to corporate formal. Every output is copy-paste ready. You use "
                "💼📋🎯 emojis sparingly in summaries only."
            ),
            llm=llm,
            verbose=False,
            max_iter=3,
            allow_delegation=False,
        )

        task_description = (
            f"Create professional {task_type} content based on this request.\n\n"
            f"**Request:** {user_input}\n"
            f"**Task Type:** {task_type}\n"
            f"**Tone:** {tone}\n\n"
        )

        if work_context:
            task_description += f"**Work Context:** {work_context}\n\n"

        if user_context:
            task_description += f"**User Context:** {user_context[:400]}\n\n"

        task_description += (
            f"**Format Requirements:**\n{config['instruction']}\n\n"
            f"**Quality Rules:**\n"
            f"1. Output must be READY TO COPY-PASTE and use immediately\n"
            f"2. Match the tone: {tone}\n"
            f"3. Be concise but complete\n"
            f"4. Use proper formatting (headers, bullets, spacing)\n"
            f"5. Include all necessary sections\n"
            f"6. No placeholder text — fill in everything possible"
        )

        work_task = Task(
            description=task_description,
            expected_output=(
                f"Professional, ready-to-use {task_type} content with proper formatting "
                f"and {tone} tone. Copy-paste ready quality."
            ),
            agent=work_agent,
        )

        crew = Crew(
            agents=[work_agent],
            tasks=[work_task],
            verbose=False,
        )

        result = crew.kickoff()
        output = str(result)

        # Save to memory
        try:
            from memory.chroma_store import get_chroma_store
            chroma = get_chroma_store()
            chroma.store(
                user_id,
                f"Work task ({task_type}): {user_input[:200]}",
                metadata={"type": "work", "task_type": task_type},
            )
        except Exception:
            pass

        logger.info(f"Work agent completed for user {user_id} ({task_type})")
        return output

    except Exception as e:
        logger.error(f"Work agent failed: {e}")
        return (
            f"💼 The work assistant hit an issue: {str(e)[:200]}\n\n"
            f"Please try again with more details about what you need."
        )
