"""
LifeOS Agent - Assignment Agent
==================================
Agent 12: Academic Excellence Assistant
Role: Write assignments, study notes, mind maps, question banks, exam prep.
Model: Gemini 2.5 Flash (long-form content quality)
"""

import logging
from typing import Optional

from crewai import Agent, Task, Crew

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════
# LLM INITIALIZATION
# ═══════════════════════════════════════════


def _get_llm():
    """Get the best available LLM for assignment agent (primary: gemini)."""
    from config.llm_factory import get_llm_with_fallback
    return get_llm_with_fallback(primary="gemini")


# ═══════════════════════════════════════════
# TASK TYPE DETECTION
# ═══════════════════════════════════════════


def _detect_academic_task(user_input: str) -> str:
    """Detect the type of academic task requested."""
    input_lower = user_input.lower()

    if any(kw in input_lower for kw in ["assignment", "write about", "essay on", "write an essay"]):
        return "assignment"
    if any(kw in input_lower for kw in ["notes", "make notes", "study notes", "summarize chapter"]):
        return "notes"
    if any(kw in input_lower for kw in ["mind map", "mindmap", "concept map"]):
        return "mindmap"
    if any(kw in input_lower for kw in ["question", "question bank", "practice questions", "quiz"]):
        return "questions"
    if any(kw in input_lower for kw in ["exam", "exam prep", "prepare for exam", "revision"]):
        return "exam_prep"
    if any(kw in input_lower for kw in ["explain", "explain the topic", "what is", "define"]):
        return "explain"
    if any(kw in input_lower for kw in ["citation", "reference", "bibliography", "apa", "mla"]):
        return "citation"
    if any(kw in input_lower for kw in ["summarize", "summary", "tldr", "brief"]):
        return "summary"
    if any(kw in input_lower for kw in ["research paper", "thesis", "dissertation"]):
        return "research_paper"

    return "assignment"


# ═══════════════════════════════════════════
# ASSIGNMENT AGENT
# ═══════════════════════════════════════════


def run_assignment_agent(user_id: str, user_input: str, user_context: str = "") -> str:
    """
    Run the Assignment Agent for academic tasks.

    Args:
        user_id: Unique user identifier.
        user_input: The user's academic request.
        user_context: Context from memory.

    Returns:
        Academic content response string.
    """
    try:
        task_type = _detect_academic_task(user_input)

        # Get user's academic context
        academic_context = ""
        try:
            from memory.user_profile import get_profile_manager
            profile = get_profile_manager().load(user_id)
            prefs = profile.get("preferences", {})
            academic_context = (
                f"Academic level: {prefs.get('academic_level', 'college')}, "
                f"Subjects: {prefs.get('subjects', 'general')}"
            )
        except Exception:
            pass

        llm = _get_llm()

        # Task-specific configurations
        task_configs = {
            "assignment": {
                "goal": "Write a comprehensive, well-structured academic assignment",
                "format_instruction": (
                    "Structure the assignment with:\n"
                    "1. Title\n"
                    "2. Introduction (hook + thesis statement + roadmap)\n"
                    "3. Body sections with clear headings (3-5 sections)\n"
                    "4. Conclusion (summary + implications + future directions)\n"
                    "5. References (at least 5 credible sources)\n"
                    "Use academic language, proper transitions, and evidence-based arguments."
                ),
            },
            "notes": {
                "goal": "Create clear, well-organized study notes",
                "format_instruction": (
                    "Format the notes with:\n"
                    "1. Topic title and overview\n"
                    "2. Key concepts (numbered with definitions)\n"
                    "3. Important formulas/rules (if applicable)\n"
                    "4. Examples for each concept\n"
                    "5. Quick revision points at the end\n"
                    "Use bullet points, bold key terms, and keep it scannable."
                ),
            },
            "mindmap": {
                "goal": "Create a text-based mind map for the topic",
                "format_instruction": (
                    "Create a text-based mind map with:\n"
                    "1. Central topic at the top\n"
                    "2. Main branches (using ├── and └── symbols)\n"
                    "3. Sub-branches for each main branch\n"
                    "4. Key details at the leaf level\n"
                    "Make it visual and easy to understand at a glance."
                ),
            },
            "questions": {
                "goal": "Generate a comprehensive question bank for exam preparation",
                "format_instruction": (
                    "Create a question bank with:\n"
                    "1. 5 Short Answer questions (2-3 marks)\n"
                    "2. 5 Long Answer questions (5-10 marks)\n"
                    "3. 5 Multiple Choice questions (with 4 options and correct answer)\n"
                    "4. 3 Case Study / Application questions\n"
                    "Include answers/answer keys at the end."
                ),
            },
            "exam_prep": {
                "goal": "Create an exam preparation guide",
                "format_instruction": (
                    "Create an exam prep guide with:\n"
                    "1. Topics checklist (what to study)\n"
                    "2. Key concepts to memorize\n"
                    "3. Important formulas/definitions\n"
                    "4. Common exam patterns and question types\n"
                    "5. Time management tips for the exam\n"
                    "6. Practice questions for each topic\n"
                    "7. Last-minute revision summary"
                ),
            },
            "explain": {
                "goal": "Explain a complex academic concept in simple terms",
                "format_instruction": (
                    "Explain the concept with:\n"
                    "1. Simple definition (ELI5 — Explain Like I'm 5)\n"
                    "2. Technical definition\n"
                    "3. Real-world analogy\n"
                    "4. Step-by-step breakdown\n"
                    "5. Examples (at least 2)\n"
                    "6. Common misconceptions\n"
                    "7. Why it matters (practical applications)"
                ),
            },
            "summary": {
                "goal": "Summarize the topic concisely",
                "format_instruction": (
                    "Provide a summary with:\n"
                    "1. One-paragraph overview\n"
                    "2. Key points (5-7 bullets)\n"
                    "3. Important details\n"
                    "4. Conclusions\n"
                    "Keep it concise but comprehensive."
                ),
            },
            "research_paper": {
                "goal": "Help structure and write a research paper",
                "format_instruction": (
                    "Structure with:\n"
                    "1. Abstract (150-250 words)\n"
                    "2. Introduction (background + research question + significance)\n"
                    "3. Literature Review\n"
                    "4. Methodology\n"
                    "5. Analysis/Discussion\n"
                    "6. Conclusion\n"
                    "7. References (APA format)\n"
                    "Use formal academic language throughout."
                ),
            },
            "citation": {
                "goal": "Generate proper citations and references",
                "format_instruction": (
                    "Generate citations in both APA and MLA format.\n"
                    "Include in-text citations and full reference list entries.\n"
                    "Provide examples of how to use them in text."
                ),
            },
        }

        config = task_configs.get(task_type, task_configs["assignment"])

        assignment_agent = Agent(
            role="Academic Excellence Assistant",
            goal=config["goal"],
            backstory=(
                "You are a brilliant academic assistant with expertise across all subjects. "
                "You write like a top university professor — clear, structured, and authoritative. "
                "You always use proper academic formatting, include relevant examples, and cite "
                "sources. You adapt your writing level to the student's needs — from undergraduate "
                "to graduate level. You use 📚✍️🎓 emojis sparingly. Your work is always "
                "plagiarism-free, original, and thoughtfully constructed."
            ),
            llm=llm,
            verbose=False,
            max_iter=3,
            allow_delegation=False,
        )

        task_description = (
            f"Create academic content for the following request.\n\n"
            f"**Request:** {user_input}\n"
            f"**Task Type:** {task_type}\n\n"
        )

        if academic_context:
            task_description += f"**Student Context:** {academic_context}\n\n"

        if user_context:
            task_description += f"**User Context:** {user_context[:400]}\n\n"

        task_description += (
            f"**Format Requirements:**\n{config['format_instruction']}\n\n"
            f"**Quality Rules:**\n"
            f"1. Use proper academic language and structure\n"
            f"2. Include specific examples and evidence\n"
            f"3. Maintain logical flow between sections\n"
            f"4. Use emojis sparingly: 📚 ✍️ 🎓\n"
            f"5. Make content original and well-researched\n"
            f"6. Adapt to the apparent academic level\n"
            f"7. Include actionable study tips where relevant"
        )

        assignment_task = Task(
            description=task_description,
            expected_output=(
                f"Well-structured academic {task_type} content with proper formatting, "
                f"examples, and references where applicable. Ready to submit quality."
            ),
            agent=assignment_agent,
        )

        crew = Crew(
            agents=[assignment_agent],
            tasks=[assignment_task],
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
                f"Academic work ({task_type}): {user_input[:200]}",
                metadata={"type": "academic", "task_type": task_type},
            )
        except Exception:
            pass

        logger.info(f"Assignment agent completed for user {user_id} ({task_type})")
        return output

    except Exception as e:
        logger.error(f"Assignment agent failed: {e}")
        return (
            f"📚 The assignment assistant hit an issue: {str(e)[:200]}\n\n"
            f"Please try rephrasing your request. Make sure to specify:\n"
            f"• Subject/Topic\n"
            f"• Type (essay, notes, questions, etc.)\n"
            f"• Any specific requirements"
        )
