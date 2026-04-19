"""
LifeOS Agent - Master Thinker Agent
======================================
Agent 11: World's Best Deep Thinker and Problem Solver
Role: First Principles, Multi-perspective, Devil's Advocate, Mental Models.
Model: Gemini 2.5 Flash (maximum reasoning depth)
"""

import logging
from typing import Optional

from crewai import Agent, Task, Crew

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════
# LLM INITIALIZATION
# ═══════════════════════════════════════════


def _get_llm():
    """Get the best available LLM for thinker agent (primary: gemini for reasoning)."""
    from config.llm_factory import get_llm_with_fallback
    return get_llm_with_fallback(primary="gemini")


# ═══════════════════════════════════════════
# THINKING FRAMEWORK DETECTION
# ═══════════════════════════════════════════

MENTAL_MODELS = {
    "first_principles": {
        "name": "First Principles Thinking",
        "description": "Break down the problem to its fundamental truths and build up from there.",
        "triggers": ["why", "fundamental", "basic", "core", "root cause", "from scratch"],
    },
    "inversion": {
        "name": "Inversion (Charlie Munger)",
        "description": "Instead of thinking about how to succeed, think about what would cause failure and avoid that.",
        "triggers": ["avoid", "prevent", "mistake", "wrong", "fail", "risk"],
    },
    "pareto": {
        "name": "Pareto Principle (80/20)",
        "description": "Focus on the 20% of inputs that create 80% of the results.",
        "triggers": ["priority", "focus", "important", "impact", "efficient", "leverage"],
    },
    "occam": {
        "name": "Occam's Razor",
        "description": "The simplest explanation is usually the best one.",
        "triggers": ["simple", "simplify", "complicated", "complex", "confusing"],
    },
    "second_order": {
        "name": "Second-Order Thinking",
        "description": "Think beyond the immediate consequence — what happens AFTER the first effect?",
        "triggers": ["consequence", "long term", "future", "impact", "chain", "domino"],
    },
    "analogy": {
        "name": "Reasoning by Analogy",
        "description": "Apply patterns from a well-understood domain to the current problem.",
        "triggers": ["like", "similar", "compare", "analogy", "metaphor", "example"],
    },
}


def _detect_frameworks(user_input: str) -> list:
    """Detect which mental models are most relevant to the user's question."""
    input_lower = user_input.lower()
    relevant = []

    for key, model in MENTAL_MODELS.items():
        if any(trigger in input_lower for trigger in model["triggers"]):
            relevant.append(model)

    # Always include first principles and a complementary model
    if not relevant:
        relevant = [MENTAL_MODELS["first_principles"], MENTAL_MODELS["second_order"]]

    return relevant[:3]


# ═══════════════════════════════════════════
# MASTER THINKER AGENT
# ═══════════════════════════════════════════


def run_thinker_agent(user_id: str, user_input: str, user_context: str = "") -> str:
    """
    Run the Master Thinker Agent for deep analysis.

    Args:
        user_id: Unique user identifier.
        user_input: The user's question or topic.
        user_context: Context from memory.

    Returns:
        Deep analysis response string.
    """
    try:
        # Detect relevant mental models
        frameworks = _detect_frameworks(user_input)
        frameworks_text = "\n".join([
            f"- **{f['name']}**: {f['description']}"
            for f in frameworks
        ])

        # Get past thinking context
        past_context = ""
        try:
            from memory.chroma_store import get_chroma_store
            chroma = get_chroma_store()
            results = chroma.query(user_id, f"deep thinking analysis {user_input}", n_results=2)
            if results:
                past_context = "\n".join([r["document"][:200] for r in results])
        except Exception:
            pass

        llm = _get_llm()

        thinker_agent = Agent(
            role="World-Class Deep Thinker & Problem Solver",
            goal="Provide profound, multi-layered analysis that goes 3 levels deeper than surface-level answers",
            backstory=(
                "You are a polymath thinker combining the analytical rigor of Charlie Munger, "
                "the creative genius of Richard Feynman, and the strategic depth of Sun Tzu. "
                "You NEVER give surface-level answers. For every question, you dig deeper — "
                "asking 'why?' three times until you reach fundamental insights. You apply "
                "mental models like First Principles, Inversion, Pareto, and Second-Order "
                "Thinking naturally. You see connections that others miss and can explain "
                "complex ideas with simple analogies. You're intellectually curious and get "
                "excited about deep questions. You use 🧠💡🔍 emojis naturally. "
                "Your response format: ONE powerful insight first (1 sentence), then deep dive, "
                "then key takeaways."
            ),
            llm=llm,
            verbose=False,
            max_iter=3,
            allow_delegation=False,
        )

        task_description = (
            f"Provide a deep, multi-layered analysis of the following question/topic.\n\n"
            f"**Question:** {user_input}\n\n"
            f"**Relevant Mental Models to Apply:**\n{frameworks_text}\n\n"
        )

        if user_context:
            task_description += f"**User Context:** {user_context[:400]}\n\n"

        if past_context:
            task_description += f"**Related Past Thinking:**\n{past_context}\n\n"

        task_description += (
            f"**Response Structure (MUST follow):**\n\n"
            f"1. **💡 Core Insight** — Start with ONE powerful sentence that captures "
            f"the deepest insight about this topic\n\n"
            f"2. **🔍 Deep Dive** — Analyze using the relevant mental models above. "
            f"Give 3-5 different perspectives. Include:\n"
            f"   - First Principles breakdown\n"
            f"   - Devil's Advocate (argue the opposite)\n"
            f"   - Historical parallels or analogies\n"
            f"   - Future implications\n\n"
            f"3. **🧠 Key Takeaways** — 3-5 bullet points of the most important insights\n\n"
            f"4. **⚡ Action Item** — One concrete thing the user can do RIGHT NOW\n\n"
            f"**Rules:**\n"
            f"- NEVER give surface-level answers — always go 3 levels deep\n"
            f"- Use simple analogies to explain complex ideas\n"
            f"- Be intellectually curious and excited about the question\n"
            f"- Keep it structured with clear headers\n"
            f"- Use 🧠💡🔍 emojis naturally"
        )

        thinker_task = Task(
            description=task_description,
            expected_output=(
                "A profound, multi-layered analysis with: Core Insight (1 sentence), "
                "Deep Dive (3-5 perspectives with mental models), Key Takeaways (bullets), "
                "and Action Item. Goes 3 levels deeper than surface answers."
            ),
            agent=thinker_agent,
        )

        crew = Crew(
            agents=[thinker_agent],
            tasks=[thinker_task],
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
                f"Deep thinking on: {user_input[:200]}. Insight: {output[:300]}",
                metadata={"type": "deep_thinking"},
            )
        except Exception:
            pass

        logger.info(f"Thinker agent completed for user {user_id}")
        return output

    except Exception as e:
        logger.error(f"Thinker agent failed: {e}")
        return (
            f"🧠 The deep thinker hit an issue: {str(e)[:200]}\n\n"
            f"Let me try a simpler approach — what specifically about "
            f"'{user_input[:50]}' would you like me to dig into?"
        )
