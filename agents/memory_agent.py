"""
LifeOS Agent - Memory Agent
=============================
Agent 6: Personal Memory Manager
Role: Remember everything about the user across sessions.
Model: OpenRouter free model
Tools: Mem0 API, ChromaDB
"""

import logging
from typing import Optional

from crewai import Agent, Task, Crew

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════
# LLM INITIALIZATION HELPERS
# ═══════════════════════════════════════════


def _get_llm():
    """Get the best available LLM for memory agent (primary: openrouter)."""
    from config.llm_factory import get_llm_with_fallback
    return get_llm_with_fallback(primary="openrouter")


# ═══════════════════════════════════════════
# MEM0 INTEGRATION
# ═══════════════════════════════════════════


def _store_to_mem0(user_id: str, content: str) -> bool:
    """Store content to Mem0 long-term memory."""
    try:
        from config.models import MEM0_CONFIG

        api_key = MEM0_CONFIG.get("api_key", "")
        if not api_key:
            logger.info("Mem0 API key not set, skipping Mem0 storage")
            return False

        from mem0 import MemoryClient

        client = MemoryClient(api_key=api_key)
        client.add(content, user_id=str(user_id))
        logger.info(f"Stored to Mem0 for user {user_id}")
        return True
    except Exception as e:
        logger.warning(f"Mem0 storage failed: {e}")
        return False


def _retrieve_from_mem0(user_id: str, query: str) -> str:
    """Retrieve relevant memories from Mem0."""
    try:
        from config.models import MEM0_CONFIG

        api_key = MEM0_CONFIG.get("api_key", "")
        if not api_key:
            return ""

        from mem0 import MemoryClient

        client = MemoryClient(api_key=api_key)
        results = client.search(query, user_id=str(user_id))

        if not results:
            return ""

        memories = []
        for mem in results[:5]:
            if isinstance(mem, dict):
                memory_text = mem.get("memory", mem.get("text", str(mem)))
            else:
                memory_text = str(mem)
            memories.append(f"- {memory_text}")

        return "🧠 **Long-term Memories:**\n" + "\n".join(memories)
    except Exception as e:
        logger.warning(f"Mem0 retrieval failed: {e}")
        return ""


# ═══════════════════════════════════════════
# MEMORY AGENT FUNCTIONS
# ═══════════════════════════════════════════


def load_user_context(user_id: str, user_input: str) -> dict:
    """
    Load complete user context from all memory systems.

    Args:
        user_id: Unique user identifier.
        user_input: Current user input for context matching.

    Returns:
        Dict with user_profile, past_context, and long_term_memory.
    """
    try:
        # Load user profile
        from memory.user_profile import get_profile_manager

        profile_mgr = get_profile_manager()
        profile = profile_mgr.load(user_id)
        profile_summary = profile_mgr.get_profile_summary(user_id)
        goals_summary = profile_mgr.get_goals_summary(user_id)

        # Load from ChromaDB
        from memory.chroma_store import get_chroma_store

        chroma = get_chroma_store()
        past_context = chroma.get_context_for_query(user_id, user_input, max_context=3)

        # Load from Mem0
        mem0_context = _retrieve_from_mem0(user_id, user_input)

        context = {
            "user_profile": profile,
            "profile_summary": profile_summary,
            "goals_summary": goals_summary,
            "past_context": past_context,
            "long_term_memory": mem0_context,
        }

        logger.info(f"Loaded full context for user {user_id}")
        return context

    except Exception as e:
        logger.error(f"Failed to load user context: {e}")
        return {
            "user_profile": {},
            "profile_summary": "Profile unavailable",
            "goals_summary": "No goals found",
            "past_context": "No past context",
            "long_term_memory": "",
        }


def save_interaction(user_id: str, user_input: str, agent_response: str) -> bool:
    """
    Save the current interaction to all memory systems.

    Args:
        user_id: Unique user identifier.
        user_input: The user's input message.
        agent_response: The agent's response summary.

    Returns:
        True if saved successfully to at least one system.
    """
    success = False

    # Save to ChromaDB
    try:
        from memory.chroma_store import get_chroma_store

        chroma = get_chroma_store()
        stored = chroma.store(
            user_id,
            f"User asked: {user_input}\nAgent responded: {agent_response[:500]}",
            metadata={"type": "interaction"},
        )
        if stored:
            success = True
    except Exception as e:
        logger.warning(f"ChromaDB storage failed: {e}")

    # Save to user profile history
    try:
        from memory.user_profile import get_profile_manager

        profile_mgr = get_profile_manager()
        profile_mgr.add_history_entry(user_id, user_input, agent_response[:500])
        success = True
    except Exception as e:
        logger.warning(f"Profile history save failed: {e}")

    # Save to Mem0
    try:
        content = f"User query: {user_input}. Key takeaway: {agent_response[:300]}"
        _store_to_mem0(user_id, content)
    except Exception as e:
        logger.warning(f"Mem0 save failed: {e}")

    return success


def run_memory_agent(user_id: str, user_input: str) -> str:
    """
    Run the Memory Agent using CrewAI to analyze and summarize user context.

    Args:
        user_id: Unique user identifier.
        user_input: Current user input.

    Returns:
        Formatted context summary string.
    """
    try:
        # Load raw context first
        context = load_user_context(user_id, user_input)

        # Build context string for the agent
        context_text = (
            f"User Profile:\n{context['profile_summary']}\n\n"
            f"Goals:\n{context['goals_summary']}\n\n"
            f"Past Context:\n{context['past_context']}\n\n"
            f"Long-term Memory:\n{context.get('long_term_memory', 'None')}\n\n"
            f"Current Query: {user_input}"
        )

        llm = _get_llm()

        memory_agent = Agent(
            role="Personal Memory Manager",
            goal="Remember everything about the user across sessions and provide relevant context",
            backstory=(
                "You are a dedicated personal memory manager for a life intelligence system. "
                "You have access to the user's complete history, goals, preferences, and past "
                "interactions. Your job is to analyze all available context and provide a concise "
                "but comprehensive summary that helps other agents personalize their responses. "
                "You identify patterns in user behavior, track goal progress, and surface "
                "relevant past information."
            ),
            llm=llm,
            verbose=False,
            max_iter=3,
            allow_delegation=False,
        )

        memory_task = Task(
            description=(
                f"Analyze the following user context and create a personalized briefing "
                f"that will help other AI agents serve this user better.\n\n"
                f"Available Context:\n{context_text}\n\n"
                f"Create a concise summary covering:\n"
                f"1. Who this user is and their key preferences\n"
                f"2. Their active goals and current progress\n"
                f"3. Relevant past interactions related to the current query\n"
                f"4. Any patterns or insights from their history\n"
                f"5. Specific context that should influence the response to their current query"
            ),
            expected_output=(
                "A structured context briefing with user identity, active goals, "
                "relevant history, and personalization notes for the current query."
            ),
            agent=memory_agent,
        )

        crew = Crew(
            agents=[memory_agent],
            tasks=[memory_task],
            verbose=False,
        )

        result = crew.kickoff()
        output = str(result)

        logger.info(f"Memory agent completed for user {user_id}")
        return output

    except Exception as e:
        logger.error(f"Memory agent failed: {e}")
        # Return raw context as fallback
        try:
            context = load_user_context(user_id, user_input)
            return (
                f"{context['profile_summary']}\n\n"
                f"{context['goals_summary']}\n\n"
                f"{context['past_context']}"
            )
        except Exception:
            return "No user context available. This appears to be a new user."
