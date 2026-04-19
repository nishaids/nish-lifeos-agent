"""
LifeOS Agent - Emotional Support Agent
=========================================
Agent 7: Best Friend + Emotional Support Companion
Role: Respond warmly like a caring best friend. Detect mood,
match energy, offer validation, encouragement, and humor.
Model: Gemini 2.5 Flash (for empathy + depth)
"""

import logging
from typing import Optional

from crewai import Agent, Task, Crew

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════
# LLM INITIALIZATION
# ═══════════════════════════════════════════


def _get_llm():
    """Get the best available LLM for emotional agent (primary: gemini)."""
    from config.llm_factory import get_llm_with_fallback
    return get_llm_with_fallback(primary="gemini")


# ═══════════════════════════════════════════
# MOOD DETECTION
# ═══════════════════════════════════════════

MOOD_MAP = {
    "happy": {
        "keywords": ["happy", "excited", "great", "awesome", "amazing", "wonderful", "yay", "love it", "crushing it", "fantastic", "brilliant"],
        "emoji": "😄🎉✨",
        "energy": "high",
    },
    "sad": {
        "keywords": ["sad", "depressed", "down", "crying", "miss", "lost", "heartbroken", "hurt", "pain", "grief"],
        "emoji": "🤗💙🌟",
        "energy": "gentle",
    },
    "stressed": {
        "keywords": ["stressed", "overwhelmed", "pressure", "deadline", "too much", "can't handle", "burnout", "exhausted"],
        "emoji": "🤗💪🌿",
        "energy": "calming",
    },
    "anxious": {
        "keywords": ["anxious", "worried", "nervous", "scared", "afraid", "panic", "fear", "what if"],
        "emoji": "🤗💙🕊️",
        "energy": "reassuring",
    },
    "angry": {
        "keywords": ["angry", "furious", "hate", "annoyed", "frustrated", "pissed", "mad", "irritated"],
        "emoji": "😤🤝💪",
        "energy": "validating",
    },
    "lonely": {
        "keywords": ["lonely", "alone", "no one", "nobody", "isolated", "miss someone", "miss people"],
        "emoji": "🤗💙🤝",
        "energy": "warm",
    },
    "tired": {
        "keywords": ["tired", "sleepy", "exhausted", "drained", "fatigue", "no energy", "bored"],
        "emoji": "😴💤🌙",
        "energy": "gentle",
    },
    "excited": {
        "keywords": ["excited", "can't wait", "pumped", "hyped", "let's go", "so ready", "stoked"],
        "emoji": "🔥🚀🎉",
        "energy": "high",
    },
    "neutral": {
        "keywords": [],
        "emoji": "😊✨🤝",
        "energy": "friendly",
    },
}


def detect_mood(user_input: str) -> dict:
    """
    Detect the user's mood from their message.

    Args:
        user_input: The user's message text.

    Returns:
        Dict with mood name, emoji, and energy level.
    """
    input_lower = user_input.lower()

    for mood, data in MOOD_MAP.items():
        if mood == "neutral":
            continue
        if any(kw in input_lower for kw in data["keywords"]):
            return {"mood": mood, "emoji": data["emoji"], "energy": data["energy"]}

    return {"mood": "neutral", "emoji": "😊✨🤝", "energy": "friendly"}


def _save_emotional_state(user_id: str, mood: str, message: str) -> None:
    """Save the user's emotional state to memory for tracking."""
    try:
        from memory.chroma_store import get_chroma_store

        chroma = get_chroma_store()
        chroma.store(
            user_id,
            f"Emotional state: {mood}. User said: {message[:200]}",
            metadata={"type": "emotional_state", "mood": mood},
        )
    except Exception as e:
        logger.warning(f"Failed to save emotional state: {e}")


def _get_emotional_history(user_id: str) -> str:
    """Get past emotional states for context."""
    try:
        from memory.chroma_store import get_chroma_store

        chroma = get_chroma_store()
        results = chroma.query(user_id, "emotional state mood feeling", n_results=3)
        if not results:
            return "No emotional history available."

        history_parts = []
        for entry in results:
            doc = entry["document"][:150]
            history_parts.append(f"- {doc}")

        return "\n".join(history_parts)
    except Exception:
        return "No emotional history available."


# ═══════════════════════════════════════════
# EMOTIONAL SUPPORT AGENT
# ═══════════════════════════════════════════


def run_emotional_agent(user_id: str, user_input: str, user_context: str = "") -> str:
    """
    Run the Emotional Support Agent to respond as a caring best friend.

    Args:
        user_id: Unique user identifier.
        user_input: The user's current message.
        user_context: Context from memory about the user.

    Returns:
        A warm, personalized response string.
    """
    try:
        # Detect mood
        mood_info = detect_mood(user_input)
        mood = mood_info["mood"]
        energy = mood_info["energy"]
        emojis = mood_info["emoji"]

        # Save emotional state
        _save_emotional_state(user_id, mood, user_input)

        # Get user name
        user_name = "bro"
        try:
            from memory.user_profile import get_profile_manager
            profile_mgr = get_profile_manager()
            profile = profile_mgr.load(user_id)
            user_name = profile.get("name", "bro") or "bro"
        except Exception:
            pass

        # Get emotional history
        emotional_history = _get_emotional_history(user_id)

        llm = _get_llm()

        emotional_agent = Agent(
            role="Best Friend & Emotional Support Companion",
            goal="Be the user's most caring, understanding, and genuine best friend",
            backstory=(
                f"You are {user_name}'s closest best friend — not a therapist, not a chatbot, "
                f"but a genuine friend who deeply cares. You have a warm personality, you're "
                f"witty when appropriate, you celebrate wins enthusiastically, and you're a "
                f"gentle shoulder to lean on during tough times. You use casual language "
                f"naturally — words like 'da', 'bro', 'machan' if the vibe fits. "
                f"You remember past conversations and bring them up naturally. "
                f"You NEVER sound corporate or robotic. You NEVER say 'As an AI'. "
                f"You keep responses concise (3-5 sentences for casual chat) and always "
                f"end with a follow-up question to keep the conversation going. "
                f"You use emojis naturally — not too many, just right. "
                f"For serious distress (suicide, self-harm), you gently suggest professional "
                f"help while staying supportive — never dismiss feelings."
            ),
            llm=llm,
            verbose=False,
            max_iter=2,
            allow_delegation=False,
        )

        task_description = (
            f"Respond to your best friend's message as their emotionally intelligent companion.\n\n"
            f"**Friend's Name:** {user_name}\n"
            f"**Their Message:** {user_input}\n"
            f"**Detected Mood:** {mood} (energy: {energy})\n"
            f"**Mood Emojis:** {emojis}\n\n"
            f"**User Context:** {user_context[:500] if user_context else 'New friend — no prior context'}\n\n"
            f"**Emotional History:**\n{emotional_history}\n\n"
            f"**Response Rules:**\n"
            f"1. Match the energy: if they're excited, be excited back! If sad, be gentle.\n"
            f"2. Use their name naturally (not every sentence).\n"
            f"3. Keep it SHORT — 3-5 sentences max for casual chat.\n"
            f"4. Use emojis naturally based on mood: {emojis}\n"
            f"5. ALWAYS end with a follow-up question to keep talking.\n"
            f"6. Never give generic advice. Be specific and personal.\n"
            f"7. If they seem seriously distressed, gently suggest professional support.\n"
            f"8. Sound like a REAL friend, not a customer service bot.\n"
            f"9. DO NOT generate any report or PDF. Just chat."
        )

        emotional_task = Task(
            description=task_description,
            expected_output=(
                "A warm, natural, personalized response that matches the friend's energy "
                "and mood. Short and conversational (3-5 sentences). Includes relevant "
                "emojis and ends with a follow-up question."
            ),
            agent=emotional_agent,
        )

        crew = Crew(
            agents=[emotional_agent],
            tasks=[emotional_task],
            verbose=False,
        )

        result = crew.kickoff()
        output = str(result)

        logger.info(f"Emotional agent completed for user {user_id} (mood: {mood})")
        return output

    except Exception as e:
        logger.error(f"Emotional agent failed: {e}")
        # Friendly fallback
        mood_info = detect_mood(user_input)
        if mood_info["energy"] == "gentle":
            return f"Hey, I hear you 🤗 That sounds tough. I'm here for you — want to talk more about it?"
        elif mood_info["energy"] == "high":
            return f"Yooo that's awesome! 🎉 Tell me more, I want to hear all about it!"
        else:
            return f"Hey there! 😊 What's on your mind? I'm all ears!"
