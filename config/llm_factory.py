"""
LifeOS Agent - LLM Factory
=============================
Centralized LLM initialization using CrewAI's native LLM class.
This ensures proper LiteLLM routing with provider/ prefixed model names.
"""

import os
import logging
from crewai import LLM
from config.models import GEMINI_CONFIG, GROQ_CONFIG, OPENROUTER_CONFIG

logger = logging.getLogger(__name__)


def get_llm(provider: str = "gemini") -> LLM:
    """
    Get a CrewAI LLM instance for the specified provider.

    Args:
        provider: One of 'gemini', 'groq', 'openrouter'.

    Returns:
        A CrewAI LLM object with proper LiteLLM routing.
    """
    if provider == "gemini":
        return LLM(
            model="gemini/gemini-2.5-flash-preview-04-17",
            api_key=GEMINI_CONFIG["api_key"],
            temperature=GEMINI_CONFIG["temperature"],
            max_tokens=GEMINI_CONFIG["max_tokens"],
        )
    elif provider == "groq":
        return LLM(
            model="groq/llama-3.3-70b-versatile",
            api_key=GROQ_CONFIG["api_key"],
            temperature=GROQ_CONFIG["temperature"],
            max_tokens=GROQ_CONFIG["max_tokens"],
        )
    elif provider == "openrouter":
        return LLM(
            model="openrouter/meta-llama/llama-3.3-70b-instruct:free",
            api_key=OPENROUTER_CONFIG["api_key"],
            base_url=OPENROUTER_CONFIG["base_url"],
            temperature=OPENROUTER_CONFIG["temperature"],
            max_tokens=OPENROUTER_CONFIG["max_tokens"],
        )
    raise ValueError(f"Unknown provider: {provider}")


def get_llm_with_fallback(primary: str = "gemini") -> LLM:
    """
    Try to initialize LLM with fallback chain: gemini → groq → openrouter.

    Args:
        primary: The preferred provider to start with.

    Returns:
        A CrewAI LLM object from the first available provider.

    Raises:
        RuntimeError: If no provider is available.
    """
    chain = ["gemini", "groq", "openrouter"]
    start = chain.index(primary) if primary in chain else 0
    for provider in chain[start:]:
        try:
            llm = get_llm(provider)
            logger.info(f"LLM initialized: {provider}")
            return llm
        except Exception as e:
            logger.warning(f"Provider {provider} failed: {e}")
    raise RuntimeError("No LLM provider available. Check your API keys.")
