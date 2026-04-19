"""
LifeOS Agent - API Key Rotator
===============================
Round-robin API key rotation with automatic fallback.
Supports multiple keys per provider for higher rate limits.
"""

import os
import logging
from itertools import cycle
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════
# API ROTATOR CLASS
# ═══════════════════════════════════════════


class APIRotator:
    """
    Round-robin API key rotator with automatic fallback.
    Supports multiple comma-separated keys per provider in .env file.
    Example: GEMINI_API_KEY=key1,key2,key3
    """

    def __init__(self):
        self._gemini_keys = self._load_keys("GEMINI_API_KEY")
        self._groq_keys = self._load_keys("GROQ_API_KEY")
        self._openrouter_keys = self._load_keys("OPENROUTER_API_KEY")
        self._tavily_keys = self._load_keys("TAVILY_API_KEY")

        self._gemini_cycle = cycle(self._gemini_keys) if self._gemini_keys else None
        self._groq_cycle = cycle(self._groq_keys) if self._groq_keys else None
        self._openrouter_cycle = (
            cycle(self._openrouter_keys) if self._openrouter_keys else None
        )
        self._tavily_cycle = cycle(self._tavily_keys) if self._tavily_keys else None

        self._failed_keys: set = set()

        logger.info(
            f"APIRotator initialized — Gemini: {len(self._gemini_keys)} keys, "
            f"Groq: {len(self._groq_keys)} keys, "
            f"OpenRouter: {len(self._openrouter_keys)} keys, "
            f"Tavily: {len(self._tavily_keys)} keys"
        )

    @staticmethod
    def _load_keys(env_var: str) -> list:
        """Load comma-separated API keys from environment variable."""
        raw = os.getenv(env_var, "")
        if not raw:
            return []
        keys = [k.strip() for k in raw.split(",") if k.strip()]
        return keys

    def _get_next_key(self, key_cycle, keys: list) -> Optional[str]:
        """Get the next available key from the cycle, skipping failed keys."""
        if not key_cycle or not keys:
            return None

        attempts = 0
        max_attempts = len(keys)

        while attempts < max_attempts:
            key = next(key_cycle)
            if key not in self._failed_keys:
                return key
            attempts += 1

        # All keys have failed — reset and try again
        logger.warning("All keys in rotation have failed. Resetting failed keys.")
        self._failed_keys.clear()
        return next(key_cycle)

    def get_gemini_key(self) -> Optional[str]:
        """Get the next available Gemini API key."""
        key = self._get_next_key(self._gemini_cycle, self._gemini_keys)
        if not key:
            logger.warning("No Gemini API keys available")
        return key

    def get_groq_key(self) -> Optional[str]:
        """Get the next available Groq API key."""
        key = self._get_next_key(self._groq_cycle, self._groq_keys)
        if not key:
            logger.warning("No Groq API keys available")
        return key

    def get_openrouter_key(self) -> Optional[str]:
        """Get the next available OpenRouter API key."""
        key = self._get_next_key(self._openrouter_cycle, self._openrouter_keys)
        if not key:
            logger.warning("No OpenRouter API keys available")
        return key

    def get_tavily_key(self) -> Optional[str]:
        """Get the next available Tavily API key."""
        key = self._get_next_key(self._tavily_cycle, self._tavily_keys)
        if not key:
            logger.warning("No Tavily API keys available")
        return key

    def mark_key_failed(self, key: str) -> None:
        """Mark a key as failed so it is skipped in future rotations."""
        self._failed_keys.add(key)
        logger.warning(f"API key marked as failed: {key[:8]}...")

    def get_key_for_provider(self, provider: str) -> Optional[str]:
        """Get the next key for a given provider name."""
        provider_map = {
            "gemini": self.get_gemini_key,
            "groq": self.get_groq_key,
            "openrouter": self.get_openrouter_key,
            "tavily": self.get_tavily_key,
        }
        getter = provider_map.get(provider)
        if getter:
            return getter()
        logger.error(f"Unknown provider: {provider}")
        return None

    def get_status(self) -> dict:
        """Return current status of all API key pools."""
        return {
            "gemini": {
                "total": len(self._gemini_keys),
                "failed": len(
                    [k for k in self._gemini_keys if k in self._failed_keys]
                ),
            },
            "groq": {
                "total": len(self._groq_keys),
                "failed": len(
                    [k for k in self._groq_keys if k in self._failed_keys]
                ),
            },
            "openrouter": {
                "total": len(self._openrouter_keys),
                "failed": len(
                    [k for k in self._openrouter_keys if k in self._failed_keys]
                ),
            },
            "tavily": {
                "total": len(self._tavily_keys),
                "failed": len(
                    [k for k in self._tavily_keys if k in self._failed_keys]
                ),
            },
        }


# ═══════════════════════════════════════════
# GLOBAL SINGLETON INSTANCE
# ═══════════════════════════════════════════

_rotator_instance: Optional[APIRotator] = None


def get_rotator() -> APIRotator:
    """Get or create the global APIRotator singleton."""
    global _rotator_instance
    if _rotator_instance is None:
        _rotator_instance = APIRotator()
    return _rotator_instance
