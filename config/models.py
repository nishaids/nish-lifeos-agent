"""
LifeOS Agent - Model Configuration
===================================
All free LLM model configurations for the LifeOS Agent system.
Supports Gemini 2.5 Flash, Groq Llama 3.3 70B, and OpenRouter free models.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ═══════════════════════════════════════════
# GEMINI CONFIGURATION (Primary - Smart Reasoning)
# ═══════════════════════════════════════════

GEMINI_CONFIG = {
    "model_name": "gemini/gemini-2.5-flash-preview-04-17",
    "api_key": os.getenv("GEMINI_API_KEY", ""),
    "temperature": 0.7,
    "max_tokens": 4096,
    "top_p": 0.95,
    "provider": "google",
}

# ═══════════════════════════════════════════
# GROQ CONFIGURATION (Secondary - Fast Responses)
# ═══════════════════════════════════════════

GROQ_CONFIG = {
    "model_name": "groq/llama-3.3-70b-versatile",
    "api_key": os.getenv("GROQ_API_KEY", ""),
    "temperature": 0.7,
    "max_tokens": 4096,
    "provider": "groq",
}

# ═══════════════════════════════════════════
# OPENROUTER CONFIGURATION (Fallback/Backup)
# ═══════════════════════════════════════════

OPENROUTER_CONFIG = {
    "model_name": "openrouter/meta-llama/llama-3.3-70b-instruct:free",
    "api_key": os.getenv("OPENROUTER_API_KEY", ""),
    "base_url": "https://openrouter.ai/api/v1",
    "temperature": 0.7,
    "max_tokens": 4096,
    "provider": "openrouter",
}

# ═══════════════════════════════════════════
# SEARCH & TOOLS CONFIGURATION
# ═══════════════════════════════════════════

TAVILY_CONFIG = {
    "api_key": os.getenv("TAVILY_API_KEY", ""),
    "max_results": 5,
}

MEM0_CONFIG = {
    "api_key": os.getenv("MEM0_API_KEY", ""),
}

TELEGRAM_CONFIG = {
    "bot_token": os.getenv("TELEGRAM_BOT_TOKEN", ""),
}

# ═══════════════════════════════════════════
# MODEL PRIORITY ORDER
# ═══════════════════════════════════════════

MODEL_PRIORITY = ["gemini", "groq", "openrouter"]

# ═══════════════════════════════════════════
# AGENT-TO-MODEL MAPPING
# ═══════════════════════════════════════════

AGENT_MODEL_MAP = {
    "goal_agent": "gemini",
    "research_agent": "groq",
    "strategy_agent": "gemini",
    "planner_agent": "groq",
    "report_agent": "gemini",
    "memory_agent": "openrouter",
}

# ═══════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════


def get_model_config(provider: str) -> dict:
    """Get model configuration by provider name."""
    configs = {
        "gemini": GEMINI_CONFIG,
        "groq": GROQ_CONFIG,
        "openrouter": OPENROUTER_CONFIG,
    }
    return configs.get(provider, GEMINI_CONFIG)


def get_fallback_provider(current_provider: str) -> str:
    """Get the next fallback provider in priority order."""
    try:
        current_index = MODEL_PRIORITY.index(current_provider)
        next_index = current_index + 1
        if next_index < len(MODEL_PRIORITY):
            return MODEL_PRIORITY[next_index]
    except ValueError:
        pass
    return MODEL_PRIORITY[-1]


def get_agent_provider(agent_name: str) -> str:
    """Get the assigned model provider for a given agent."""
    return AGENT_MODEL_MAP.get(agent_name, "gemini")


def validate_api_keys() -> dict:
    """Validate which API keys are available and return status."""
    status = {
        "gemini": bool(GEMINI_CONFIG["api_key"]),
        "groq": bool(GROQ_CONFIG["api_key"]),
        "openrouter": bool(OPENROUTER_CONFIG["api_key"]),
        "tavily": bool(TAVILY_CONFIG["api_key"]),
        "mem0": bool(MEM0_CONFIG["api_key"]),
        "telegram": bool(TELEGRAM_CONFIG["bot_token"]),
    }
    return status
