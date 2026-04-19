"""
LifeOS Agent - Main Entry Point
==================================
Starts the LifeOS Agent Telegram bot.
Run with: python main.py
"""

import os
import sys
import logging
from datetime import datetime

from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# ═══════════════════════════════════════════
# LOGGING CONFIGURATION
# ═══════════════════════════════════════════

# Create data directories (uses env vars on Railway, local defaults otherwise)
os.makedirs(os.getenv("CHROMA_DB_PATH", os.path.join("data", "chroma_db")), exist_ok=True)
os.makedirs(os.getenv("PROFILES_PATH", os.path.join("data", "profiles")), exist_ok=True)

# Create logs directory
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            os.path.join("logs", f"lifeos_{datetime.now().strftime('%Y%m%d')}.log"),
            encoding="utf-8",
        ),
    ],
)

# Reduce noise from third-party libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)

logger = logging.getLogger("LifeOS")


# ═══════════════════════════════════════════
# STARTUP BANNER
# ═══════════════════════════════════════════


def print_banner():
    """Print the LifeOS Agent startup banner."""
    banner = """
╔══════════════════════════════════════════════════════╗
║                                                      ║
║        🧠  L I F E O S   A G E N T  🧠              ║
║                                                      ║
║   "One message. Full life intelligence. Zero cost."  ║
║                                                      ║
╠══════════════════════════════════════════════════════╣
║                                                      ║
║   Agents:   6 specialized AI agents                  ║
║   Models:   Gemini 2.5 Flash + Groq Llama 3.3       ║
║   Flow:     LangGraph state machine                  ║
║   Memory:   ChromaDB + Mem0 persistent memory        ║
║   Interface: Telegram Bot                            ║
║                                                      ║
╚══════════════════════════════════════════════════════╝
    """
    print(banner)


# ═══════════════════════════════════════════
# VALIDATE ENVIRONMENT
# ═══════════════════════════════════════════


def validate_environment():
    """Validate that required API keys and dependencies are available."""
    from config.models import validate_api_keys

    print("\n📋 Validating API Keys...")
    status = validate_api_keys()

    all_ok = True
    for service, available in status.items():
        icon = "✅" if available else "❌"
        print(f"  {icon} {service.upper()}: {'configured' if available else 'MISSING'}")
        if service == "telegram" and not available:
            all_ok = False

    if not status.get("telegram"):
        logger.error("TELEGRAM_BOT_TOKEN is required! Get one from @BotFather.")
        print("\n❌ CRITICAL: TELEGRAM_BOT_TOKEN is not set!")
        print("   Get a bot token from @BotFather on Telegram and add it to .env")
        sys.exit(1)

    # Check if at least one LLM is available
    has_llm = status.get("gemini") or status.get("groq") or status.get("openrouter")
    if not has_llm:
        logger.warning("No LLM API keys configured! Bot will have limited functionality.")
        print("\n⚠️  WARNING: No LLM API keys found!")
        print("   Add at least one of: GEMINI_API_KEY, GROQ_API_KEY, OPENROUTER_API_KEY")

    print()
    return all_ok


# ═══════════════════════════════════════════
# INITIALIZE SYSTEMS
# ═══════════════════════════════════════════


def initialize_systems():
    """Initialize all subsystems (memory, vector store, etc.)."""
    print("🔧 Initializing subsystems...")

    # Initialize ChromaDB
    try:
        from memory.chroma_store import get_chroma_store

        chroma = get_chroma_store()
        print("  ✅ ChromaDB vector store initialized")
    except Exception as e:
        logger.warning(f"ChromaDB initialization failed: {e}")
        print(f"  ⚠️  ChromaDB: {e}")

    # Initialize User Profile Manager
    try:
        from memory.user_profile import get_profile_manager

        profile_mgr = get_profile_manager()
        print("  ✅ User profile manager initialized")
    except Exception as e:
        logger.warning(f"Profile manager initialization failed: {e}")
        print(f"  ⚠️  Profile Manager: {e}")

    # Initialize API Rotator
    try:
        from tools.api_rotator import get_rotator

        rotator = get_rotator()
        status = rotator.get_status()
        total_keys = sum(s["total"] for s in status.values())
        print(f"  ✅ API rotator initialized ({total_keys} total keys)")
    except Exception as e:
        logger.warning(f"API rotator initialization failed: {e}")
        print(f"  ⚠️  API Rotator: {e}")

    # Test LangGraph workflow compilation
    try:
        from orchestrator.langgraph_flow import build_lifeos_graph

        graph = build_lifeos_graph()
        print("  ✅ LangGraph workflow compiled")
    except Exception as e:
        logger.warning(f"LangGraph compilation failed: {e}")
        print(f"  ⚠️  LangGraph: {e}")

    print()


# ═══════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════


def main():
    """Main entry point — start the LifeOS Agent."""
    print_banner()

    logger.info("LifeOS Agent starting...")

    # Validate environment
    validate_environment()

    # Initialize subsystems
    initialize_systems()

    # Start the Telegram bot
    print("🚀 Starting Telegram Bot...")
    logger.info("Launching Telegram bot...")

    try:
        from telegram_bot import run_bot

        run_bot()
    except KeyboardInterrupt:
        logger.info("LifeOS Agent stopped by user")
        print("\n\n👋 LifeOS Agent shut down gracefully. Goodbye!")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
