"""
LifeOS Agent - Telegram Bot Interface
========================================
Telegram bot that serves as the primary user interface.
Handles commands, messages, status updates, and PDF delivery.
"""

import os
import logging
import asyncio
from datetime import datetime

from telegram import Update, BotCommand
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from telegram.constants import ChatAction, ParseMode

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════
# WELCOME / HELP MESSAGES
# ═══════════════════════════════════════════

WELCOME_MESSAGE = """
🧠 **Welcome to LifeOS Agent!**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

*One message. Full life intelligence. Zero cost.*

I'm your personal AI life operating system. I manage your career, learning, goals, and daily priorities using 6 specialized AI agents.

**🔮 What I Can Do:**
• 🔍 Deep research on any topic
• 🎯 Track and analyze your goals
• ♟️ Build personalized life strategies
• 📋 Create actionable daily plans
• 📄 Generate professional PDF reports
• 🧠 Remember your preferences across chats

**📝 How to Use:**
Just send me any message! For example:
→ *"I want to become a machine learning engineer"*
→ *"Help me build a study plan for data science"*
→ *"What career path should I take in AI?"*

**⚡ Commands:**
/goals — View your saved goals
/clear — Clear memory and start fresh
/help — Show this guide

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 *Send me your first message to get started!*
"""

HELP_MESSAGE = """
📖 **LifeOS Agent — How to Use**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**🗨️ Just Chat Naturally:**
Send any message about your goals, career, learning, or life questions. I'll automatically:

1. 🧠 Load your profile & history
2. 🔍 Research your topic online
3. 🎯 Track your goals
4. ♟️ Build a personalized strategy
5. 📋 Create an action plan
6. 📄 Generate a PDF report

**💬 Example Messages:**
• "I want to learn Python for data science"
• "How can I get promoted to senior developer?"
• "Create a fitness plan for building muscle"
• "What should I know about investing in index funds?"
• "Help me prepare for a job interview at Google"

**⚡ Commands:**
• /start — Welcome message
• /goals — View your tracked goals
• /clear — Reset your profile
• /help — This guide

**💡 Tips:**
• Be specific in your messages for better results
• Mention your goals and I'll track them
• I remember your preferences across conversations

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
*Powered by 6 AI Agents • Gemini • Groq • LangGraph*
"""

# ═══════════════════════════════════════════
# COMMAND HANDLERS
# ═══════════════════════════════════════════


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command — send welcome message."""
    try:
        user = update.effective_user
        user_id = str(user.id)

        # Save user name if available
        try:
            from memory.user_profile import get_profile_manager

            profile_mgr = get_profile_manager()
            name = user.first_name or user.username or "User"
            profile_mgr.update_name(user_id, name)
        except Exception as e:
            logger.warning(f"Failed to save user name: {e}")

        await update.message.reply_text(
            WELCOME_MESSAGE,
            parse_mode=ParseMode.MARKDOWN,
        )
        logger.info(f"/start from user {user_id}")

    except Exception as e:
        logger.error(f"Start command failed: {e}")
        await update.message.reply_text(
            "👋 Welcome to LifeOS Agent! Send me any message to get started."
        )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help command — send usage guide."""
    try:
        await update.message.reply_text(
            HELP_MESSAGE,
            parse_mode=ParseMode.MARKDOWN,
        )
    except Exception as e:
        logger.error(f"Help command failed: {e}")
        await update.message.reply_text(
            "Send me any message and I'll research, strategize, and plan for you!"
        )


async def goals_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /goals command — show user's saved goals."""
    try:
        user_id = str(update.effective_user.id)

        from memory.user_profile import get_profile_manager

        profile_mgr = get_profile_manager()
        goals_summary = profile_mgr.get_goals_summary(user_id)

        await update.message.reply_text(
            f"🎯 **Your Goals Dashboard**\n\n{goals_summary}",
            parse_mode=ParseMode.MARKDOWN,
        )

    except Exception as e:
        logger.error(f"Goals command failed: {e}")
        await update.message.reply_text(
            "Could not retrieve goals. Try sending a message with your goals!"
        )


async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /clear command — clear user memory and start fresh."""
    try:
        user_id = str(update.effective_user.id)

        # Clear user profile
        from memory.user_profile import get_profile_manager

        profile_mgr = get_profile_manager()
        profile_mgr.clear_profile(user_id)

        # Clear ChromaDB data
        try:
            from memory.chroma_store import get_chroma_store

            chroma = get_chroma_store()
            chroma.clear_user_data(user_id)
        except Exception as e:
            logger.warning(f"Failed to clear ChromaDB: {e}")

        await update.message.reply_text(
            "🗑️ **Memory cleared!**\n\n"
            "Your profile, goals, and conversation history have been reset.\n"
            "Send me a message to start fresh! 🚀",
            parse_mode=ParseMode.MARKDOWN,
        )
        logger.info(f"/clear from user {user_id}")

    except Exception as e:
        logger.error(f"Clear command failed: {e}")
        await update.message.reply_text("Failed to clear memory. Please try again.")


# ═══════════════════════════════════════════
# MESSAGE HANDLER (MAIN PIPELINE)
# ═══════════════════════════════════════════


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle incoming text messages — triggers the full LifeOS pipeline.
    Sends status updates, text summary, and PDF report.
    """
    user = update.effective_user
    user_id = str(user.id)
    user_input = update.message.text.strip()
    chat_id = update.effective_chat.id

    if not user_input:
        await update.message.reply_text("Please send a text message!")
        return

    logger.info(f"Message from user {user_id}: {user_input[:50]}")

    # Save user name
    try:
        from memory.user_profile import get_profile_manager

        profile_mgr = get_profile_manager()
        name = user.first_name or user.username or "User"
        profile_mgr.update_name(user_id, name)
    except Exception:
        pass

    # Send initial status
    status_msg = await update.message.reply_text(
        "🚀 **LifeOS Agent Activated!**\n\n"
        "Processing your request through 6 AI agents...\n"
        "⏳ This may take 1-2 minutes.",
        parse_mode=ParseMode.MARKDOWN,
    )

    # Status callback to update the message
    async def status_callback(emoji: str, message: str):
        """Update the status message with current progress."""
        try:
            await context.bot.send_chat_action(
                chat_id=chat_id, action=ChatAction.TYPING
            )
            await status_msg.edit_text(
                f"🚀 **LifeOS Processing...**\n\n"
                f"{emoji} {message}",
                parse_mode=ParseMode.MARKDOWN,
            )
        except Exception as e:
            logger.warning(f"Status update failed: {e}")

    # Run the LifeOS pipeline
    try:
        from orchestrator.langgraph_flow import run_lifeos_pipeline

        final_state = await run_lifeos_pipeline(
            user_id=user_id,
            user_input=user_input,
            status_callback=status_callback,
        )

        # Update status to complete
        try:
            await status_msg.edit_text(
                "✅ **Analysis Complete!**",
                parse_mode=ParseMode.MARKDOWN,
            )
        except Exception:
            pass

        # Send the text summary (first 2500 chars)
        report_content = final_state.get("report_content", "")
        if report_content:
            # Split long messages (Telegram limit is 4096 chars)
            summary = report_content[:2500]
            try:
                await update.message.reply_text(
                    f"📊 **LifeOS Intelligence Report**\n\n{summary}",
                    parse_mode=ParseMode.MARKDOWN,
                )
            except Exception:
                # If markdown fails, send without formatting
                await update.message.reply_text(
                    f"📊 LifeOS Intelligence Report\n\n{summary}"
                )
        else:
            await update.message.reply_text(
                "⚠️ Report generation had issues. Here's what I found:\n\n"
                f"Research: {final_state.get('research_data', 'N/A')[:500]}\n\n"
                f"Strategy: {final_state.get('strategy_data', 'N/A')[:500]}"
            )

        # Send PDF report if available
        pdf_path = final_state.get("pdf_path", "")
        if pdf_path and os.path.exists(pdf_path):
            try:
                with open(pdf_path, "rb") as pdf_file:
                    await context.bot.send_document(
                        chat_id=chat_id,
                        document=pdf_file,
                        filename=os.path.basename(pdf_path),
                        caption="📄 Your full LifeOS Intelligence Report",
                    )
                logger.info(f"PDF sent to user {user_id}: {pdf_path}")

                # Clean up PDF file after sending
                try:
                    os.remove(pdf_path)
                except Exception:
                    pass

            except Exception as e:
                logger.error(f"Failed to send PDF: {e}")
                await update.message.reply_text(
                    "📄 PDF report was generated but couldn't be sent. "
                    "Please try again."
                )
        else:
            logger.warning(f"No PDF available for user {user_id}")

    except Exception as e:
        logger.error(f"Pipeline failed for user {user_id}: {e}")
        try:
            await status_msg.edit_text(
                "⚠️ Processing encountered an issue.",
                parse_mode=ParseMode.MARKDOWN,
            )
        except Exception:
            pass

        await update.message.reply_text(
            "❌ **Sorry, something went wrong.**\n\n"
            f"Error: {str(e)[:200]}\n\n"
            "Please try again. If the issue persists, try:\n"
            "1. Simplify your message\n"
            "2. Use /clear to reset\n"
            "3. Wait a moment and retry",
            parse_mode=ParseMode.MARKDOWN,
        )


# ═══════════════════════════════════════════
# ERROR HANDLER
# ═══════════════════════════════════════════


async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle errors in the bot."""
    logger.error(f"Bot error: {context.error}")

    if update and update.message:
        try:
            await update.message.reply_text(
                "⚠️ An unexpected error occurred. Please try again."
            )
        except Exception:
            pass


# ═══════════════════════════════════════════
# BOT SETUP AND RUN
# ═══════════════════════════════════════════


def create_bot_application() -> Application:
    """
    Create and configure the Telegram bot application.

    Returns:
        Configured Application instance.
    """
    from config.models import TELEGRAM_CONFIG

    token = TELEGRAM_CONFIG["bot_token"]
    if not token:
        raise ValueError(
            "TELEGRAM_BOT_TOKEN not set! Get one from @BotFather on Telegram."
        )

    # Build the application
    app = Application.builder().token(token).build()

    # Register command handlers
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("goals", goals_command))
    app.add_handler(CommandHandler("clear", clear_command))

    # Register message handler (for all text messages)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Register error handler
    app.add_error_handler(error_handler)

    logger.info("Telegram bot application created and configured")
    return app


async def set_bot_commands(app: Application) -> None:
    """Set the bot's command menu in Telegram."""
    try:
        commands = [
            BotCommand("start", "Welcome and features overview"),
            BotCommand("help", "How to use LifeOS Agent"),
            BotCommand("goals", "View your saved goals"),
            BotCommand("clear", "Clear memory and start fresh"),
        ]
        await app.bot.set_my_commands(commands)
        logger.info("Bot commands set successfully")
    except Exception as e:
        logger.warning(f"Failed to set bot commands: {e}")


def run_bot() -> None:
    """Start the Telegram bot with polling."""
    try:
        app = create_bot_application()

        # Set bot commands on startup
        async def post_init(application: Application) -> None:
            await set_bot_commands(application)

        app.post_init = post_init

        logger.info("Starting LifeOS Telegram bot with polling...")
        print("🧠 LifeOS Agent bot is running! Press Ctrl+C to stop.")
        app.run_polling(
            drop_pending_updates=True,
            allowed_updates=Update.ALL_TYPES,
        )

    except Exception as e:
        logger.error(f"Bot failed to start: {e}")
        raise
