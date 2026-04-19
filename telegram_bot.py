"""
LifeOS Agent - Telegram Bot Interface (v2 — Diamond Edition)
===============================================================
Telegram bot with intent-based routing, photo handling,
new commands, and smart emoji system.
Handles: text, photos, commands, and status updates.
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
# EMOJI SYSTEM
# ═══════════════════════════════════════════

INTENT_EMOJIS = {
    "emotional": "😊🤗✨",
    "email": "📧✉️📬",
    "image": "🖼️📸🔍",
    "code": "💻🔧✅",
    "thinker": "🧠💡🔍",
    "assignment": "📚✍️🎓",
    "work": "💼📋🎯",
    "life_pipeline": "🚀🎯📊",
}

# ═══════════════════════════════════════════
# WELCOME / HELP MESSAGES
# ═══════════════════════════════════════════

WELCOME_MESSAGE = """
🧠 **Welcome to LifeOS Diamond Agent!**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

*Your brilliant best friend who happens to be an expert in everything.*

I'm 12 specialized AI agents in one — I understand what you need and respond accordingly. No setup needed, just chat naturally!

**🔮 What I Can Do:**
• 🤗 Chat like your best friend — I match your energy
• 📧 Manage your Gmail — read, clean, send emails
• 🖼️ Analyze any image you send me
• 💻 Write, debug, and review code
• 🧠 Think deeply about complex questions
• 📚 Write assignments, notes, and study material
• 💼 Draft emails, resumes, and presentations
• 🎯 Track goals and build life strategies
• 📄 Generate PDF reports (only when you ask)

**⚡ Commands:**
/email — Email dashboard
/clean — Clean spam from inbox
/mood — How are you feeling?
/code — Code assistant mode
/think — Deep thinking mode
/goals — View your goals
/clear — Reset memory
/help — This guide

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 *Just send me any message — I'll figure out what you need!*
"""

HELP_MESSAGE = """
📖 **LifeOS Diamond Agent — How to Use**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**🗨️ Just Chat Naturally!** I automatically detect your intent:

💬 **Casual Chat** → "Hey!", "How are you?" → Friendly conversation
😔 **Emotional Support** → "I'm stressed", "Feeling low" → Caring response
📧 **Email** → "Check my inbox", "Delete spam" → Gmail management
🖼️ **Send a Photo** → Any image → Visual analysis
💻 **Code** → "Debug this", "Write a Python script" → Code help
🧠 **Deep Questions** → "Why do we dream?", "Explain quantum physics" → Deep analysis
📚 **Academics** → "Write an essay on...", "Make notes" → Academic content
💼 **Work** → "Draft an email", "Create a presentation" → Professional content
🎯 **Life Planning** → "Help me plan my career" → Full 6-agent pipeline

**📄 PDF Reports:**
Only generated when you specifically ask: "Generate a report" or "Give me a PDF"

**💡 Tips:**
• Be specific for better results
• Send images for instant analysis
• I remember everything across conversations
• For code: paste the error directly

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
*Powered by 12 AI Agents • Gemini • Groq • LangGraph*
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
            "👋 Welcome to LifeOS Diamond Agent! Send me any message to get started."
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
            "Send me any message and I'll figure out what you need! 🧠"
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


async def email_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /email command — show email dashboard."""
    try:
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id, action=ChatAction.TYPING
        )

        from agents.email_manager_agent import get_email_dashboard
        dashboard = get_email_dashboard()

        await update.message.reply_text(
            dashboard,
            parse_mode=ParseMode.MARKDOWN,
        )
    except Exception as e:
        logger.error(f"Email command failed: {e}")
        await update.message.reply_text(
            "📧 Email dashboard unavailable. Make sure Gmail is configured."
        )


async def clean_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /clean command — delete spam/promotional emails."""
    try:
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id, action=ChatAction.TYPING
        )

        user_id = str(update.effective_user.id)

        # Trigger email agent to handle spam deletion
        from agents.email_manager_agent import _handle_delete_spam
        result = _handle_delete_spam(user_id)

        await update.message.reply_text(
            result,
            parse_mode=ParseMode.MARKDOWN,
        )
    except Exception as e:
        logger.error(f"Clean command failed: {e}")
        await update.message.reply_text(
            "🗑️ Inbox cleaning failed. Make sure Gmail is configured."
        )


async def mood_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /mood command — emotional check-in."""
    await update.message.reply_text(
        "🤗 **Mood Check-in**\n\n"
        "Hey! How are you feeling right now? Tell me honestly — I'm here for you.\n\n"
        "You can say things like:\n"
        "• \"I'm feeling great today!\"\n"
        "• \"Kind of stressed about exams\"\n"
        "• \"Bored and don't know what to do\"\n"
        "• \"Need someone to talk to\"\n\n"
        "I'll match your energy and we'll figure it out together! 💙",
        parse_mode=ParseMode.MARKDOWN,
    )


async def code_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /code command — activate code mode."""
    await update.message.reply_text(
        "💻 **Code Assistant Activated!**\n\n"
        "I can help you with:\n"
        "• ✍️ **Write code** — \"Write a Python function to sort a list\"\n"
        "• 🐛 **Debug** — Paste your error or code\n"
        "• 🔍 **Review** — \"Review this code for bugs\"\n"
        "• 📖 **Explain** — \"Explain this code line by line\"\n"
        "• 🔄 **Convert** — \"Convert this Python to JavaScript\"\n"
        "• 🧪 **Test** — \"Write unit tests for this function\"\n\n"
        "Just send your code or question! 🔧",
        parse_mode=ParseMode.MARKDOWN,
    )


async def think_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /think command — activate deep thinking mode."""
    await update.message.reply_text(
        "🧠 **Deep Thinking Mode Activated!**\n\n"
        "Ask me anything complex:\n"
        "• 🤔 \"Why do humans procrastinate?\"\n"
        "• 🌍 \"What if gravity was twice as strong?\"\n"
        "• 💡 \"Explain blockchain like I'm 5\"\n"
        "• ♟️ \"Should I choose stability or passion?\"\n"
        "• 🔮 \"What will AI look like in 2050?\"\n\n"
        "I'll use mental models, multiple perspectives, and deep analysis "
        "to give you insights that go 3 levels deeper than usual. 💡",
        parse_mode=ParseMode.MARKDOWN,
    )


# ═══════════════════════════════════════════
# MESSAGE HANDLER (MAIN — TEXT)
# ═══════════════════════════════════════════


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle incoming text messages — routes to the correct agent via intent detection.
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

    # Detect intent for status message
    from orchestrator.langgraph_flow import detect_intent
    intent = detect_intent(user_input, has_image=False)
    intent_emojis = INTENT_EMOJIS.get(intent, "✨")

    # Choose status message based on intent
    if intent == "emotional":
        initial_status = f"{intent_emojis[0]} Thinking..."
    elif intent == "life_pipeline":
        initial_status = (
            "🚀 **LifeOS Agent Activated!**\n\n"
            "Processing your request through 6 AI agents...\n"
            "⏳ This may take 1-2 minutes."
        )
    else:
        status_map = {
            "email": "📧 Checking your emails...",
            "code": "💻 Working on your code...",
            "thinker": "🧠 Thinking deeply...",
            "assignment": "📚 Working on your content...",
            "work": "💼 Creating your content...",
        }
        initial_status = status_map.get(intent, f"{intent_emojis[0]} Processing...")

    # Send initial status
    status_msg = await update.message.reply_text(
        initial_status,
        parse_mode=ParseMode.MARKDOWN,
    )

    # Show typing indicator
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    # Status callback to update the message
    async def status_callback(emoji: str, message: str):
        """Update the status message with current progress."""
        try:
            await context.bot.send_chat_action(
                chat_id=chat_id, action=ChatAction.TYPING
            )
            await status_msg.edit_text(
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
            has_image=False,
            status_callback=status_callback,
        )

        detected_intent = final_state.get("intent", intent)

        # Delete the status message for clean UX (non-pipeline routes)
        if detected_intent != "life_pipeline":
            try:
                await status_msg.delete()
            except Exception:
                try:
                    await status_msg.edit_text("✅")
                except Exception:
                    pass

        # Handle response based on intent
        if detected_intent == "life_pipeline":
            await _handle_pipeline_response(update, context, final_state, status_msg, chat_id)
        else:
            await _handle_agent_response(update, context, final_state, detected_intent)

    except Exception as e:
        logger.error(f"Pipeline failed for user {user_id}: {e}")
        try:
            await status_msg.edit_text("⚠️ Processing encountered an issue.")
        except Exception:
            pass

        await update.message.reply_text(
            f"❌ Sorry, something went wrong.\n\n"
            f"Error: {str(e)[:200]}\n\n"
            f"Please try again or use /clear to reset.",
            parse_mode=ParseMode.MARKDOWN,
        )


async def _handle_agent_response(update: Update, context: ContextTypes.DEFAULT_TYPE,
                                  final_state: dict, intent: str) -> None:
    """Handle a single-agent response (non-pipeline)."""
    response = final_state.get("agent_response", "")

    if not response:
        response = "Hmm, I couldn't generate a response. Could you try rephrasing? 🤔"

    # Split long messages (Telegram limit is 4096 chars)
    if len(response) > 4000:
        chunks = [response[i:i + 4000] for i in range(0, len(response), 4000)]
        for chunk in chunks:
            try:
                await update.message.reply_text(chunk, parse_mode=ParseMode.MARKDOWN)
            except Exception:
                await update.message.reply_text(chunk)
    else:
        try:
            await update.message.reply_text(response, parse_mode=ParseMode.MARKDOWN)
        except Exception:
            # If markdown fails, send without formatting
            await update.message.reply_text(response)


async def _handle_pipeline_response(update: Update, context: ContextTypes.DEFAULT_TYPE,
                                     final_state: dict, status_msg, chat_id: int) -> None:
    """Handle a full pipeline response (life_pipeline route)."""
    # Update status to complete
    try:
        await status_msg.edit_text(
            "✅ **Analysis Complete!**",
            parse_mode=ParseMode.MARKDOWN,
        )
    except Exception:
        pass

    # Send the text summary
    report_content = final_state.get("report_content", "")
    agent_response = final_state.get("agent_response", "")
    response = report_content or agent_response

    if response:
        # Split long messages (Telegram limit is 4096 chars)
        summary = response[:3500]
        try:
            await update.message.reply_text(
                f"📊 **LifeOS Intelligence Report**\n\n{summary}",
                parse_mode=ParseMode.MARKDOWN,
            )
        except Exception:
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
            logger.info(f"PDF sent: {pdf_path}")

            # Clean up PDF file after sending
            try:
                os.remove(pdf_path)
            except Exception:
                pass

        except Exception as e:
            logger.error(f"Failed to send PDF: {e}")
            await update.message.reply_text(
                "📄 PDF was generated but couldn't be sent. Please try again."
            )


# ═══════════════════════════════════════════
# PHOTO HANDLER
# ═══════════════════════════════════════════


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle incoming photos — download and analyze with Gemini Vision.
    """
    user = update.effective_user
    user_id = str(user.id)
    chat_id = update.effective_chat.id

    logger.info(f"Photo from user {user_id}")

    # Get caption if any
    caption = update.message.caption or ""

    # Send processing status
    status_msg = await update.message.reply_text(
        "🖼️ Analyzing your image...",
        parse_mode=ParseMode.MARKDOWN,
    )

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    try:
        # Download the highest resolution photo
        photo = update.message.photo[-1]  # Last element is highest res
        photo_file = await context.bot.get_file(photo.file_id)

        # Download to bytes
        image_bytes = await photo_file.download_as_bytearray()
        image_data = bytes(image_bytes)

        # Detect mime type from file path
        file_path = photo_file.file_path or ""
        if file_path.endswith(".png"):
            mime_type = "image/png"
        elif file_path.endswith(".webp"):
            mime_type = "image/webp"
        else:
            mime_type = "image/jpeg"

        # Run through the pipeline with image
        from orchestrator.langgraph_flow import run_lifeos_pipeline

        final_state = await run_lifeos_pipeline(
            user_id=user_id,
            user_input=caption or "Analyze this image",
            has_image=True,
            image_data=image_data,
            image_mime=mime_type,
        )

        response = final_state.get("agent_response", "")

        # Delete status message
        try:
            await status_msg.delete()
        except Exception:
            try:
                await status_msg.edit_text("✅")
            except Exception:
                pass

        if response:
            # Split long responses
            if len(response) > 4000:
                chunks = [response[i:i + 4000] for i in range(0, len(response), 4000)]
                for chunk in chunks:
                    try:
                        await update.message.reply_text(chunk, parse_mode=ParseMode.MARKDOWN)
                    except Exception:
                        await update.message.reply_text(chunk)
            else:
                try:
                    await update.message.reply_text(response, parse_mode=ParseMode.MARKDOWN)
                except Exception:
                    await update.message.reply_text(response)
        else:
            await update.message.reply_text(
                "🖼️ I received your image but couldn't analyze it. Please try again."
            )

    except Exception as e:
        logger.error(f"Photo handler failed for user {user_id}: {e}")
        try:
            await status_msg.edit_text(
                f"⚠️ Image analysis failed: {str(e)[:200]}\n\n"
                f"Please try sending the image again."
            )
        except Exception:
            await update.message.reply_text(
                f"⚠️ Image analysis failed. Please try again."
            )


# ═══════════════════════════════════════════
# ERROR HANDLER
# ═══════════════════════════════════════════


async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle errors in the bot."""
    # Handle Conflict errors gracefully (e.g., stale deployment still polling)
    if "Conflict" in str(context.error):
        logger.warning("Another bot instance conflict detected - will auto-resolve in seconds")
        return

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

    # Build the application with timeouts
    app = Application.builder().token(token).connect_timeout(30).read_timeout(30).build()

    # Register command handlers
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("goals", goals_command))
    app.add_handler(CommandHandler("clear", clear_command))
    app.add_handler(CommandHandler("email", email_command))
    app.add_handler(CommandHandler("clean", clean_command))
    app.add_handler(CommandHandler("mood", mood_command))
    app.add_handler(CommandHandler("code", code_command))
    app.add_handler(CommandHandler("think", think_command))

    # Register photo handler
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    # Register text message handler (for all text messages)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Register error handler
    app.add_error_handler(error_handler)

    logger.info("Telegram bot application created and configured (v2 — Diamond Edition)")
    return app


async def set_bot_commands(app: Application) -> None:
    """Set the bot's command menu in Telegram."""
    try:
        commands = [
            BotCommand("start", "Welcome and features overview"),
            BotCommand("help", "How to use LifeOS Agent"),
            BotCommand("goals", "View your saved goals"),
            BotCommand("email", "Email dashboard"),
            BotCommand("clean", "Clean spam from inbox"),
            BotCommand("mood", "Emotional check-in"),
            BotCommand("code", "Code assistant mode"),
            BotCommand("think", "Deep thinking mode"),
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

        # Forcefully clear any existing webhook/polling connections
        logger.info("Clearing any existing webhook connections...")
        async def clear_webhook():
            await app.bot.delete_webhook(drop_pending_updates=True)
        asyncio.get_event_loop().run_until_complete(clear_webhook())
        logger.info("Webhook cleared successfully")

        logger.info("Starting LifeOS Telegram bot with polling...")
        print("🧠 LifeOS Diamond Agent is running! Press Ctrl+C to stop.")
        app.run_polling(
            drop_pending_updates=True,
            allowed_updates=Update.ALL_TYPES,
            timeout=10,
            pool_timeout=10,
        )

    except Exception as e:
        logger.error(f"Bot failed to start: {e}")
        raise
