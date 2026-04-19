"""
LifeOS Agent - Email Manager Agent
=====================================
Agent 8: Gmail Intelligence Manager
Role: Connect to Gmail API, scan/categorize/delete emails, send emails.
Model: Groq (fast for categorization)
"""

import os
import base64
import logging
from typing import Optional
from email.mime.text import MIMEText

from crewai import Agent, Task, Crew

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════
# LLM INITIALIZATION
# ═══════════════════════════════════════════


def _get_llm():
    """Get the best available LLM for email agent (primary: groq for speed)."""
    from config.llm_factory import get_llm_with_fallback
    return get_llm_with_fallback(primary="groq")


# ═══════════════════════════════════════════
# GMAIL API SETUP
# ═══════════════════════════════════════════

SCOPES = [
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
]

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
CREDENTIALS_PATH = os.getenv("GMAIL_CREDENTIALS_PATH", os.path.join(DATA_DIR, "credentials.json"))
TOKEN_PATH = os.getenv("GMAIL_TOKEN_PATH", os.path.join(DATA_DIR, "token.json"))


def _get_gmail_service():
    """
    Authenticate and return a Gmail API service instance.

    Returns:
        Gmail API service object or None if not configured.
    """
    try:
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from google.auth.transport.requests import Request
        from googleapiclient.discovery import build

        creds = None

        # Load existing token
        if os.path.exists(TOKEN_PATH):
            creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)

        # Refresh or create new credentials
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            elif os.path.exists(CREDENTIALS_PATH):
                flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
                creds = flow.run_local_server(port=0)
            else:
                logger.warning("Gmail credentials.json not found. Email features disabled.")
                return None

            # Save token for future use
            os.makedirs(os.path.dirname(TOKEN_PATH), exist_ok=True)
            with open(TOKEN_PATH, "w") as token_file:
                token_file.write(creds.to_json())

        service = build("gmail", "v1", credentials=creds)
        logger.info("Gmail API service initialized successfully")
        return service

    except Exception as e:
        logger.error(f"Gmail API initialization failed: {e}")
        return None


# ═══════════════════════════════════════════
# GMAIL OPERATIONS
# ═══════════════════════════════════════════


def get_unread_emails(max_results: int = 10) -> list:
    """Fetch unread emails from inbox."""
    service = _get_gmail_service()
    if not service:
        return []

    try:
        results = service.users().messages().list(
            userId="me", q="is:unread", maxResults=max_results
        ).execute()

        messages = results.get("messages", [])
        emails = []

        for msg in messages:
            msg_data = service.users().messages().get(
                userId="me", id=msg["id"], format="metadata",
                metadataHeaders=["From", "Subject", "Date"]
            ).execute()

            headers = {h["name"]: h["value"] for h in msg_data.get("payload", {}).get("headers", [])}
            snippet = msg_data.get("snippet", "")

            emails.append({
                "id": msg["id"],
                "from": headers.get("From", "Unknown"),
                "subject": headers.get("Subject", "No Subject"),
                "date": headers.get("Date", "Unknown"),
                "snippet": snippet[:150],
                "labels": msg_data.get("labelIds", []),
            })

        return emails
    except Exception as e:
        logger.error(f"Failed to fetch unread emails: {e}")
        return []


def get_spam_emails() -> list:
    """Fetch spam/promotional emails."""
    service = _get_gmail_service()
    if not service:
        return []

    try:
        spam_emails = []

        # Get SPAM folder
        spam_results = service.users().messages().list(
            userId="me", labelIds=["SPAM"], maxResults=50
        ).execute()
        spam_msgs = spam_results.get("messages", [])

        # Get PROMOTIONS category
        promo_results = service.users().messages().list(
            userId="me", q="category:promotions", maxResults=50
        ).execute()
        promo_msgs = promo_results.get("messages", [])

        all_msgs = spam_msgs + promo_msgs
        seen_ids = set()

        for msg in all_msgs:
            if msg["id"] in seen_ids:
                continue
            seen_ids.add(msg["id"])

            try:
                msg_data = service.users().messages().get(
                    userId="me", id=msg["id"], format="metadata",
                    metadataHeaders=["From", "Subject"]
                ).execute()

                headers = {h["name"]: h["value"] for h in msg_data.get("payload", {}).get("headers", [])}
                size_estimate = msg_data.get("sizeEstimate", 0)

                spam_emails.append({
                    "id": msg["id"],
                    "from": headers.get("From", "Unknown"),
                    "subject": headers.get("Subject", "No Subject"),
                    "size_bytes": size_estimate,
                    "labels": msg_data.get("labelIds", []),
                })
            except Exception:
                continue

        return spam_emails
    except Exception as e:
        logger.error(f"Failed to fetch spam emails: {e}")
        return []


def delete_spam_emails() -> dict:
    """Delete all spam and promotional emails."""
    service = _get_gmail_service()
    if not service:
        return {"deleted": 0, "freed_mb": 0, "error": "Gmail not connected"}

    try:
        spam_emails = get_spam_emails()
        if not spam_emails:
            return {"deleted": 0, "freed_mb": 0}

        total_size = 0
        deleted_count = 0

        for email in spam_emails:
            try:
                service.users().messages().delete(
                    userId="me", id=email["id"]
                ).execute()
                total_size += email.get("size_bytes", 0)
                deleted_count += 1
            except Exception as e:
                logger.warning(f"Failed to delete email {email['id']}: {e}")

        freed_mb = round(total_size / (1024 * 1024), 2)

        logger.info(f"Deleted {deleted_count} spam emails, freed {freed_mb}MB")
        return {"deleted": deleted_count, "freed_mb": freed_mb}

    except Exception as e:
        logger.error(f"Failed to delete spam: {e}")
        return {"deleted": 0, "freed_mb": 0, "error": str(e)}


def send_email(to: str, subject: str, body: str) -> bool:
    """Send an email via Gmail."""
    service = _get_gmail_service()
    if not service:
        return False

    try:
        message = MIMEText(body)
        message["to"] = to
        message["subject"] = subject

        raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
        body_data = {"raw": raw}

        service.users().messages().send(userId="me", body=body_data).execute()
        logger.info(f"Email sent to {to}")
        return True

    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        return False


def get_email_dashboard() -> str:
    """Get a formatted email dashboard summary."""
    service = _get_gmail_service()
    if not service:
        return (
            "📧 **Gmail Not Connected**\n\n"
            "To use email features, add your `credentials.json` file to the `/data/` folder.\n\n"
            "**Setup Steps:**\n"
            "1. Go to Google Cloud Console → APIs → Gmail API\n"
            "2. Create OAuth2 credentials (Desktop App)\n"
            "3. Download `credentials.json`\n"
            "4. Place it in the `data/` folder\n"
            "5. Run the bot and authenticate"
        )

    try:
        # Get unread count
        unread = get_unread_emails(max_results=50)
        unread_count = len(unread)

        # Get spam count
        spam = get_spam_emails()
        spam_count = len(spam)
        spam_size = sum(e.get("size_bytes", 0) for e in spam)
        spam_size_mb = round(spam_size / (1024 * 1024), 2)

        # Format top 5 unread
        unread_summary = ""
        for email in unread[:5]:
            sender = email["from"].split("<")[0].strip()[:30]
            subject = email["subject"][:40]
            unread_summary += f"  📩 **{sender}** — {subject}\n"

        if not unread_summary:
            unread_summary = "  ✅ Inbox zero! No unread emails.\n"

        dashboard = (
            f"📧 **Email Dashboard**\n"
            f"━━━━━━━━━━━━━━━━━━━━\n\n"
            f"📬 **Unread:** {unread_count} emails\n"
            f"🗑️ **Spam/Promos:** {spam_count} emails ({spam_size_mb}MB)\n\n"
            f"📩 **Latest Unread:**\n{unread_summary}\n"
        )

        if spam_count > 0:
            dashboard += (
                f"💡 Say **\"delete spam\"** or **\"clean inbox\"** to remove "
                f"{spam_count} spam emails and free {spam_size_mb}MB!"
            )

        return dashboard

    except Exception as e:
        logger.error(f"Email dashboard failed: {e}")
        return "⚠️ Could not load email dashboard. Please try again."


# ═══════════════════════════════════════════
# EMAIL MANAGER AGENT
# ═══════════════════════════════════════════


def run_email_agent(user_id: str, user_input: str, user_context: str = "") -> str:
    """
    Run the Email Manager Agent to handle email operations.

    Args:
        user_id: Unique user identifier.
        user_input: The user's email-related command.
        user_context: Context from memory.

    Returns:
        Response string with email operation results.
    """
    input_lower = user_input.lower()

    # Direct action handlers (no LLM needed for these)
    if any(kw in input_lower for kw in ["delete spam", "clean inbox", "delete junk", "remove spam"]):
        return _handle_delete_spam(user_id)

    if any(kw in input_lower for kw in ["dashboard", "check mail", "check email", "inbox", "unread"]):
        return get_email_dashboard()

    if "send" in input_lower and ("email" in input_lower or "mail" in input_lower):
        return _handle_send_email(user_id, user_input)

    # For complex email queries, use the LLM agent
    try:
        # Get email data for context
        unread = get_unread_emails(5)
        unread_text = ""
        for e in unread:
            unread_text += f"- From: {e['from']}, Subject: {e['subject']}, Snippet: {e['snippet']}\n"

        if not unread_text:
            unread_text = "No unread emails."

        llm = _get_llm()

        email_agent = Agent(
            role="Gmail Intelligence Manager",
            goal="Help the user manage their email efficiently and intelligently",
            backstory=(
                "You are an expert email manager who helps users tame their inbox. "
                "You can summarize emails, suggest replies, categorize messages, "
                "and give smart recommendations for inbox management. You're efficient, "
                "direct, and always provide actionable advice. Use 📧 emojis naturally."
            ),
            llm=llm,
            verbose=False,
            max_iter=2,
            allow_delegation=False,
        )

        task_description = (
            f"Help the user with their email request.\n\n"
            f"**User Request:** {user_input}\n\n"
            f"**Current Unread Emails:**\n{unread_text}\n\n"
            f"**User Context:** {user_context[:300] if user_context else 'No context'}\n\n"
            f"**Instructions:**\n"
            f"1. Address the user's specific email request\n"
            f"2. Provide clear, actionable response\n"
            f"3. Use 📧 emojis naturally\n"
            f"4. Keep response concise and helpful\n"
            f"5. If they want to send an email, ask for recipient, subject, and body\n"
            f"6. Suggest follow-up actions"
        )

        email_task = Task(
            description=task_description,
            expected_output="A helpful, concise response addressing the user's email request with actionable items.",
            agent=email_agent,
        )

        crew = Crew(
            agents=[email_agent],
            tasks=[email_task],
            verbose=False,
        )

        result = crew.kickoff()
        return str(result)

    except Exception as e:
        logger.error(f"Email agent failed: {e}")
        return get_email_dashboard()


def _handle_delete_spam(user_id: str) -> str:
    """Handle the spam deletion flow."""
    spam = get_spam_emails()
    if not spam:
        return "✅ Your inbox is already clean! No spam or promotional emails found. 📧"

    spam_count = len(spam)
    spam_size = sum(e.get("size_bytes", 0) for e in spam)
    spam_size_mb = round(spam_size / (1024 * 1024), 2)

    # Get top spam senders
    senders = {}
    for e in spam:
        sender = e["from"].split("<")[0].strip()[:30]
        senders[sender] = senders.get(sender, 0) + 1

    top_senders = sorted(senders.items(), key=lambda x: x[1], reverse=True)[:5]
    sender_list = "\n".join([f"  • {s[0]} ({s[1]} emails)" for s in top_senders])

    # Perform deletion
    result = delete_spam_emails()

    if result["deleted"] > 0:
        return (
            f"🗑️ **Inbox Cleaned!**\n\n"
            f"✅ Deleted **{result['deleted']}** spam/promotional emails\n"
            f"💾 Freed **{result['freed_mb']}MB** of storage!\n\n"
            f"**Top spam senders removed:**\n{sender_list}\n\n"
            f"📧 Your inbox is now cleaner! 🎉"
        )
    else:
        return f"⚠️ Could not delete emails. Error: {result.get('error', 'Unknown')}"


def _handle_send_email(user_id: str, user_input: str) -> str:
    """Handle the email sending flow using LLM to parse the request."""
    try:
        llm = _get_llm()

        parse_agent = Agent(
            role="Email Parser",
            goal="Extract email recipient, subject, and body from user's message",
            backstory="You extract email details from natural language.",
            llm=llm,
            verbose=False,
            max_iter=1,
            allow_delegation=False,
        )

        parse_task = Task(
            description=(
                f"Extract email details from this message: '{user_input}'\n\n"
                f"Return EXACTLY in this format:\n"
                f"TO: [email address]\n"
                f"SUBJECT: [email subject]\n"
                f"BODY: [email body]\n\n"
                f"If any field is missing, write MISSING for that field."
            ),
            expected_output="TO: [email], SUBJECT: [subject], BODY: [body]",
            agent=parse_agent,
        )

        crew = Crew(agents=[parse_agent], tasks=[parse_task], verbose=False)
        result = str(crew.kickoff())

        # Parse the result
        to_addr = ""
        subject = ""
        body = ""

        for line in result.split("\n"):
            line = line.strip()
            if line.upper().startswith("TO:"):
                to_addr = line[3:].strip()
            elif line.upper().startswith("SUBJECT:"):
                subject = line[8:].strip()
            elif line.upper().startswith("BODY:"):
                body = line[5:].strip()

        if not to_addr or to_addr == "MISSING" or "@" not in to_addr:
            return (
                "📧 I need more details to send the email!\n\n"
                "Please specify:\n"
                "• **To:** recipient@email.com\n"
                "• **Subject:** What's the email about?\n"
                "• **Message:** What should the email say?\n\n"
                "Example: *\"Send email to john@gmail.com saying meeting postponed to 3pm\"*"
            )

        if send_email(to_addr, subject, body):
            return f"✅ **Email Sent!** ✉️\n\n📬 To: {to_addr}\n📝 Subject: {subject}\n\nEmail delivered successfully!"
        else:
            return "⚠️ Failed to send email. Please check Gmail connection."

    except Exception as e:
        logger.error(f"Send email failed: {e}")
        return "⚠️ Could not process email request. Please try again."
