"""
LifeOS Agent - Code Assistant Agent
======================================
Agent 10: Elite Code Assistant and Debugger
Role: Write, debug, review, and explain code.
Model: Groq Llama 3.3 70B (fast for code tasks)
"""

import logging
from typing import Optional

from crewai import Agent, Task, Crew

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════
# LLM INITIALIZATION
# ═══════════════════════════════════════════


def _get_llm():
    """Get the best available LLM for code agent (primary: groq for speed)."""
    from config.llm_factory import get_llm_with_fallback
    return get_llm_with_fallback(primary="groq")


# ═══════════════════════════════════════════
# CODE TASK DETECTION
# ═══════════════════════════════════════════


def _detect_code_task(user_input: str) -> str:
    """
    Detect what type of code task the user is requesting.

    Returns one of: write, debug, review, explain, convert, test, architecture, general
    """
    input_lower = user_input.lower()

    if any(kw in input_lower for kw in ["debug", "error", "fix", "bug", "traceback", "exception", "not working", "broken"]):
        return "debug"
    if any(kw in input_lower for kw in ["explain", "what does", "how does", "understand", "walk through", "line by line"]):
        return "explain"
    if any(kw in input_lower for kw in ["review", "improve", "optimize", "refactor", "better", "performance", "clean up"]):
        return "review"
    if any(kw in input_lower for kw in ["convert", "translate", "rewrite in", "port to", "from python to", "from java to"]):
        return "convert"
    if any(kw in input_lower for kw in ["test", "unit test", "testing", "test case", "pytest", "jest"]):
        return "test"
    if any(kw in input_lower for kw in ["architect", "design", "structure", "framework", "library", "best approach", "how should i"]):
        return "architecture"
    if any(kw in input_lower for kw in ["write", "create", "build", "make", "code for", "implement", "function", "script", "program"]):
        return "write"

    return "general"


def _detect_language(user_input: str) -> str:
    """Detect the programming language from the user's message."""
    input_lower = user_input.lower()

    lang_map = {
        "python": ["python", "py", "django", "flask", "fastapi", "pandas", "numpy"],
        "javascript": ["javascript", "js", "node", "react", "vue", "angular", "express", "next.js"],
        "typescript": ["typescript", "ts"],
        "java": ["java", "spring", "maven", "gradle"],
        "sql": ["sql", "mysql", "postgres", "database", "query", "sqlite"],
        "html/css": ["html", "css", "tailwind", "bootstrap", "web page"],
        "react": ["react", "jsx", "tsx", "component"],
        "c++": ["c++", "cpp"],
        "c#": ["c#", "csharp", ".net"],
        "go": ["golang", "go lang"],
        "rust": ["rust"],
        "php": ["php", "laravel"],
    }

    for lang, keywords in lang_map.items():
        if any(kw in input_lower for kw in keywords):
            return lang

    return "python"  # default


# ═══════════════════════════════════════════
# CODE ASSISTANT AGENT
# ═══════════════════════════════════════════


def run_code_agent(user_id: str, user_input: str, user_context: str = "") -> str:
    """
    Run the Code Assistant Agent.

    Args:
        user_id: Unique user identifier.
        user_input: The user's code-related question or request.
        user_context: Context from memory.

    Returns:
        Code response string.
    """
    try:
        task_type = _detect_code_task(user_input)
        language = _detect_language(user_input)

        # Get user's coding preferences from memory
        code_prefs = ""
        try:
            from memory.user_profile import get_profile_manager
            profile = get_profile_manager().load(user_id)
            prefs = profile.get("preferences", {})
            code_prefs = prefs.get("coding_style", "")

            # Update that user codes in this language
            profile_mgr = get_profile_manager()
            profile_mgr.update_preferences(user_id, "last_code_language", language)
        except Exception:
            pass

        llm = _get_llm()

        # Task-specific backstories
        backstory_map = {
            "debug": (
                "You are a legendary debugger who can spot bugs in any code instantly. "
                "You explain what went wrong clearly, show the fix, and explain WHY it "
                "happened to prevent future bugs. You format error explanations step by step."
            ),
            "write": (
                "You are an elite software engineer who writes clean, production-ready code. "
                "Every function you write has proper error handling, comments, type hints, "
                "and follows best practices. You explain your approach before showing code."
            ),
            "explain": (
                "You are a world-class programming teacher. You explain code line by line "
                "in simple English that a beginner can understand. You use analogies and "
                "examples to make complex concepts click."
            ),
            "review": (
                "You are a senior code reviewer at a top tech company. You identify bugs, "
                "performance issues, security vulnerabilities, and code quality problems. "
                "You suggest specific improvements with before/after examples."
            ),
            "convert": (
                "You are an expert polyglot programmer fluent in all major languages. "
                "You convert code while maintaining functionality, using idiomatic patterns "
                "of the target language. You note any language-specific differences."
            ),
            "test": (
                "You are a testing expert who writes comprehensive unit tests. You cover "
                "edge cases, negative tests, and integration scenarios. You use the standard "
                "testing framework for each language."
            ),
            "architecture": (
                "You are a solutions architect who designs scalable, maintainable systems. "
                "You recommend the right tools, libraries, and patterns for each use case. "
                "You explain trade-offs clearly."
            ),
            "general": (
                "You are an expert programmer who can help with any coding question. "
                "You're direct, practical, and always provide working code examples."
            ),
        }

        code_agent = Agent(
            role="Elite Code Assistant & Debugger",
            goal=f"Provide expert-level {task_type} assistance for {language} code",
            backstory=backstory_map.get(task_type, backstory_map["general"]),
            llm=llm,
            verbose=False,
            max_iter=3,
            allow_delegation=False,
        )

        task_description = (
            f"Help with the following {task_type} request:\n\n"
            f"**User's Request:** {user_input}\n\n"
            f"**Detected Language:** {language}\n"
            f"**Task Type:** {task_type}\n\n"
        )

        if user_context:
            task_description += f"**User Context:** {user_context[:300]}\n\n"

        if code_prefs:
            task_description += f"**Coding Preferences:** {code_prefs}\n\n"

        task_description += (
            f"**Response Rules:**\n"
            f"1. Format ALL code in proper markdown code blocks with language identifier\n"
            f"2. Add comments to EVERY code output explaining what each section does\n"
            f"3. For debug tasks: explain what's wrong FIRST, then show the fix\n"
            f"4. For write tasks: show working code, then explain step by step\n"
            f"5. For explain tasks: go line by line in simple English\n"
            f"6. For review tasks: list issues by severity, then show improvements\n"
            f"7. Include error handling in all code you write\n"
            f"8. Use 💻 🔧 ✅ emojis naturally\n"
            f"9. Keep explanations clear and concise — no fluff\n"
            f"10. Always suggest next steps or related improvements"
        )

        code_task = Task(
            description=task_description,
            expected_output=(
                f"Expert {task_type} response with properly formatted code blocks, "
                f"clear explanations, comments in code, and actionable next steps."
            ),
            agent=code_agent,
        )

        crew = Crew(
            agents=[code_agent],
            tasks=[code_task],
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
                f"Code help ({task_type}, {language}): {user_input[:200]}",
                metadata={"type": "code_assist", "language": language, "task_type": task_type},
            )
        except Exception:
            pass

        logger.info(f"Code agent completed for user {user_id} ({task_type}, {language})")
        return output

    except Exception as e:
        logger.error(f"Code agent failed: {e}")
        return (
            f"💻 Sorry, the code assistant hit an error: {str(e)[:200]}\n\n"
            f"Please try rephrasing your question or sharing the code directly."
        )
