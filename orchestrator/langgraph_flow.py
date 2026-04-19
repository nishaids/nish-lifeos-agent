"""
LifeOS Agent - LangGraph Orchestration Flow (v2 — Diamond Edition)
====================================================================
Intent-based routing state machine that connects 12+ agents.
First detects intent, then routes to the correct agent or pipeline.

Routes:
  emotional  → Emotional Agent (friendly chat, mood support)
  email      → Email Manager Agent (Gmail operations)
  image      → Image Analysis Agent (Gemini Vision)
  code       → Code Assistant Agent (write/debug/review)
  thinker    → Master Thinker Agent (deep analysis)
  assignment → Assignment Agent (academic content)
  work       → Work Agent (professional content)
  life_pipeline → Full 6-agent pipeline (Memory→Research→Goals→Strategy→Plan→Report)
"""

import logging
from typing import TypedDict, Optional

from langgraph.graph import StateGraph, END

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════
# STATE DEFINITION
# ═══════════════════════════════════════════


class LifeOSState(TypedDict):
    """State that flows through the entire LifeOS pipeline."""
    user_id: str
    user_input: str
    has_image: bool
    image_data: bytes
    image_mime: str
    intent: str
    user_profile: dict
    user_context: str
    goals_data: str
    research_data: str
    strategy_data: str
    plan_data: str
    report_content: str
    pdf_path: str
    agent_response: str
    status: str
    error: str


# ═══════════════════════════════════════════
# INTENT DETECTION
# ═══════════════════════════════════════════


def detect_intent(user_input: str, has_image: bool = False) -> str:
    """
    Detect the user's intent from their message.

    Priority order matters — more specific intents checked first.

    Args:
        user_input: The user's message text.
        has_image: Whether the user sent an image.

    Returns:
        Intent string: one of emotional, email, image, code, assignment,
        work, thinker, life_pipeline.
    """
    if has_image:
        return "image"

    input_lower = user_input.lower().strip()

    # Exact match greetings — short casual messages
    greetings = [
        "hi", "hello", "hey", "sup", "what's up", "how are you",
        "good morning", "good night", "gm", "gn", "hola", "yo",
        "wassup", "whats up", "howdy", "namaste", "vanakkam",
        "hii", "hiii", "hiiii", "heyy", "heyyy",
    ]
    if input_lower in greetings or len(input_lower) < 10:
        return "emotional"

    # Emotional keywords — feelings and mood
    emotional_keywords = [
        "stressed", "sad", "anxious", "worried", "depressed", "lonely",
        "happy", "excited", "angry", "tired", "bored", "miss", "love",
        "hate", "feel", "feeling", "emotion", "cry", "crying", "hurt",
        "scared", "afraid", "panic", "overwhelmed", "frustrated",
        "grateful", "thankful", "proud", "confused", "lost",
        "heartbroken", "burned out", "burnout", "vent", "rant",
    ]
    if any(kw in input_lower for kw in emotional_keywords):
        return "emotional"

    # Email keywords — Gmail operations
    email_keywords = [
        "email", "gmail", "inbox", "spam", "mail", "send mail",
        "check mail", "delete mail", "clean inbox", "unread",
        "send email", "check email", "delete spam", "junk",
    ]
    if any(kw in input_lower for kw in email_keywords):
        return "email"

    # Code keywords — programming tasks
    code_keywords = [
        "code", "debug", "error", "function", "program", "script",
        "bug", "syntax", "python", "javascript", "java ", "sql",
        "html", "css", "react", "api", "fix this", "write code",
        "traceback", "exception", "compile", "runtime", "algorithm",
        "data structure", "class", "object", "variable", "loop",
        "array", "list", "dict", "json", "regex", "git", "docker",
        "deploy", "server", "backend", "frontend", "database",
        "mongodb", "postgresql", "flask", "django", "fastapi",
        "node.js", "express", "typescript", "npm", "pip",
    ]
    if any(kw in input_lower for kw in code_keywords):
        return "code"

    # Assignment keywords — academic tasks
    assignment_keywords = [
        "assignment", "essay", "notes", "summarize", "exam",
        "study", "chapter", "academic", "write about", "research paper",
        "explain the topic", "homework", "thesis", "dissertation",
        "question bank", "mind map", "revision", "semester",
        "textbook", "syllabus", "marks", "grade",
    ]
    if any(kw in input_lower for kw in assignment_keywords):
        return "assignment"

    # Work keywords — professional tasks
    work_keywords = [
        "draft", "write email", "professional", "linkedin", "resume",
        "cover letter", "meeting", "proposal", "presentation",
        "business", "corporate", "agenda", "minutes", "task list",
        "prioritize", "deadline", "project plan", "slide",
    ]
    if any(kw in input_lower for kw in work_keywords):
        return "work"

    # Thinker keywords — deep questions
    thinker_keywords = [
        "why", "how does", "explain", "what if", "philosophy",
        "meaning", "theory", "analyze", "deep", "complex",
        "understand", "think about", "perspective", "opinion",
        "debate", "argue", "pros and cons", "trade-off",
        "first principles", "mental model", "thought experiment",
    ]
    if any(kw in input_lower for kw in thinker_keywords):
        return "thinker"

    # Life pipeline keywords — goals, strategy, reports
    life_keywords = [
        "goal", "plan", "strategy", "life", "career", "future",
        "report", "pdf", "generate report", "life plan",
        "roadmap", "milestone", "vision", "track progress",
    ]
    if any(kw in input_lower for kw in life_keywords):
        return "life_pipeline"

    # Default to emotional/friendly chat
    return "emotional"


# ═══════════════════════════════════════════
# INTENT ROUTER NODE
# ═══════════════════════════════════════════


def intent_router_node(state: LifeOSState) -> dict:
    """
    Node 0: Detect intent and route to the correct agent.
    """
    user_input = state["user_input"]
    has_image = state.get("has_image", False)

    intent = detect_intent(user_input, has_image)
    logger.info(f"[Intent Router] Detected intent: {intent} for: {user_input[:50]}")

    return {"intent": intent, "status": "intent_detected"}


# ═══════════════════════════════════════════
# SINGLE-AGENT NODES (direct response, no pipeline)
# ═══════════════════════════════════════════


def emotional_node(state: LifeOSState) -> dict:
    """Run the Emotional Support Agent for friendly chat."""
    user_id = state["user_id"]
    user_input = state["user_input"]
    user_context = state.get("user_context", "")
    logger.info(f"[Emotional Node] Processing for user {user_id}")

    try:
        from agents.emotional_agent import run_emotional_agent
        response = run_emotional_agent(user_id, user_input, user_context)
        return {"agent_response": response, "status": "complete"}
    except Exception as e:
        logger.error(f"[Emotional Node] Failed: {e}")
        return {
            "agent_response": "Hey! 😊 I'm here for you. What's on your mind?",
            "status": "complete",
            "error": str(e),
        }


def email_node(state: LifeOSState) -> dict:
    """Run the Email Manager Agent."""
    user_id = state["user_id"]
    user_input = state["user_input"]
    user_context = state.get("user_context", "")
    logger.info(f"[Email Node] Processing for user {user_id}")

    try:
        from agents.email_manager_agent import run_email_agent
        response = run_email_agent(user_id, user_input, user_context)
        return {"agent_response": response, "status": "complete"}
    except Exception as e:
        logger.error(f"[Email Node] Failed: {e}")
        return {
            "agent_response": "📧 Email features aren't configured yet. Add credentials.json to data/ folder.",
            "status": "complete",
            "error": str(e),
        }


def image_node(state: LifeOSState) -> dict:
    """Run the Image Analysis Agent."""
    user_id = state["user_id"]
    user_input = state["user_input"]
    image_data = state.get("image_data", b"")
    image_mime = state.get("image_mime", "image/jpeg")
    logger.info(f"[Image Node] Processing for user {user_id}")

    try:
        from agents.image_analysis_agent import run_image_agent
        response = run_image_agent(user_id, image_data, user_input, image_mime)
        return {"agent_response": response, "status": "complete"}
    except Exception as e:
        logger.error(f"[Image Node] Failed: {e}")
        return {
            "agent_response": "🖼️ Image analysis failed. Please try again.",
            "status": "complete",
            "error": str(e),
        }


def code_node(state: LifeOSState) -> dict:
    """Run the Code Assistant Agent."""
    user_id = state["user_id"]
    user_input = state["user_input"]
    user_context = state.get("user_context", "")
    logger.info(f"[Code Node] Processing for user {user_id}")

    try:
        from agents.code_assistant_agent import run_code_agent
        response = run_code_agent(user_id, user_input, user_context)
        return {"agent_response": response, "status": "complete"}
    except Exception as e:
        logger.error(f"[Code Node] Failed: {e}")
        return {
            "agent_response": f"💻 Code assistant error: {str(e)[:200]}",
            "status": "complete",
            "error": str(e),
        }


def thinker_node(state: LifeOSState) -> dict:
    """Run the Master Thinker Agent."""
    user_id = state["user_id"]
    user_input = state["user_input"]
    user_context = state.get("user_context", "")
    logger.info(f"[Thinker Node] Processing for user {user_id}")

    try:
        from agents.master_thinker_agent import run_thinker_agent
        response = run_thinker_agent(user_id, user_input, user_context)
        return {"agent_response": response, "status": "complete"}
    except Exception as e:
        logger.error(f"[Thinker Node] Failed: {e}")
        return {
            "agent_response": f"🧠 Deep thinking hit an issue: {str(e)[:200]}",
            "status": "complete",
            "error": str(e),
        }


def assignment_node(state: LifeOSState) -> dict:
    """Run the Assignment Agent."""
    user_id = state["user_id"]
    user_input = state["user_input"]
    user_context = state.get("user_context", "")
    logger.info(f"[Assignment Node] Processing for user {user_id}")

    try:
        from agents.assignment_agent import run_assignment_agent
        response = run_assignment_agent(user_id, user_input, user_context)
        return {"agent_response": response, "status": "complete"}
    except Exception as e:
        logger.error(f"[Assignment Node] Failed: {e}")
        return {
            "agent_response": f"📚 Assignment assistant error: {str(e)[:200]}",
            "status": "complete",
            "error": str(e),
        }


def work_node(state: LifeOSState) -> dict:
    """Run the Work Agent."""
    user_id = state["user_id"]
    user_input = state["user_input"]
    user_context = state.get("user_context", "")
    logger.info(f"[Work Node] Processing for user {user_id}")

    try:
        from agents.work_agent import run_work_agent
        response = run_work_agent(user_id, user_input, user_context)
        return {"agent_response": response, "status": "complete"}
    except Exception as e:
        logger.error(f"[Work Node] Failed: {e}")
        return {
            "agent_response": f"💼 Work assistant error: {str(e)[:200]}",
            "status": "complete",
            "error": str(e),
        }


# ═══════════════════════════════════════════
# LIFE PIPELINE NODES (existing 6-agent pipeline)
# ═══════════════════════════════════════════


def memory_node(state: LifeOSState) -> dict:
    """
    Node 1: Load user profile, goals, and historical context.
    Calls the Memory Agent to analyze and summarize user context.
    """
    user_id = state["user_id"]
    user_input = state["user_input"]
    logger.info(f"[Memory Node] Processing user {user_id}")

    try:
        from agents.memory_agent import run_memory_agent, load_user_context

        # Load raw profile data
        context = load_user_context(user_id, user_input)
        user_profile = context.get("user_profile", {})

        # Run memory agent for intelligent context summary
        try:
            user_context = run_memory_agent(user_id, user_input)
        except Exception as e:
            logger.warning(f"Memory agent CrewAI failed, using raw context: {e}")
            user_context = (
                f"{context.get('profile_summary', '')}\n"
                f"{context.get('goals_summary', '')}\n"
                f"{context.get('past_context', '')}"
            )

        return {
            "user_profile": user_profile,
            "user_context": user_context,
            "status": "memory_loaded",
        }

    except Exception as e:
        logger.error(f"[Memory Node] Failed: {e}")
        return {
            "user_profile": {},
            "user_context": "New user - no prior context available.",
            "status": "memory_failed",
            "error": str(e),
        }


def quick_memory_node(state: LifeOSState) -> dict:
    """
    Lightweight memory load for single-agent routes.
    Loads raw context without running the memory CrewAI agent.
    """
    user_id = state["user_id"]
    user_input = state["user_input"]
    logger.info(f"[Quick Memory] Loading context for user {user_id}")

    try:
        from agents.memory_agent import load_user_context

        context = load_user_context(user_id, user_input)
        user_context = (
            f"{context.get('profile_summary', '')}\n"
            f"{context.get('goals_summary', '')}\n"
            f"{context.get('past_context', '')}"
        )

        return {
            "user_profile": context.get("user_profile", {}),
            "user_context": user_context,
            "status": "memory_loaded",
        }
    except Exception as e:
        logger.warning(f"[Quick Memory] Failed: {e}")
        return {
            "user_profile": {},
            "user_context": "",
            "status": "memory_skipped",
        }


def research_node(state: LifeOSState) -> dict:
    """
    Node 2: Research the user's topic using web search, Wikipedia, and ArXiv.
    """
    user_input = state["user_input"]
    user_context = state.get("user_context", "")
    logger.info(f"[Research Node] Researching: {user_input[:50]}")

    try:
        from agents.research_agent import run_research_agent

        research_data = run_research_agent(user_input, user_context)

        return {
            "research_data": research_data,
            "status": "research_complete",
        }

    except Exception as e:
        logger.error(f"[Research Node] Failed: {e}")
        # Retry with direct search as fallback
        try:
            from tools.search_tools import smart_search

            fallback_data = smart_search(user_input)
            return {
                "research_data": fallback_data,
                "status": "research_partial",
            }
        except Exception as e2:
            return {
                "research_data": f"Research unavailable for: {user_input}",
                "status": "research_failed",
                "error": str(e2),
            }


def goals_node(state: LifeOSState) -> dict:
    """
    Node 3: Track and analyze user goals.
    """
    user_id = state["user_id"]
    user_input = state["user_input"]
    user_context = state.get("user_context", "")
    logger.info(f"[Goals Node] Processing goals for user {user_id}")

    try:
        from agents.goal_agent import run_goal_agent

        goals_data = run_goal_agent(user_id, user_input, user_context)

        return {
            "goals_data": goals_data,
            "status": "goals_complete",
        }

    except Exception as e:
        logger.error(f"[Goals Node] Failed: {e}")
        # Fallback to raw goals
        try:
            from memory.user_profile import get_profile_manager

            goals_summary = get_profile_manager().get_goals_summary(user_id)
            return {
                "goals_data": goals_summary,
                "status": "goals_partial",
            }
        except Exception:
            return {
                "goals_data": "No goals data available.",
                "status": "goals_failed",
                "error": str(e),
            }


def strategy_node(state: LifeOSState) -> dict:
    """
    Node 4: Build personalized life strategy.
    """
    user_input = state["user_input"]
    research_data = state.get("research_data", "")
    goals_data = state.get("goals_data", "")
    user_context = state.get("user_context", "")
    logger.info(f"[Strategy Node] Building strategy for: {user_input[:50]}")

    try:
        from agents.strategy_agent import run_strategy_agent

        strategy_data = run_strategy_agent(
            user_input, research_data, goals_data, user_context
        )

        return {
            "strategy_data": strategy_data,
            "status": "strategy_complete",
        }

    except Exception as e:
        logger.error(f"[Strategy Node] Failed: {e}")
        return {
            "strategy_data": (
                f"Strategic focus: Prioritize actions related to '{user_input}'. "
                f"Break down into manageable steps and review progress weekly."
            ),
            "status": "strategy_failed",
            "error": str(e),
        }


def planning_node(state: LifeOSState) -> dict:
    """
    Node 5: Create actionable daily/weekly plan.
    """
    user_input = state["user_input"]
    strategy_data = state.get("strategy_data", "")
    goals_data = state.get("goals_data", "")
    user_context = state.get("user_context", "")
    logger.info(f"[Planning Node] Creating plan for: {user_input[:50]}")

    try:
        from agents.planner_agent import run_planner_agent

        plan_data = run_planner_agent(
            user_input, strategy_data, goals_data, user_context
        )

        return {
            "plan_data": plan_data,
            "status": "planning_complete",
        }

    except Exception as e:
        logger.error(f"[Planning Node] Failed: {e}")
        return {
            "plan_data": (
                f"Action Plan for '{user_input}':\n"
                f"1. Research the topic thoroughly (30 min)\n"
                f"2. Identify top 3 priorities (15 min)\n"
                f"3. Take the first concrete step today\n"
                f"4. Review progress at end of day"
            ),
            "status": "planning_failed",
            "error": str(e),
        }


def report_node(state: LifeOSState) -> dict:
    """
    Node 6: Write the comprehensive intelligence report.
    """
    user_input = state["user_input"]
    research_data = state.get("research_data", "")
    strategy_data = state.get("strategy_data", "")
    plan_data = state.get("plan_data", "")
    goals_data = state.get("goals_data", "")
    user_context = state.get("user_context", "")
    logger.info(f"[Report Node] Writing report for: {user_input[:50]}")

    try:
        from agents.report_agent import run_report_agent

        report_content = run_report_agent(
            user_input, research_data, strategy_data,
            plan_data, goals_data, user_context,
        )

        return {
            "report_content": report_content,
            "agent_response": report_content,
            "status": "report_complete",
        }

    except Exception as e:
        logger.error(f"[Report Node] Failed: {e}")
        # Assemble basic report from available data
        from datetime import datetime

        date_str = datetime.now().strftime("%B %d, %Y")
        report_content = (
            f"# LifeOS Intelligence Report\n"
            f"**Topic:** {user_input}\n"
            f"**Date:** {date_str}\n\n"
            f"## Research\n{research_data[:1500]}\n\n"
            f"## Strategy\n{strategy_data[:1500]}\n\n"
            f"## Action Plan\n{plan_data[:1500]}\n\n"
            f"## Goals\n{goals_data[:1000]}\n\n"
            f"---\n*Generated by LifeOS Agent*"
        )
        return {
            "report_content": report_content,
            "agent_response": report_content,
            "status": "report_partial",
        }


def pdf_node(state: LifeOSState) -> dict:
    """
    Node 7: Generate PDF from the report content.
    Only runs when explicitly requested in life_pipeline route.
    """
    report_content = state.get("report_content", "")
    user_input = state["user_input"]
    user_profile = state.get("user_profile", {})
    user_name = user_profile.get("name", "User") or "User"

    # Only generate PDF if user explicitly asked for it
    input_lower = user_input.lower()
    wants_pdf = any(kw in input_lower for kw in ["pdf", "report", "generate", "document"])

    if not wants_pdf:
        logger.info("[PDF Node] Skipping PDF — not explicitly requested")
        return {"pdf_path": "", "status": "complete"}

    logger.info(f"[PDF Node] Generating PDF for: {user_input[:50]}")

    try:
        from agents.report_agent import generate_pdf_report

        pdf_path = generate_pdf_report(
            report_content=report_content,
            topic=user_input,
            user_name=user_name,
        )

        return {
            "pdf_path": pdf_path,
            "status": "complete",
        }

    except Exception as e:
        logger.error(f"[PDF Node] Failed: {e}")
        return {
            "pdf_path": "",
            "status": "complete_no_pdf",
            "error": str(e),
        }


def save_memory_node(state: LifeOSState) -> dict:
    """
    Final node: Save the interaction to memory systems.
    """
    user_id = state["user_id"]
    user_input = state["user_input"]
    agent_response = state.get("agent_response", "")
    report_content = state.get("report_content", "")
    logger.info(f"[Save Memory Node] Saving interaction for user {user_id}")

    try:
        from agents.memory_agent import save_interaction

        # Save a summary of the interaction
        response_text = report_content or agent_response
        summary = response_text[:500] if response_text else "No response generated"
        save_interaction(user_id, user_input, summary)

        # Try to extract and save topics of interest
        try:
            from memory.user_profile import get_profile_manager

            profile_mgr = get_profile_manager()
            # Add the query topic as an interest
            topic_words = user_input.split()[:5]
            topic = " ".join(topic_words)
            if len(topic) > 3:
                profile_mgr.add_interest(user_id, topic)
        except Exception as e:
            logger.warning(f"Failed to save interest: {e}")

    except Exception as e:
        logger.warning(f"[Save Memory Node] Failed: {e}")

    return {"status": state.get("status", "complete")}


# ═══════════════════════════════════════════
# ROUTING LOGIC
# ═══════════════════════════════════════════


def route_by_intent(state: LifeOSState) -> str:
    """
    Conditional edge: route to the correct agent node based on detected intent.
    """
    intent = state.get("intent", "emotional")
    logger.info(f"[Router] Routing to: {intent}")

    route_map = {
        "emotional": "emotional_agent",
        "email": "email_agent",
        "image": "image_agent",
        "code": "code_agent",
        "thinker": "thinker_agent",
        "assignment": "assignment_agent",
        "work": "work_agent",
        "life_pipeline": "memory",
    }

    return route_map.get(intent, "emotional_agent")


# ═══════════════════════════════════════════
# BUILD THE LANGGRAPH WORKFLOW
# ═══════════════════════════════════════════


def build_lifeos_graph() -> StateGraph:
    """
    Build and compile the LifeOS LangGraph workflow with intent routing.

    Flow:
      START → intent_router → (conditional routing)
        → emotional_agent → save_memory → END
        → email_agent → save_memory → END
        → image_agent → save_memory → END
        → code_agent → save_memory → END
        → thinker_agent → save_memory → END
        → assignment_agent → save_memory → END
        → work_agent → save_memory → END
        → memory → research → goals → strategy → planning → report → pdf → save_memory → END

    Returns:
        Compiled LangGraph workflow.
    """
    # Create the state graph
    workflow = StateGraph(LifeOSState)

    # Add intent router node
    workflow.add_node("intent_router", intent_router_node)

    # Add single-agent nodes (quick routes)
    workflow.add_node("quick_memory", quick_memory_node)
    workflow.add_node("emotional_agent", emotional_node)
    workflow.add_node("email_agent", email_node)
    workflow.add_node("image_agent", image_node)
    workflow.add_node("code_agent", code_node)
    workflow.add_node("thinker_agent", thinker_node)
    workflow.add_node("assignment_agent", assignment_node)
    workflow.add_node("work_agent", work_node)

    # Add life pipeline nodes (full 6-agent pipeline)
    workflow.add_node("memory", memory_node)
    workflow.add_node("research", research_node)
    workflow.add_node("goals", goals_node)
    workflow.add_node("strategy", strategy_node)
    workflow.add_node("planning", planning_node)
    workflow.add_node("report", report_node)
    workflow.add_node("pdf", pdf_node)

    # Add save memory node (shared)
    workflow.add_node("save_memory", save_memory_node)

    # Set entry point
    workflow.set_entry_point("intent_router")

    # Conditional routing based on intent
    workflow.add_conditional_edges(
        "intent_router",
        route_by_intent,
        {
            "emotional_agent": "quick_memory",
            "email_agent": "email_agent",
            "image_agent": "image_agent",
            "code_agent": "quick_memory",
            "thinker_agent": "quick_memory",
            "assignment_agent": "quick_memory",
            "work_agent": "quick_memory",
            "memory": "memory",
        },
    )

    # Quick memory → agent routing (needs a second conditional)
    # For simplicity, we route quick_memory to the correct agent via intent check
    workflow.add_conditional_edges(
        "quick_memory",
        lambda state: state.get("intent", "emotional"),
        {
            "emotional": "emotional_agent",
            "code": "code_agent",
            "thinker": "thinker_agent",
            "assignment": "assignment_agent",
            "work": "work_agent",
        },
    )

    # Single-agent nodes → save_memory → END
    for agent_node in ["emotional_agent", "email_agent", "image_agent",
                        "code_agent", "thinker_agent", "assignment_agent", "work_agent"]:
        workflow.add_edge(agent_node, "save_memory")

    # Life pipeline: memory → research → goals → strategy → planning → report → pdf → save_memory
    workflow.add_edge("memory", "research")
    workflow.add_edge("research", "goals")
    workflow.add_edge("goals", "strategy")
    workflow.add_edge("strategy", "planning")
    workflow.add_edge("planning", "report")
    workflow.add_edge("report", "pdf")
    workflow.add_edge("pdf", "save_memory")

    # save_memory → END
    workflow.add_edge("save_memory", END)

    # Compile the graph
    compiled = workflow.compile()
    logger.info("LifeOS LangGraph workflow compiled successfully (v2 — Diamond Edition)")
    return compiled


# ═══════════════════════════════════════════
# RUN THE PIPELINE
# ═══════════════════════════════════════════


async def run_lifeos_pipeline(
    user_id: str,
    user_input: str,
    has_image: bool = False,
    image_data: bytes = b"",
    image_mime: str = "image/jpeg",
    status_callback=None,
) -> LifeOSState:
    """
    Run the LifeOS pipeline for a user query with intent-based routing.

    Args:
        user_id: Unique user identifier.
        user_input: The user's message/query.
        has_image: Whether the message includes an image.
        image_data: Raw image bytes (if has_image is True).
        image_mime: MIME type of the image.
        status_callback: Optional async callback for status updates.

    Returns:
        Final LifeOSState with all results.
    """
    logger.info(f"Starting LifeOS pipeline for user {user_id}: {user_input[:50]}")

    # Build the graph
    graph = build_lifeos_graph()

    # Initial state
    initial_state: LifeOSState = {
        "user_id": str(user_id),
        "user_input": user_input,
        "has_image": has_image,
        "image_data": image_data,
        "image_mime": image_mime,
        "intent": "",
        "user_profile": {},
        "user_context": "",
        "goals_data": "",
        "research_data": "",
        "strategy_data": "",
        "plan_data": "",
        "report_content": "",
        "pdf_path": "",
        "agent_response": "",
        "status": "starting",
        "error": "",
    }

    # Status messages for each stage
    status_messages = {
        "intent_router": ("🧭", "Understanding your message..."),
        "quick_memory": ("🧠", "Loading your context..."),
        "emotional_agent": ("🤗", "Thinking of the best response..."),
        "email_agent": ("📧", "Managing your emails..."),
        "image_agent": ("🖼️", "Analyzing your image..."),
        "code_agent": ("💻", "Working on your code..."),
        "thinker_agent": ("🧠", "Thinking deeply..."),
        "assignment_agent": ("📚", "Working on your assignment..."),
        "work_agent": ("💼", "Creating professional content..."),
        "memory": ("🧠", "Loading your profile and memories..."),
        "research": ("🔍", "Researching your topic..."),
        "goals": ("🎯", "Analyzing your goals..."),
        "strategy": ("♟️", "Building your strategy..."),
        "planning": ("📋", "Creating your action plan..."),
        "report": ("📝", "Writing your intelligence report..."),
        "pdf": ("📄", "Generating PDF report..."),
        "save_memory": ("💾", "Saving to memory..."),
    }

    # Run the graph with status updates
    final_state = initial_state.copy()

    try:
        # Stream through nodes for status updates
        for event in graph.stream(initial_state):
            for node_name, node_output in event.items():
                # Send status update
                if status_callback and node_name in status_messages:
                    emoji, message = status_messages[node_name]
                    try:
                        await status_callback(emoji, message)
                    except Exception as e:
                        logger.warning(f"Status callback failed: {e}")

                # Update final state with node output
                if isinstance(node_output, dict):
                    final_state.update(node_output)

                logger.info(f"Completed node: {node_name}")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        final_state["status"] = "pipeline_error"
        final_state["error"] = str(e)

    logger.info(f"Pipeline completed with status: {final_state.get('status', 'unknown')}")
    return final_state


def run_lifeos_pipeline_sync(user_id: str, user_input: str) -> LifeOSState:
    """
    Synchronous version of run_lifeos_pipeline.

    Args:
        user_id: Unique user identifier.
        user_input: The user's message/query.

    Returns:
        Final LifeOSState with all results.
    """
    import asyncio

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're inside an async context, create a new thread
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run,
                    run_lifeos_pipeline(user_id, user_input),
                )
                return future.result()
        else:
            return loop.run_until_complete(
                run_lifeos_pipeline(user_id, user_input)
            )
    except RuntimeError:
        return asyncio.run(run_lifeos_pipeline(user_id, user_input))
