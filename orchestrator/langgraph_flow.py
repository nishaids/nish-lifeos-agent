"""
LifeOS Agent - LangGraph Orchestration Flow
==============================================
State machine orchestrator that connects all 6 CrewAI agents
in sequence: Memory → Research → Goals → Strategy → Planning → Report → PDF.
Uses TypedDict state and automatic retry with fallback.
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
    user_profile: dict
    user_context: str
    goals_data: str
    research_data: str
    strategy_data: str
    plan_data: str
    report_content: str
    pdf_path: str
    status: str
    error: str


# ═══════════════════════════════════════════
# NODE FUNCTIONS
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
            "status": "report_partial",
        }


def pdf_node(state: LifeOSState) -> dict:
    """
    Node 7: Generate PDF from the report content.
    """
    report_content = state.get("report_content", "")
    user_input = state["user_input"]
    user_profile = state.get("user_profile", {})
    user_name = user_profile.get("name", "User") or "User"
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
    report_content = state.get("report_content", "")
    logger.info(f"[Save Memory Node] Saving interaction for user {user_id}")

    try:
        from agents.memory_agent import save_interaction

        # Save a summary of the interaction
        summary = report_content[:500] if report_content else "No report generated"
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
# BUILD THE LANGGRAPH WORKFLOW
# ═══════════════════════════════════════════


def build_lifeos_graph() -> StateGraph:
    """
    Build and compile the LifeOS LangGraph workflow.

    Flow: START → memory → research → goals → strategy → planning → report → pdf → save_memory → END

    Returns:
        Compiled LangGraph workflow.
    """
    # Create the state graph
    workflow = StateGraph(LifeOSState)

    # Add all nodes
    workflow.add_node("memory", memory_node)
    workflow.add_node("research", research_node)
    workflow.add_node("goals", goals_node)
    workflow.add_node("strategy", strategy_node)
    workflow.add_node("planning", planning_node)
    workflow.add_node("report", report_node)
    workflow.add_node("pdf", pdf_node)
    workflow.add_node("save_memory", save_memory_node)

    # Define the flow (linear pipeline)
    workflow.set_entry_point("memory")
    workflow.add_edge("memory", "research")
    workflow.add_edge("research", "goals")
    workflow.add_edge("goals", "strategy")
    workflow.add_edge("strategy", "planning")
    workflow.add_edge("planning", "report")
    workflow.add_edge("report", "pdf")
    workflow.add_edge("pdf", "save_memory")
    workflow.add_edge("save_memory", END)

    # Compile the graph
    compiled = workflow.compile()
    logger.info("LifeOS LangGraph workflow compiled successfully")
    return compiled


# ═══════════════════════════════════════════
# RUN THE PIPELINE
# ═══════════════════════════════════════════


async def run_lifeos_pipeline(
    user_id: str,
    user_input: str,
    status_callback=None,
) -> LifeOSState:
    """
    Run the complete LifeOS pipeline for a user query.

    Args:
        user_id: Unique user identifier.
        user_input: The user's message/query.
        status_callback: Optional async callback for status updates.
            Called with (status_emoji, status_message) at each stage.

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
        "user_profile": {},
        "user_context": "",
        "goals_data": "",
        "research_data": "",
        "strategy_data": "",
        "plan_data": "",
        "report_content": "",
        "pdf_path": "",
        "status": "starting",
        "error": "",
    }

    # Status messages for each stage
    status_messages = {
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
