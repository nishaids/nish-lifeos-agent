"""
LifeOS Agent - Search Tools
=============================
Tavily web search, Wikipedia, ArXiv research tools.
Includes smart_search with automatic fallback chain.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════
# TAVILY SEARCH
# ═══════════════════════════════════════════


def tavily_search(query: str, max_results: int = 5) -> str:
    """
    Search the web using Tavily API.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return.

    Returns:
        Formatted search results as a string.
    """
    try:
        from tavily import TavilyClient
        from config.models import TAVILY_CONFIG

        api_key = TAVILY_CONFIG["api_key"]
        if not api_key:
            logger.warning("Tavily API key not found, skipping web search")
            return "Web search unavailable — no API key configured."

        client = TavilyClient(api_key=api_key)
        response = client.search(query=query, max_results=max_results)

        results = []
        for idx, item in enumerate(response.get("results", []), 1):
            title = item.get("title", "No Title")
            url = item.get("url", "")
            content = item.get("content", "No content available")
            results.append(
                f"**Result {idx}: {title}**\n"
                f"URL: {url}\n"
                f"Summary: {content}\n"
            )

        if not results:
            return f"No web search results found for: {query}"

        output = f"🔍 **Web Search Results for: '{query}'**\n\n" + "\n---\n".join(
            results
        )
        logger.info(f"Tavily search returned {len(results)} results for: {query}")
        return output

    except Exception as e:
        logger.error(f"Tavily search failed: {e}")
        return f"Web search failed: {str(e)}"


# ═══════════════════════════════════════════
# WIKIPEDIA SEARCH
# ═══════════════════════════════════════════


def wikipedia_search(query: str, sentences: int = 5) -> str:
    """
    Search Wikipedia for information on a topic.

    Args:
        query: The search query string.
        sentences: Number of sentences to return in the summary.

    Returns:
        Wikipedia summary as a string.
    """
    try:
        import wikipedia

        wikipedia.set_lang("en")

        # Search for matching pages
        search_results = wikipedia.search(query, results=3)
        if not search_results:
            return f"No Wikipedia articles found for: {query}"

        # Try to get summary from the best matching page
        for result_title in search_results:
            try:
                summary = wikipedia.summary(result_title, sentences=sentences)
                page = wikipedia.page(result_title, auto_suggest=False)
                output = (
                    f"📚 **Wikipedia: {page.title}**\n\n"
                    f"{summary}\n\n"
                    f"Source: {page.url}"
                )
                logger.info(f"Wikipedia search successful for: {result_title}")
                return output
            except wikipedia.exceptions.DisambiguationError as e:
                # Try the first option from disambiguation
                if e.options:
                    try:
                        summary = wikipedia.summary(e.options[0], sentences=sentences)
                        output = (
                            f"📚 **Wikipedia: {e.options[0]}**\n\n"
                            f"{summary}"
                        )
                        return output
                    except Exception:
                        continue
            except wikipedia.exceptions.PageError:
                continue

        return f"Could not find a clear Wikipedia article for: {query}"

    except Exception as e:
        logger.error(f"Wikipedia search failed: {e}")
        return f"Wikipedia search failed: {str(e)}"


# ═══════════════════════════════════════════
# ARXIV SEARCH
# ═══════════════════════════════════════════


def arxiv_search(query: str, max_results: int = 3) -> str:
    """
    Search ArXiv for research papers on a topic.

    Args:
        query: The search query string.
        max_results: Maximum number of papers to return.

    Returns:
        Formatted research paper results as a string.
    """
    try:
        import arxiv

        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        results = []
        for paper in client.results(search):
            title = paper.title
            authors = ", ".join([a.name for a in paper.authors[:3]])
            if len(paper.authors) > 3:
                authors += " et al."
            summary = paper.summary[:300] + "..." if len(paper.summary) > 300 else paper.summary
            published = paper.published.strftime("%Y-%m-%d") if paper.published else "N/A"
            pdf_url = paper.pdf_url or "N/A"

            results.append(
                f"**📄 {title}**\n"
                f"Authors: {authors}\n"
                f"Published: {published}\n"
                f"Abstract: {summary}\n"
                f"PDF: {pdf_url}\n"
            )

        if not results:
            return f"No ArXiv papers found for: {query}"

        output = f"🔬 **ArXiv Research Papers for: '{query}'**\n\n" + "\n---\n".join(
            results
        )
        logger.info(f"ArXiv search returned {len(results)} papers for: {query}")
        return output

    except Exception as e:
        logger.error(f"ArXiv search failed: {e}")
        return f"ArXiv search failed: {str(e)}"


# ═══════════════════════════════════════════
# SMART SEARCH (WITH FALLBACK CHAIN)
# ═══════════════════════════════════════════


def smart_search(query: str) -> str:
    """
    Smart search that tries Tavily first, then falls back to Wikipedia,
    then ArXiv for academic topics.

    Args:
        query: The search query string.

    Returns:
        Combined search results from the best available source.
    """
    results_parts = []

    # Try Tavily web search first
    try:
        tavily_result = tavily_search(query, max_results=3)
        if "failed" not in tavily_result.lower() and "unavailable" not in tavily_result.lower():
            results_parts.append(tavily_result)
            logger.info("Smart search: Tavily succeeded")
    except Exception as e:
        logger.warning(f"Smart search: Tavily failed — {e}")

    # Always try Wikipedia as supplementary source
    try:
        wiki_result = wikipedia_search(query, sentences=3)
        if "failed" not in wiki_result.lower() and "could not find" not in wiki_result.lower():
            results_parts.append(wiki_result)
            logger.info("Smart search: Wikipedia succeeded")
    except Exception as e:
        logger.warning(f"Smart search: Wikipedia failed — {e}")

    # Try ArXiv for research/academic topics
    academic_keywords = [
        "research", "study", "paper", "algorithm", "machine learning",
        "AI", "science", "theory", "analysis", "technology", "engineering",
        "data", "model", "framework", "method",
    ]
    is_academic = any(kw.lower() in query.lower() for kw in academic_keywords)

    if is_academic:
        try:
            arxiv_result = arxiv_search(query, max_results=2)
            if "failed" not in arxiv_result.lower() and "no arxiv" not in arxiv_result.lower():
                results_parts.append(arxiv_result)
                logger.info("Smart search: ArXiv succeeded")
        except Exception as e:
            logger.warning(f"Smart search: ArXiv failed — {e}")

    if not results_parts:
        return (
            f"⚠️ No search results found for: '{query}'. "
            f"All search providers were unavailable. "
            f"Please check your API keys and try again."
        )

    combined = "\n\n" + "═" * 50 + "\n\n".join(results_parts)
    return combined
