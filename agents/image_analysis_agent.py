"""
LifeOS Agent - Image Analysis Agent
======================================
Agent 9: Visual Intelligence Analyzer
Role: Analyze images using Gemini Vision API.
Handles: OCR, food analysis, code screenshots, documents, charts, scenes.
Model: Gemini (Vision capable)
"""

import os
import base64
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════
# IMAGE TYPE DETECTION
# ═══════════════════════════════════════════

IMAGE_ANALYSIS_PROMPTS = {
    "general": (
        "Analyze this image comprehensively. Describe:\n"
        "1. What type of image this is (photo, document, screenshot, chart, etc.)\n"
        "2. What you see in detail\n"
        "3. Any text visible in the image\n"
        "4. Key observations and insights\n"
        "5. Any actionable information\n\n"
        "Be thorough but concise. Use emojis naturally."
    ),
    "document": (
        "This appears to be a document or text image. Please:\n"
        "1. Extract ALL text from the image accurately\n"
        "2. Preserve the formatting and structure\n"
        "3. Identify the type of document (receipt, notes, form, ID, certificate, etc.)\n"
        "4. Highlight key information (dates, amounts, names, etc.)\n"
        "5. Summarize the content"
    ),
    "code": (
        "This appears to be a code screenshot. Please:\n"
        "1. Extract the code from the image\n"
        "2. Identify the programming language\n"
        "3. Explain what the code does\n"
        "4. Identify any bugs or issues\n"
        "5. Suggest improvements or fixes\n"
        "Format the extracted code in a proper code block."
    ),
    "food": (
        "This appears to be a food image. Please:\n"
        "1. Identify the food items visible\n"
        "2. Estimate approximate calories for each item\n"
        "3. Provide a nutrition breakdown (protein, carbs, fats)\n"
        "4. Rate the healthiness (1-10)\n"
        "5. Suggest healthier alternatives if applicable\n"
        "Use 🍽️ food emojis naturally."
    ),
    "chart": (
        "This appears to be a chart or graph. Please:\n"
        "1. Identify the type of chart (bar, line, pie, etc.)\n"
        "2. Describe the data being presented\n"
        "3. Identify key trends and patterns\n"
        "4. Extract specific data points if visible\n"
        "5. Provide insights and interpretation"
    ),
}


# ═══════════════════════════════════════════
# GEMINI VISION API
# ═══════════════════════════════════════════


def analyze_image(image_data: bytes, user_input: str = "", mime_type: str = "image/jpeg") -> str:
    """
    Analyze an image using Gemini Vision API.

    Args:
        image_data: Raw image bytes.
        user_input: Optional user message accompanying the image.
        mime_type: MIME type of the image.

    Returns:
        Analysis result string.
    """
    try:
        import google.generativeai as genai
        from config.models import GEMINI_CONFIG

        api_key = GEMINI_CONFIG.get("api_key", "")
        if not api_key:
            return "⚠️ Gemini API key not configured. Cannot analyze images."

        genai.configure(api_key=api_key)

        # Use a vision-capable Gemini model
        model = genai.GenerativeModel("gemini-2.5-flash-preview-04-17")

        # Build the prompt based on context
        if user_input:
            prompt = (
                f"The user sent this image with the message: '{user_input}'\n\n"
                f"Analyze the image in the context of their message. "
                f"Provide a detailed, helpful response. Use emojis naturally. "
                f"Start by identifying what type of image this is."
            )
        else:
            prompt = IMAGE_ANALYSIS_PROMPTS["general"]

        # Create image part
        image_part = {
            "mime_type": mime_type,
            "data": base64.b64encode(image_data).decode("utf-8"),
        }

        # Generate analysis
        response = model.generate_content(
            [prompt, image_part],
            generation_config={
                "temperature": 0.4,
                "max_output_tokens": 2048,
            },
        )

        if response and response.text:
            result = response.text
            logger.info("Image analysis completed successfully")
            return f"🖼️ **Image Analysis**\n\n{result}"
        else:
            return "⚠️ Could not analyze the image. Please try again with a clearer image."

    except Exception as e:
        logger.error(f"Image analysis failed: {e}")
        return f"⚠️ Image analysis failed: {str(e)}\n\nPlease make sure your Gemini API key supports vision capabilities."


def analyze_image_from_file(file_path: str, user_input: str = "") -> str:
    """
    Analyze an image from a file path.

    Args:
        file_path: Path to the image file.
        user_input: Optional user message.

    Returns:
        Analysis result string.
    """
    try:
        with open(file_path, "rb") as f:
            image_data = f.read()

        # Detect mime type
        ext = os.path.splitext(file_path)[1].lower()
        mime_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
        }
        mime_type = mime_map.get(ext, "image/jpeg")

        return analyze_image(image_data, user_input, mime_type)

    except Exception as e:
        logger.error(f"Failed to read image file: {e}")
        return f"⚠️ Could not read image file: {str(e)}"


def run_image_agent(user_id: str, image_data: bytes, user_input: str = "",
                    mime_type: str = "image/jpeg") -> str:
    """
    Run the Image Analysis Agent.

    Args:
        user_id: Unique user identifier.
        image_data: Raw image bytes.
        user_input: Optional caption/message with the image.
        mime_type: MIME type of the image.

    Returns:
        Image analysis result string.
    """
    try:
        # Perform analysis
        result = analyze_image(image_data, user_input, mime_type)

        # Save to memory
        try:
            from memory.chroma_store import get_chroma_store

            chroma = get_chroma_store()
            summary = result[:300] if result else "Image analyzed"
            chroma.store(
                user_id,
                f"Image analysis: {summary}. User context: {user_input[:100]}",
                metadata={"type": "image_analysis"},
            )
        except Exception as e:
            logger.warning(f"Failed to save image analysis to memory: {e}")

        logger.info(f"Image agent completed for user {user_id}")
        return result

    except Exception as e:
        logger.error(f"Image agent failed: {e}")
        return "⚠️ Image analysis failed. Please try sending the image again."
