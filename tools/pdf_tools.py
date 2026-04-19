"""
LifeOS Agent - PDF Tools
==========================
Professional PDF report generation using ReportLab.
Creates styled LifeOS Intelligence Reports with dark blue headers,
sections, and professional formatting.
"""

import os
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════
# PDF CREATION
# ═══════════════════════════════════════════


def create_pdf(
    content: str,
    topic: str,
    user_name: str = "User",
    output_dir: Optional[str] = None,
) -> str:
    """
    Create a professionally styled PDF report.

    Args:
        content: The full report content text.
        topic: The topic/subject of the report.
        user_name: Name of the user requesting the report.
        output_dir: Directory to save the PDF. Defaults to /tmp/.

    Returns:
        Absolute path to the generated PDF file.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.colors import HexColor
        from reportlab.lib.units import inch, cm
        from reportlab.platypus import (
            SimpleDocTemplate,
            Paragraph,
            Spacer,
            Table,
            TableStyle,
            PageBreak,
        )
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

        # Determine output path
        if output_dir is None:
            output_dir = "/tmp" if os.path.exists("/tmp") else os.path.join(os.getcwd(), "reports")

        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_topic = "".join(c if c.isalnum() or c in " _-" else "_" for c in topic)[:50]
        filename = f"LifeOS_Report_{safe_topic}_{timestamp}.pdf"
        filepath = os.path.join(output_dir, filename)

        # Create the document
        doc = SimpleDocTemplate(
            filepath,
            pagesize=A4,
            rightMargin=1 * inch,
            leftMargin=1 * inch,
            topMargin=0.75 * inch,
            bottomMargin=0.75 * inch,
        )

        # ═══════════════════════════════════════
        # CUSTOM STYLES
        # ═══════════════════════════════════════

        styles = getSampleStyleSheet()

        # Dark blue title style
        title_style = ParagraphStyle(
            "LifeOSTitle",
            parent=styles["Title"],
            fontSize=24,
            textColor=HexColor("#FFFFFF"),
            alignment=TA_CENTER,
            spaceAfter=6,
            fontName="Helvetica-Bold",
        )

        # Subtitle style
        subtitle_style = ParagraphStyle(
            "LifeOSSubtitle",
            parent=styles["Normal"],
            fontSize=12,
            textColor=HexColor("#B0C4DE"),
            alignment=TA_CENTER,
            spaceAfter=4,
            fontName="Helvetica",
        )

        # Section header style
        section_style = ParagraphStyle(
            "LifeOSSection",
            parent=styles["Heading1"],
            fontSize=16,
            textColor=HexColor("#1B3A5C"),
            spaceBefore=20,
            spaceAfter=10,
            fontName="Helvetica-Bold",
            borderWidth=0,
            borderPadding=0,
        )

        # Body text style
        body_style = ParagraphStyle(
            "LifeOSBody",
            parent=styles["Normal"],
            fontSize=10,
            textColor=HexColor("#2C3E50"),
            alignment=TA_JUSTIFY,
            spaceAfter=8,
            fontName="Helvetica",
            leading=14,
        )

        # Quote/highlight style
        highlight_style = ParagraphStyle(
            "LifeOSHighlight",
            parent=styles["Normal"],
            fontSize=10,
            textColor=HexColor("#1B3A5C"),
            alignment=TA_LEFT,
            spaceAfter=8,
            fontName="Helvetica-Oblique",
            leftIndent=20,
            borderWidth=1,
            borderColor=HexColor("#3498DB"),
            borderPadding=8,
        )

        # Footer style
        footer_style = ParagraphStyle(
            "LifeOSFooter",
            parent=styles["Normal"],
            fontSize=8,
            textColor=HexColor("#95A5A6"),
            alignment=TA_CENTER,
            fontName="Helvetica",
        )

        # ═══════════════════════════════════════
        # BUILD DOCUMENT ELEMENTS
        # ═══════════════════════════════════════

        elements = []

        # ─── DARK BLUE HEADER BLOCK ───
        header_data = [[""]]
        header_table = Table(header_data, colWidths=[doc.width], rowHeights=[120])
        header_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), HexColor("#0D1B2A")),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 20),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 20),
                    ("LEFTPADDING", (0, 0), (-1, -1), 20),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 20),
                ]
            )
        )
        elements.append(header_table)

        # Title inside header (overlay simulation via separate elements)
        elements.append(Spacer(1, -100))

        # LifeOS Logo Text
        logo_text = Paragraph("🧠 LifeOS Agent", title_style)
        elements.append(logo_text)

        # Report topic
        topic_text = Paragraph(f"Intelligence Report: {_escape_xml(topic)}", subtitle_style)
        elements.append(topic_text)

        # Date and user
        date_str = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        meta_text = Paragraph(
            f"Prepared for: {_escape_xml(user_name)} | Date: {date_str}",
            subtitle_style,
        )
        elements.append(meta_text)

        elements.append(Spacer(1, 20))

        # ─── DIVIDER ───
        divider_data = [[""]]
        divider_table = Table(divider_data, colWidths=[doc.width], rowHeights=[3])
        divider_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), HexColor("#3498DB")),
                    ("LINEBELOW", (0, 0), (-1, -1), 0, HexColor("#3498DB")),
                ]
            )
        )
        elements.append(divider_table)
        elements.append(Spacer(1, 15))

        # ─── PARSE AND ADD CONTENT SECTIONS ───
        sections = _parse_content_sections(content)

        for section_title, section_body in sections:
            # Section header
            elements.append(
                Paragraph(f"▎ {_escape_xml(section_title)}", section_style)
            )

            # Section divider line
            line_data = [[""]]
            line_table = Table(line_data, colWidths=[doc.width], rowHeights=[1])
            line_table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, -1), HexColor("#E8F4FD")),
                    ]
                )
            )
            elements.append(line_table)
            elements.append(Spacer(1, 8))

            # Section body paragraphs
            paragraphs = section_body.strip().split("\n")
            for para in paragraphs:
                para = para.strip()
                if not para:
                    elements.append(Spacer(1, 4))
                    continue

                # Handle bullet points
                if para.startswith("- ") or para.startswith("• ") or para.startswith("* "):
                    bullet_text = para.lstrip("-•* ").strip()
                    elements.append(
                        Paragraph(
                            f"  •  {_escape_xml(bullet_text)}", body_style
                        )
                    )
                # Handle numbered items
                elif len(para) > 2 and para[0].isdigit() and para[1] in [".", ")"]:
                    elements.append(
                        Paragraph(f"  {_escape_xml(para)}", body_style)
                    )
                # Handle highlighted/quoted text
                elif para.startswith(">"):
                    quote_text = para.lstrip("> ").strip()
                    elements.append(
                        Paragraph(_escape_xml(quote_text), highlight_style)
                    )
                else:
                    elements.append(
                        Paragraph(_escape_xml(para), body_style)
                    )

            elements.append(Spacer(1, 10))

        # ─── FOOTER ───
        elements.append(Spacer(1, 30))
        divider_table2 = Table(divider_data, colWidths=[doc.width], rowHeights=[1])
        divider_table2.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), HexColor("#BDC3C7")),
                ]
            )
        )
        elements.append(divider_table2)
        elements.append(Spacer(1, 8))

        footer_text = Paragraph(
            f"Generated by LifeOS Agent — Your Personal Life Intelligence System | {date_str}",
            footer_style,
        )
        elements.append(footer_text)

        # ═══════════════════════════════════════
        # BUILD PDF WITH PAGE NUMBERS
        # ═══════════════════════════════════════

        doc.build(elements, onFirstPage=_add_page_number, onLaterPages=_add_page_number)

        logger.info(f"PDF report generated: {filepath}")
        return filepath

    except Exception as e:
        logger.error(f"PDF creation failed: {e}")
        raise RuntimeError(f"Failed to create PDF report: {str(e)}")


# ═══════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════


def _escape_xml(text: str) -> str:
    """Escape XML special characters for ReportLab Paragraph."""
    if not text:
        return ""
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    text = text.replace('"', "&quot;")
    text = text.replace("'", "&apos;")
    return text


def _parse_content_sections(content: str) -> list:
    """
    Parse report content into sections.
    Looks for markdown-style headers (## or **Header**) and splits accordingly.

    Returns:
        List of (title, body) tuples.
    """
    lines = content.strip().split("\n")
    sections = []
    current_title = "Executive Summary"
    current_body = []

    for line in lines:
        stripped = line.strip()

        # Check for markdown headers
        if stripped.startswith("## "):
            if current_body:
                sections.append((current_title, "\n".join(current_body)))
            current_title = stripped.lstrip("# ").strip()
            current_body = []
        elif stripped.startswith("# "):
            if current_body:
                sections.append((current_title, "\n".join(current_body)))
            current_title = stripped.lstrip("# ").strip()
            current_body = []
        elif stripped.startswith("**") and stripped.endswith("**") and len(stripped) > 4:
            if current_body:
                sections.append((current_title, "\n".join(current_body)))
            current_title = stripped.strip("*").strip()
            current_body = []
        else:
            current_body.append(line)

    # Add the last section
    if current_body:
        sections.append((current_title, "\n".join(current_body)))

    # If no sections were found, create a default one
    if not sections:
        sections = [("Report Content", content)]

    return sections


def _add_page_number(canvas, doc):
    """Add page number to the bottom of each page."""
    from reportlab.lib.colors import HexColor

    page_num = canvas.getPageNumber()
    text = f"LifeOS Agent — Page {page_num}"
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(HexColor("#95A5A6"))
    canvas.drawCentredString(doc.pagesize[0] / 2, 30, text)
    canvas.restoreState()
