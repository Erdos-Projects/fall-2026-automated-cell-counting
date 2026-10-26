"""Render the English check-in Markdown as a one-page PDF."""
import html
import re
from pathlib import Path

from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]


def inline(text):
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = html.escape(text)
    return re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)


def main():
    source = ROOT/"docs/weekly_checkin.md"
    output = source.with_suffix(".pdf")
    body = ParagraphStyle("Body", fontName="Helvetica", fontSize=10.1,
                          leading=12.9, spaceAfter=6, textColor=colors.black)
    title = ParagraphStyle("Title", parent=body, fontName="Helvetica-Bold", fontSize=16,
                           leading=19, spaceAfter=7)
    subtitle = ParagraphStyle("Subtitle", parent=body, fontSize=10.1, spaceAfter=10)
    cell = ParagraphStyle("Cell", parent=body, fontSize=9.7, leading=12, spaceAfter=0)
    story = []
    blocks = source.read_text().strip().split("\n\n")
    for i, block in enumerate(blocks):
        if block.startswith("# "):
            story.append(Paragraph(inline(block[2:]), title))
        elif block.startswith("|"):
            rows = []
            for line in block.splitlines():
                values = [x.strip() for x in line.strip().strip("|").split("|")]
                if all(re.fullmatch(r":?-+:?", v) for v in values):
                    continue
                rows.append([Paragraph(inline(v), cell) for v in values])
            table = Table(rows, colWidths=[300, 113, 113], hAlign="LEFT")
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF3")),
                ("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#D9D9D9")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            table.spaceAfter = 7
            story.append(table)
        else:
            story.append(Paragraph(inline(block.replace("\n", " ")), subtitle if i == 1 else body))
    doc = SimpleDocTemplate(str(output), pagesize=letter, leftMargin=43, rightMargin=43,
                            topMargin=35, bottomMargin=35,
                            title="Week 5 cell counting check-in", author="Ruiping Huang")
    doc.build(story)
    reader = PdfReader(output)
    assert len(reader.pages) == 1, f"Expected one page; found {len(reader.pages)}"
    text = reader.pages[0].extract_text()
    for expected in ["1.173", "1.587", "Team context", "Evidence"]:
        assert expected in text, expected
    print(f"One-page PDF verified: {output}")


if __name__ == "__main__":
    main()
