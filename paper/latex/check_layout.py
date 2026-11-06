"""Fail if a page lacks text it must show, if a section that must open a page does not, if a main-text paragraph
ends on a short hanging line, or if an appendix heading is stranded at a page foot; list the appendix's hanging lines.

    just check-layout [FILE]   (from paper/)
"""

import logging
import re
import sys
from collections import Counter

import pdfplumber
from pypdf import PdfReader

ON_PAGE = {
    1: ("Homepage: https://m2b3.github.io/CanViT/", "rather than tabula rasa."),
    6: ("reducing storage from 19 TiB to zero.",),
    7: ("own weights (i.e., LP-FT",),
    8: ("RTX 4090 GPU (24 GB).",),
}
"""Text each 1-based page must show: on page 1, the project page's link and the end of the Introduction's second
paragraph; on page 6, the end of the Teacher paragraph of the distillation subsection; on page 7, the end of the
Experiments' "Task-specific probing" paragraph; on page 8, the end of its "Learned policies" paragraph."""
OPENS_PAGE = {"Results": 9, "Conclusion": 10}
"""Sections that must begin their page, with only floats (figures, tables and their captions) above their heading: title,
as the PDF outline gives it, and 1-based page."""
CAPTION = re.compile(r"(Figure|Table)\d+:")
"""How a caption's first line starts, spaces removed."""

HANGING = 0.2
"""A paragraph's last line is hanging when it is narrower than this fraction of the paragraph's full lines."""
BODY_SIZE = 10.0  # pt: body text and captions
NEW_PARAGRAPH = 15.0  # pt between baselines: 12 pt within a paragraph, 17.5 pt across \parskip
COLUMN_GAP = 8.0  # pt: wider than any word space, narrower than the gap beside a wrapped table or figure


def outline(reader: PdfReader) -> list[tuple[str, int, int]]:
    """(title, depth, first page) of every outline entry, in order."""
    entries: list[tuple[str, int, int]] = []

    def walk(items: list, depth: int) -> None:
        for item in items:
            if isinstance(item, list):
                walk(item, depth + 1)
            else:
                entries.append((item.title, depth, reader.get_destination_page_number(item)))

    walk(reader.outline, 0)
    return entries


def hanging_lines(pdf: pdfplumber.PDF, pages: range, body_font: str) -> list[str]:
    """Paragraphs of body text on these pages whose last line is hanging, from 10 pt lines in the body font.

    A line belongs to the paragraph above it when its baseline is under NEW_PARAGRAPH below; headings, equations and
    other fonts end a paragraph. The bibliography, from its heading on, is left out. A single line that opens a page is
    the end of a paragraph carried over from the previous page, and is measured against the page's widest body line.
    """
    found = []
    for index in pages:
        page = pdf.pages[index]
        heading = [w["bottom"] for w in page.extract_words(extra_attrs=["size"])
                   if w["text"] == "References" and w["size"] > BODY_SIZE + 0.5]
        lines = body_lines(page, body_font, above=min(heading, default=float("inf")))
        page_width = max((line["width"] for line in lines if line["body"]), default=0.0)
        first_body = next((line for line in lines if line["body"]), None)
        paragraph: list[dict] = []
        for line in [*lines, None]:
            if line and line["body"] and paragraph and line["bottom"] - paragraph[-1]["bottom"] < NEW_PARAGRAPH:
                paragraph.append(line)
                continue
            carried = len(paragraph) == 1 and paragraph[0] is first_body and index != pages.start
            if len(paragraph) >= 2 or carried:
                full = max(p["width"] for p in paragraph[:-1]) if len(paragraph) >= 2 else page_width
                if paragraph[-1]["width"] < HANGING * full:
                    found.append(f"page {index + 1}: …{paragraph[-1]['text'][-40:]}")
            paragraph = [line] if line and line["body"] else []
    return found


MIN_LINES_UNDER_HEADING = 3
"""An appendix heading needs at least this many lines of text below it on its page; fewer strand it at the page foot."""


def stranded_headings(reader: PdfReader, pdf: pdfplumber.PDF, pages: range) -> list[str]:
    """Appendix headings with fewer than MIN_LINES_UNDER_HEADING lines of text between them and the page's foot, from
    the outline's destination heights."""
    found = []

    def walk(items: list) -> None:
        for item in items:
            if isinstance(item, list):
                walk(item)
                continue
            index = reader.get_destination_page_number(item)
            top = item.get("/Top")
            if index not in pages or top is None:
                continue
            page = pdf.pages[index]
            heading = page.height - float(top)  # pdfplumber measures from the top
            # Every text line below the destination down to the page number, in any size and font (code, captions and
            # math count); the first one is the heading itself.
            baselines = sorted({round(c["bottom"]) for c in page.chars
                                if c["size"] >= 6 and c["text"].strip() and heading < c["bottom"] < page.height - 60})
            if len(baselines) - 1 < MIN_LINES_UNDER_HEADING:
                found.append(f"page {index + 1}: {item.title!r} has {max(len(baselines) - 1, 0)} lines below it")

    walk(reader.outline)
    return found


def body_lines(page, body_font: str, below: float = 0.0, above: float = float("inf")) -> list[dict]:
    """The page's lines of 10 pt text in the body font between two heights, each its leftmost column only."""
    lines: dict[int, list[dict]] = {}
    for c in page.chars:
        if abs(c["size"] - BODY_SIZE) < 0.3 and c["text"].strip() and below < c["bottom"] < above:
            lines.setdefault(round(c["bottom"]), []).append(c)
    out = []
    for bottom, chars in sorted(lines.items()):
        chars.sort(key=lambda c: c["x0"])
        # Beside a wrapped table the text and the table share baselines: keep the text, the leftmost column, finding the
        # gap between them among all the line's characters (citations and math included).
        top = min(c["top"] for c in chars)
        band = sorted((c for c in page.chars if c["text"].strip() and top <= (c["top"] + c["bottom"]) / 2 <= bottom
                       and c["x0"] >= chars[0]["x0"]), key=lambda c: c["x0"])
        gap = next((band[i]["x0"] for i in range(1, len(band)) if band[i]["x0"] - band[i - 1]["x1"] > COLUMN_GAP),
                   float("inf"))
        chars = [c for c in chars if c["x0"] < gap]
        text = "".join(c["text"] for c in chars)
        # A caption's label line is not body prose, even when its label is set in the body font.
        in_body = Counter(c["fontname"] for c in chars)[body_font] > len(chars) / 2 and not CAPTION.match(text)
        out.append(dict(bottom=bottom, width=chars[-1]["x1"] - chars[0]["x0"], text=text, body=in_body))
    return out


def prose_above(page, title: str, body_font: str) -> str:
    """Body text above the section's heading that is not part of a caption ('' when none); the heading not found on
    the page counts as text above it."""
    words = page.extract_words(extra_attrs=["size", "fontname"])
    heading = next((w["top"] for w in words if w["text"] == title
                    and (w["size"] > BODY_SIZE + 0.5 or "Medi" in w["fontname"] or "Bold" in w["fontname"])), None)
    if heading is None:
        return f"no {title} heading on the page"
    in_caption, prose = False, []
    previous = None
    for line in body_lines(page, body_font, above=heading):
        starts_block = previous is None or line["bottom"] - previous >= NEW_PARAGRAPH
        if starts_block:
            in_caption = bool(CAPTION.match(line["text"]))
        if not in_caption and line["body"]:
            prose.append(line["text"])
        previous = line["bottom"]
    return " ".join(prose)[-60:]


def main(pdf_path: str) -> None:
    reader = PdfReader(pdf_path)
    entries = outline(reader)
    failures = []
    for page, texts in ON_PAGE.items():
        # Compared without whitespace: text extraction splits words where colour or font changes.
        shown = "".join(reader.pages[page - 1].extract_text().split())
        failures += [f"page {page} lacks {text!r}" for text in texts if "".join(text.split()) not in shown]
    start = {title: page for title, _, page in entries}
    for title, page in OPENS_PAGE.items():
        if start[title] + 1 != page:
            failures.append(f"{title!r} starts on page {start[title] + 1}, not {page}")
    checklist = next(i for i, page in enumerate(reader.pages) if "NeurIPS Paper Checklist" in (page.extract_text() or ""))
    with pdfplumber.open(pdf_path) as pdf:
        body_font = Counter(
            c["fontname"] for page in pdf.pages[: start["References"]] for c in page.chars if abs(c["size"] - BODY_SIZE) < 0.3
        ).most_common(1)[0][0]
        main_text = hanging_lines(pdf, range(start["References"] + 1), body_font)
        appendix = hanging_lines(pdf, range(start["Appendix"], checklist), body_font)
        failures += [f"stranded heading, {h}" for h in stranded_headings(reader, pdf, range(start["Appendix"], checklist))]
        for title in OPENS_PAGE:
            prose = prose_above(pdf.pages[start[title]], title, body_font)
            if prose:
                failures.append(f"{title!r} does not open page {start[title] + 1}: {prose!r} above it")
    failures += [f"hanging paragraph end, {line}" for line in main_text]
    for line in appendix:
        print(f"warning: hanging paragraph end in the appendix, {line}")
    assert not failures, "; ".join(failures)
    print(f"Layout: {'; '.join(f'page {p} holds {t}' for p, t in ON_PAGE.items())}; "
          f"{', '.join(f'{t!r} opens page {p}' for t, p in OPENS_PAGE.items())}; "
          f"no main-text paragraph ends on a line under {HANGING:.0%} of its width; "
          f"every appendix heading has at least {MIN_LINES_UNDER_HEADING} lines below it on its page.")


if __name__ == "__main__":
    logging.getLogger("pypdf").setLevel(logging.ERROR)  # font-encoding notices from text extraction
    main(sys.argv[1])
