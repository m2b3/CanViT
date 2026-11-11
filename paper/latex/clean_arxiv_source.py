"""Copy an arXiv source bundle without its comments and without unprinted BibTeX material.

    uv run --no-project python -I latex/clean_arxiv_source.py BUNDLE_DIR OUTPUT_DIR

arXiv publishes the source it is given. BUNDLE_DIR holds the files `just build-arxiv` assembles; OUTPUT_DIR receives
the same files with

- every comment removed from the `.tex` files, by TeX's own rules, so the typeset result cannot change: a line that
  is only a comment is dropped; a comment after code is cut back to its `%`, which keeps the line end swallowed; a
  bare `%` stays; listings are left as they are;
- `references.bib` reduced to its entries, without the text between them and without the fields BibTeX styles do
  not print (DROPPED_FIELDS, reference-manager exports).

Style files and figures are copied unchanged. `just build-arxiv` compiles the result. The script stops on a construct
it does not handle rather than guess.
"""

import argparse
import re
import shutil
from collections import Counter
from pathlib import Path

LISTING = re.compile(r"\\(begin|end)\{(verbatim\*?|Verbatim|lstlisting|minted)\}")
UNHANDLED = re.compile(r"\\verb\b|\\lstinline\b|\\catcode|\\(?:url|href|nolinkurl|path)\{[^}]*%")
DROPPED_FIELDS = frozenset({
    "abstract", "keywords", "urldate", "langid", "copyright", "shorttitle", "timestamp", "biburl", "bibsource",
    "file", "annote", "annotation",
})


def comment_start(line: str) -> int | None:
    """Index of the `%` that opens a comment: the first one after an even number of backslashes."""
    index = line.find("%")
    while index != -1:
        backslashes = index - len(line[:index].rstrip("\\"))
        if backslashes % 2 == 0:
            return index
        index = line.find("%", index + 1)
    return None


def without_comments(text: str, name: str) -> tuple[str, Counter]:
    kept, counts, listing = [], Counter(), None
    for number, line in enumerate(text.split("\n"), start=1):
        boundary = LISTING.search(line)
        if listing is not None:
            kept.append(line)
            if boundary and boundary.group(1) == "end" and boundary.group(2) == listing:
                listing = None
            continue
        if boundary and boundary.group(1) == "begin":
            assert comment_start(line) is None, f"{name}:{number}: a comment on a listing's opening line"
            listing = boundary.group(2)
            kept.append(line)
            continue
        assert not UNHANDLED.search(line), f"{name}:{number}: a construct this script does not handle: {line.strip()}"
        start = comment_start(line)
        if start is None or not line[start + 1:].strip():
            kept.append(line)
        elif not line[:start].strip():
            counts["comment lines dropped"] += 1
        else:
            kept.append(line[:start] + "%")
            counts["trailing comments cut"] += 1
    assert listing is None, f"{name}: a listing is not closed"
    return "\n".join(kept), counts


def closing_brace(text: str, opening: int) -> int:
    depth = 0
    for index in range(opening, len(text)):
        depth += {"{": 1, "}": -1}.get(text[index], 0)
        if depth == 0:
            return index
    raise ValueError(f"unbalanced braces from offset {opening}")


def top_level_parts(body: str) -> list[str]:
    """An entry's body split at the commas that separate its key and fields."""
    parts, depth, quoted, start = [], 0, False, 0
    for index, char in enumerate(body):
        if char == '"' and depth == 0:
            quoted = not quoted
        elif char in "{}":
            depth += 1 if char == "{" else -1
        elif char == "," and depth == 0 and not quoted:
            parts.append(body[start:index])
            start = index + 1
    parts.append(body[start:])
    return parts


def entries_only(text: str) -> tuple[str, Counter]:
    entries, counts, position = [], Counter(), 0
    for match in re.finditer(r"^[ \t]*@(\w+)\s*\{", text, re.MULTILINE):  # BibTeX accepts an indented entry
        assert match.start() >= position, f"an entry opens inside another near offset {match.start()}"
        kind = match.group(1).lower()
        assert kind not in {"string", "preamble", "comment"}, f"@{kind} is not handled"
        end = closing_brace(text, match.end() - 1)
        between = text[position:match.start()].split("\n")
        counts["lines between entries dropped"] += sum(bool(line.strip()) for line in between)
        key, *fields = top_level_parts(text[match.end():end])
        lines = [f"@{match.group(1)}{{{key.strip()},"]
        for field in fields:
            if not field.strip():
                continue
            name = field.split("=", 1)[0].strip().lower()
            assert re.fullmatch(r"[a-z_]+", name), f"{key.strip()}: not a field: {field.strip()[:60]}"
            if name in DROPPED_FIELDS:
                counts[f"{name} fields dropped"] += 1
            else:
                lines.append(f"  {field.strip()},")
        entries.append("\n".join(lines) + "\n}\n")
        position = end + 1
    counts["lines between entries dropped"] += sum(bool(line.strip()) for line in text[position:].split("\n"))
    counts["entries kept"] = len(entries)
    return "\n".join(entries), counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    assert args.bundle.is_dir(), args.bundle
    assert not args.output.exists(), f"{args.output} exists; choose a new directory"

    for source in sorted(path for path in args.bundle.rglob("*") if path.is_file()):
        relative = source.relative_to(args.bundle)
        if relative.name.startswith("."):
            continue  # Finder's files, never part of a bundle
        target = args.output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix == ".tex":
            cleaned, counts = without_comments(source.read_text(), str(relative))
        elif source.suffix == ".bib":
            cleaned, counts = entries_only(source.read_text())
        else:
            shutil.copyfile(source, target)
            print(f"{relative}: copied unchanged")
            continue
        target.write_text(cleaned)
        summary = ", ".join(f"{count} {what}" for what, count in sorted(counts.items())) or "nothing to remove"
        print(f"{relative}: {summary}")


if __name__ == "__main__":
    main()
