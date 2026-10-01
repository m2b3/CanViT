# /// script
# requires-python = ">=3.12"
# dependencies = ["playwright>=1.63", "tyro"]
# ///
"""Skip through a talk the way a nervous speaker does, then check where it lands.

    uv run site/slides/stress.py --url http://127.0.0.1:8765/slides/2026-10-30-Montreal-AI-and-Neuroscience-MAIN/

Walks every fragment forward and back with --step-ms between moves, then, for each slide before its fragments and
after all of them, lands there after a burst of random moves and compares the slide's state (its shows-* classes and
the attributes that drive its components) with a second page that went there once and settled. Also reports anything
playing on a slide that is not shown, and every browser error (--ignore skips the ones expected, such as a missing
local model). Exits with status 1 on any mismatch or error.
"""

import asyncio
import random
import re
import sys
from dataclasses import dataclass

import tyro
from playwright.async_api import Page, async_playwright


@dataclass(frozen=True)
class Stress:
    url: str
    step_ms: int = 60
    """Time between moves, shorter than any transition."""
    burst: int = 6
    """Random moves before landing on each checked position."""
    settle_ms: int = 1500
    seed: int = 0
    ignore: str = ""
    """A regular expression for browser errors to leave out of the report."""


# The state a slide's CSS and components are a function of: its build classes and the attributes the deck sets.
STATE = """() => {
  const slide = deck.getCurrentSlide();
  const driven = ["t", "stage", "series", "readout", "y-range", "policies"];
  const attributes = [...slide.querySelectorAll("*")]
    .filter((el) => driven.some((name) => el.hasAttribute(name)))
    .map((el, i) => `${i}:${el.localName}:` + driven.filter((n) => el.hasAttribute(n)).map((n) => `${n}=${el.getAttribute(n)}`).join(","));
  const classes = [...slide.classList].filter((c) => c.startsWith("shows-")).sort();
  return { id: slide.id, indices: deck.getIndices(), classes, attributes };
}"""

# Animations playing on a slide that is not shown.
STRAY = """() => {
  const current = deck.getCurrentSlide();
  return [...document.querySelectorAll(".reveal .slides [data-play]")]
    .filter((el) => el.playing && !current.contains(el))
    .map((el) => `${el.closest("section").id}: ${el.localName}`);
}"""


async def open_deck(browser, url: str, errors: list[str], name: str) -> Page:
    page = await browser.new_page(viewport={"width": 1280, "height": 720})
    page.on("console", lambda m: m.type == "error" and errors.append(f"[{name}] console: {m.text}"))
    page.on("pageerror", lambda e: errors.append(f"[{name}] page error: {e}"))
    await page.goto(url)
    await page.wait_for_function("window.deck?.isReady()")
    return page


async def stress(args: Stress) -> int:
    rng = random.Random(args.seed)
    errors: list[str] = []
    problems: list[str] = []
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(channel="chrome")
        page = await open_deck(browser, args.url, errors, "stressed")
        reference = await open_deck(browser, args.url, errors, "reference")
        # Fragment counts per slide: reveal.js numbers each slide's fragments from 0.
        counts = await page.evaluate("""() => deck.getSlides().map((s) =>
          1 + Math.max(-1, ...[...s.querySelectorAll('.fragment')].map((f) => Number(f.dataset.fragmentIndex))))""")

        async def move(how: str) -> None:
            await page.evaluate(how)
            await page.wait_for_timeout(args.step_ms)

        while not await page.evaluate("deck.isLastSlide() && !deck.availableFragments().next"):
            await move("deck.next()")
        print(f"forward through {len(counts)} slides", flush=True)
        while not await page.evaluate("deck.isFirstSlide() && !deck.availableFragments().prev"):
            await move("deck.prev()")
        print("back to the start", flush=True)

        for index, count in enumerate(counts):
            for fragment in sorted({-1, count - 1}):
                for _ in range(args.burst):
                    other = rng.randrange(len(counts))
                    await move(f"deck.slide({other}, 0, {rng.randrange(-1, counts[other])})")
                await page.evaluate(f"deck.slide({index}, 0, {fragment})")
                await reference.evaluate(f"deck.slide({index}, 0, {fragment})")
                await page.wait_for_timeout(args.settle_ms)
                got, expected = await page.evaluate(STATE), await reference.evaluate(STATE)
                if got != expected:
                    problems.append(f"#{expected['id']} at fragment {fragment}: {got} after skipping, {expected} settled")
                problems.extend(f"#{expected['id']} at fragment {fragment}: plays off its slide: {s}" for s in await page.evaluate(STRAY))
            print(f"#{expected['id']}: checked", flush=True)
        await browser.close()

    shown = [e for e in errors if not (args.ignore and re.search(args.ignore, e))]
    for line in problems + shown:
        print(line, file=sys.stderr)
    print(f"{len(problems)} state problems, {len(shown)} browser errors ({len(errors) - len(shown)} ignored)", flush=True)
    return 1 if problems or shown else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(stress(tyro.cli(Stress))))
