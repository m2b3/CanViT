# /// script
# requires-python = ">=3.12"
# dependencies = ["playwright>=1.63", "tyro"]
# ///
"""Screenshot every slide of a talk in Chromium, each with all its fragments shown, and report browser errors.

    node site/serve.mjs 8765 &                            # from the repository root
    uv run site/slides/shoot.py --url http://127.0.0.1:8765/slides/2026-10-30-Montreal-AI-and-Neuroscience-MAIN/ \
        --out site/.screens/main-2026

Writes NN-<slide id>.png (with --steps, one image per click) and errors.txt into --out; exits with status 1 when the page logged an error or a slide is
missing an id. Animations are captured after --wait-ms on each slide. Layout problems on each captured state (a title
that wraps, visible content in the footer band, a word alone on the last line of a text block) go to layout.txt and
are printed, without changing the exit status: some titles wrap by design.
"""

import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path

import tyro
from playwright.async_api import ConsoleMessage, Page, async_playwright

FOOTER_TOP_PX = 676  # the footer band of the 1280 x 720 slide: the talk's name and the slide number
# The current slide's layout problems, in slide pixels: its title wrapping; the visible leaves (images, canvases, SVGs,
# text without child elements) whose bottom enters the footer band; and visible text blocks whose last line holds a
# single word, found by comparing the line boxes of their last two words when one text node holds both and no
# preserved line break separates them (words in separate elements, such as a label and its name, are often on
# separate lines by design). Visible: displayed, not hidden, and every ancestor up to the slide with nonzero opacity;
# an ancestor that clips its overflow cuts the bottom at its own.
LAYOUT_PROBLEMS_JS = r"""([footerTop]) => {
  const slide = deck.getCurrentSlide(), scale = deck.getScale();
  const origin = document.querySelector(".reveal .slides").getBoundingClientRect().top;
  const problems = [];
  const title = slide.querySelector("h2");
  if (title && title.getBoundingClientRect().height / scale > 1.5 * parseFloat(getComputedStyle(title).lineHeight)) {
    problems.push(`title wraps: "${title.textContent.trim()}"`);
  }
  const visible = (el) => {
    for (let node = el; node && node !== slide.parentElement; node = node.parentElement) {
      const style = getComputedStyle(node);
      if (style.display === "none" || style.visibility === "hidden" || Number(style.opacity) < 0.05) return false;
    }
    return true;
  };
  for (const el of slide.querySelectorAll("img, canvas, svg, :not(svg *):not(:has(*))")) {
    if (el.closest("aside.notes, .step-marker") || !el.getClientRects().length) continue;
    const box = el.getBoundingClientRect();
    let clipped = box.bottom;
    for (let node = el.parentElement; node && node !== slide.parentElement; node = node.parentElement) {
      if (getComputedStyle(node).overflowY !== "visible") clipped = Math.min(clipped, node.getBoundingClientRect().bottom);
    }
    const bottom = (clipped - origin) / scale;
    if (box.height > 0 && bottom > footerTop + 1 && visible(el)) {
      problems.push(`in the footer band (bottom ${Math.round(bottom)} px): <${el.tagName.toLowerCase()}${el.className && typeof el.className === "string" ? "." + el.className.trim().replace(/\s+/g, ".") : ""}> ${(el.textContent || "").trim().slice(0, 40)}`);
    }
  }
  for (const block of slide.querySelectorAll("h2, h3, p, figcaption, li, dt, dd")) {
    if (block.closest("aside.notes") || !block.getClientRects().length || !visible(block)) continue;
    const words = [];
    const walker = document.createTreeWalker(block, NodeFilter.SHOW_TEXT);
    for (let node = walker.nextNode(); node; node = walker.nextNode()) {
      for (const match of node.data.matchAll(/\S+/g)) words.push([node, match.index, match.index + match[0].length]);
    }
    if (words.length < 2) continue;
    const top = ([node, start, end]) => {
      const range = document.createRange();
      range.setStart(node, start);
      range.setEnd(node, end);
      return range.getBoundingClientRect().top;
    };
    const [last, before] = [words.at(-1), words.at(-2)];
    const breaks = /^pre/.test(getComputedStyle(last[0].parentElement).whiteSpace) && last[0].data.slice(before[2], last[1]).includes("\n");
    if (last[0] === before[0] && !breaks && top(last) - top(before) > 4 * scale) {
      problems.push(`a word alone on the last line: "${last[0].data.slice(last[1], last[2])}" in <${block.tagName.toLowerCase()}> ${block.textContent.trim().slice(0, 50)}`);
    }
  }
  return problems;
}"""


async def layout_problems(page: Page) -> list[str]:
    return await page.evaluate(LAYOUT_PROBLEMS_JS, [FOOTER_TOP_PX])


@dataclass(frozen=True)
class Shoot:
    url: str
    out: Path
    only: tuple[str, ...] = ()
    """Slide ids to capture; every slide when empty."""
    wait_ms: int = 2500
    scale: float = 1.5
    """Device pixels per CSS pixel: 1.5 gives 1920 × 1080 images."""
    first_fragment: bool = False
    """Capture each slide before its fragments, instead of after all of them."""
    steps: bool = False
    """Capture each slide before its fragments and after each one (NN-<id>-SS.png), to review a build click by click."""


async def shoot(args: Shoot) -> int:
    args.out.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    layout: list[str] = []
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(channel="chrome", args=["--enable-unsafe-webgpu"])
        page = await browser.new_page(viewport={"width": 1280, "height": 720}, device_scale_factor=args.scale)
        where = {"slide": "load"}

        def on_console(message: ConsoleMessage) -> None:
            if message.type == "error":
                errors.append(f"[{where['slide']}] console: {message.text}")

        page.on("console", on_console)
        page.on("pageerror", lambda error: errors.append(f"[{where['slide']}] page error: {error}"))
        await page.goto(args.url)
        await page.wait_for_function("window.deck?.isReady()")
        count = await page.evaluate("deck.getTotalSlides()")
        for index in range(count):
            await page.evaluate(f"deck.slide({index}, 0, {-1 if args.first_fragment else 'Infinity'})")
            slide_id = await page.evaluate("deck.getCurrentSlide().id")
            if not slide_id:
                errors.append(f"slide {index + 1} has no id")
                slide_id = "no-id"
            where["slide"] = slide_id
            if args.only and slide_id not in args.only:
                continue
            if args.steps:
                await page.evaluate(f"deck.slide({index}, 0, -1)")
                step = 0
                while True:
                    await page.wait_for_timeout(args.wait_ms)
                    path = args.out / f"{index + 1:02d}-{slide_id}-{step:02d}.png"
                    await page.screenshot(path=path)
                    print(f"{path}", flush=True)
                    layout += [f"[{slide_id} step {step}] {p}" for p in await layout_problems(page)]
                    if not await page.evaluate("deck.nextFragment()"):
                        break
                    step += 1
                continue
            await page.wait_for_timeout(args.wait_ms)
            path = args.out / f"{index + 1:02d}-{slide_id}.png"
            await page.screenshot(path=path)
            print(f"{path}", flush=True)
            layout += [f"[{slide_id}] {p}" for p in await layout_problems(page)]
        await browser.close()
    (args.out / "layout.txt").write_text("".join(f"{line}\n" for line in layout))
    for line in layout:
        print(f"layout: {line}", file=sys.stderr)
    (args.out / "errors.txt").write_text("".join(f"{e}\n" for e in errors))
    for error in errors:
        print(error, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(shoot(tyro.cli(Shoot))))
