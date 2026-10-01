# /// script
# requires-python = ">=3.12"
# dependencies = ["playwright>=1.63", "tyro"]
# ///
"""Screenshot every slide of a talk in Chromium, each with all its fragments shown, and report browser errors.

    python3 -m http.server 8765 --directory site &        # from the repository root
    uv run site/slides/shoot.py --url http://127.0.0.1:8765/slides/2026-10-30-Montreal-AI-and-Neuroscience-MAIN/ \
        --out site/.screens/main-2026

Writes NN-<slide id>.png and errors.txt into --out; exits with status 1 when the page logged an error or a slide is
missing an id. Animations are captured after --wait-ms on each slide.
"""

import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path

import tyro
from playwright.async_api import ConsoleMessage, async_playwright


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


async def shoot(args: Shoot) -> int:
    args.out.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
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
            await page.wait_for_timeout(args.wait_ms)
            path = args.out / f"{index + 1:02d}-{slide_id}.png"
            await page.screenshot(path=path)
            print(f"{path}", flush=True)
        await browser.close()
    (args.out / "errors.txt").write_text("".join(f"{e}\n" for e in errors))
    for error in errors:
        print(error, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(shoot(tyro.cli(Shoot))))
