# /// script
# requires-python = ">=3.12"
# dependencies = ["playwright>=1.63", "tyro"]
# ///
"""Every file the project page and a talk load from the local site is served by the published one.

    node site/serve.mjs 8765 &
    uv run site/slides/check_published.py --talk 2026-10-30-Montreal-AI-and-Neuroscience-MAIN

Opens the page and scrolls through it, then walks the talk's every slide and click, showing each scene and readout of
its episodes, and records the site's files they request; then requests each from the published site. Exits with
status 1 when a file is missing there or a local request failed.
"""

import asyncio
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from urllib.parse import urlsplit

import tyro
from playwright.async_api import Page, async_playwright

LOCAL_ONLY = {"/__reload"}  # the dev server's reload channel


@dataclass(frozen=True)
class Check:
    talk: str
    """the talk's directory under site/slides/"""
    local: str = "http://127.0.0.1:8765"
    published: str = "https://m2b3.github.io/CanViT"
    wait_ms: int = 700


async def show_every_episode_view(page: Page, wait_ms: int) -> None:
    """Click each scene tab and readout of the current slide's episodes, so their files load."""
    count = await page.evaluate("deck.getCurrentSlide().querySelectorAll('canvit-episode').length")
    for e in range(count):
        for selector in (".scenes button", ".readouts button"):
            buttons = await page.evaluate(
                f"deck.getCurrentSlide().querySelectorAll('canvit-episode')[{e}].shadowRoot"
                f".querySelectorAll('{selector}').length")
            for b in range(buttons):
                await page.evaluate(
                    f"deck.getCurrentSlide().querySelectorAll('canvit-episode')[{e}].shadowRoot"
                    f".querySelectorAll('{selector}')[{b}].click()")
                await page.wait_for_timeout(wait_ms)


async def requested_paths(args: Check) -> tuple[set[str], list[str]]:
    paths: set[str] = set()
    failures: list[str] = []
    local = urlsplit(args.local)
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(channel="chrome")
        page = await browser.new_page(viewport={"width": 1280, "height": 720})

        def on_request(request) -> None:
            url = urlsplit(request.url)
            if url.netloc == local.netloc and url.path not in LOCAL_ONLY:
                paths.add(url.path)

        page.on("request", on_request)
        page.on("requestfailed", lambda r: failures.append(f"failed locally: {r.url} ({r.failure})"))
        page.on("response", lambda r: r.status >= 400 and failures.append(f"{r.status} locally: {r.url}"))

        await page.goto(f"{args.local}/")
        height = await page.evaluate("document.body.scrollHeight")
        for y in range(0, height, 600):
            await page.evaluate(f"window.scrollTo(0, {y})")
            await page.wait_for_timeout(args.wait_ms // 2)

        await page.goto(f"{args.local}/slides/{args.talk}/")
        await page.wait_for_function("window.deck?.isReady()")
        for index in range(await page.evaluate("deck.getTotalSlides()")):
            await page.evaluate(f"deck.slide({index}, 0, -1)")
            while True:
                await page.wait_for_timeout(args.wait_ms)
                await show_every_episode_view(page, args.wait_ms)
                if not await page.evaluate("deck.nextFragment()"):
                    break
        await browser.close()
    return paths, failures


def published_status(url: str) -> int:
    try:
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=30) as response:
            return response.status
    except urllib.error.HTTPError as error:
        return error.code


def main(args: Check) -> int:
    paths, failures = asyncio.run(requested_paths(args))
    missing = [f"{status} published: {path}" for path in sorted(paths)
               if (status := published_status(args.published + path)) != 200]
    print(f"{len(paths)} files requested locally")
    for line in failures + missing:
        print(line, file=sys.stderr)
    return 1 if failures or missing else 0


if __name__ == "__main__":
    sys.exit(main(tyro.cli(Check)))
