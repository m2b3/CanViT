# /// script
# requires-python = ">=3.12"
# dependencies = ["playwright>=1.63", "tyro"]
# ///
"""A talk as one self-contained HTML file, which runs from a download folder (file://) without a server or a network:
the fallback between the served deck and its images (pdf_backup.py), for an organizer who asks for a file.

    node site/serve.mjs 8765 &                            # from the repository root
    uv run site/slides/single_file.py --talk 2026-10-30-Montreal-AI-and-Neuroscience-MAIN --out site/.screens/single

A browser loads no module script, font or data file from file://, so the file carries everything: the page's module
script and its imports bundled into one by esbuild (site/slides/package.json pins it), the stylesheets inlined with their
fonts and images as data URIs, and every other file the deck requests (found by walking it locally, every click and
every view of its episodes) embedded under a <base> at a made-up origin; a first script answers the deck's fetches and
image sources from them. Then the file is opened from disk offline and walked again: the build fails on a request it
cannot answer, a page error or a slide that does not appear. Backup slides stay in.
"""

import asyncio
import base64
import json
import mimetypes
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import tyro
from playwright.async_api import Page, async_playwright

from check_published import show_every_episode_view

SITE = Path(__file__).resolve().parent.parent
ORIGIN = "https://canvit.local"  # the made-up origin the embedded files live under; never requested
MIME = {".mjs": "text/javascript", ".js": "text/javascript", ".json": "application/json", ".woff2": "font/woff2",
        ".svg": "image/svg+xml", ".py": "text/plain", ".md": "text/plain"}

SHIM = """(() => {
  // Embedded files under ORIGIN, answered to fetch() and to image sources (single_file.py).
  const FILES = JSON.parse(document.getElementById("single-file-data").textContent);
  const ORIGIN = "%s";
  const bytes = (data) => Uint8Array.from(atob(data), (c) => c.charCodeAt(0));
  const path = (url) => {
    try { const u = new URL(url, document.baseURI); return u.origin === ORIGIN ? decodeURIComponent(u.pathname) : null; }
    catch { return null; }
  };
  const blobs = new Map();
  const local = (value) => {
    const p = path(value);
    if (p === null) return value;
    if (!(p in FILES)) { console.error(`single file: ${p} is not embedded`); return value; }
    if (!blobs.has(p)) blobs.set(p, URL.createObjectURL(new Blob([bytes(FILES[p][1])], { type: FILES[p][0] })));
    return blobs.get(p);
  };
  const fetch = window.fetch.bind(window);
  window.fetch = async (input, init) => {
    const p = path(typeof input === "string" ? input : input instanceof URL ? input.href : input.url);
    if (p === null) return fetch(input, init);
    if (!(p in FILES)) { console.error(`single file: ${p} is not embedded`); return new Response("", { status: 404 }); }
    return new Response(bytes(FILES[p][1]), { status: 200, headers: { "content-type": FILES[p][0] } });
  };
  for (const proto of [HTMLImageElement.prototype, HTMLSourceElement.prototype, HTMLMediaElement.prototype]) {
    const src = Object.getOwnPropertyDescriptor(proto, "src");
    Object.defineProperty(proto, "src", { ...src, set(value) { src.set.call(this, local(value)); } });
  }
  // The <base> would send reveal.js's hash updates to ORIGIN, which a file:// page may not name: they stay on the file.
  for (const name of ["replaceState", "pushState"]) {
    const original = history[name].bind(history);
    history[name] = (state, title, url) => {
      if (url != null && new URL(url, document.baseURI).origin === ORIGIN) {
        const here = new URL(location.href);
        here.hash = new URL(url, document.baseURI).hash;
        url = here.href;
      }
      return original(state, title, url);
    };
  }
  const setAttribute = Element.prototype.setAttribute;
  Element.prototype.setAttribute = function (name, value) {
    const media = this instanceof HTMLImageElement || this instanceof SVGImageElement || this instanceof HTMLSourceElement;
    return setAttribute.call(this, name, media && /^(src|href|xlink:href)$/.test(name) ? local(value) : value);
  };
})();""" % ORIGIN


@dataclass(frozen=True)
class Single:
    talk: str
    """the talk's directory under site/slides/"""
    out: Path = Path("site/.screens/single")
    local: str = "http://127.0.0.1:8765"
    wait_ms: int = 700
    reuse_walk: bool = False
    """take the requested files from the last walk (files.json in --out) instead of walking the local deck again"""


def mime(path: Path) -> str:
    return MIME.get(path.suffix) or mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def data_uri(path: Path) -> str:
    return f"data:{mime(path)};base64,{base64.b64encode(path.read_bytes()).decode()}"


async def walk(page: Page, wait_ms: int) -> list[str]:
    """Every slide and click of the deck, each episode view shown; returns the slides' ids."""
    await page.wait_for_function("window.deck?.isReady()")
    ids = await page.evaluate("deck.getSlides().map((s) => s.id)")
    for index in range(len(ids)):
        await page.evaluate(f"deck.slide({index}, 0, -1)")
        await page.wait_for_timeout(wait_ms)
        while await page.evaluate("deck.nextFragment()"):
            await page.wait_for_timeout(wait_ms)
        await show_every_episode_view(page, wait_ms)
    return ids


async def requested_files(args: Single) -> set[str]:
    """The site's files the deck requests, as paths under site/."""
    paths: set[str] = set()
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(channel="chrome")
        page = await browser.new_page(viewport={"width": 1280, "height": 720})
        page.on("request", lambda r: r.url.startswith(args.local) and paths.add(r.url[len(args.local):].split("?")[0]))
        await page.goto(f"{args.local}/slides/{args.talk}/?present")
        await walk(page, args.wait_ms)
        await browser.close()
    return {p.lstrip("/") for p in paths if p not in ("/__reload",) and not p.endswith("/")}


def bundle(talk_dir: Path, module: str) -> str:
    entry = talk_dir / "_single_entry.mjs"  # beside index.html, so the script's relative imports resolve
    entry.write_text(module)
    try:
        esbuild = SITE / "slides/node_modules/.bin/esbuild"
        result = subprocess.run([str(esbuild), str(entry), "--bundle", "--format=esm", "--target=es2022",
                                 "--log-level=warning"], capture_output=True, text=True)
        if result.returncode:
            raise SystemExit(f"esbuild failed:\n{result.stderr}")
        return result.stdout
    finally:
        entry.unlink()


def inline_css(path: Path) -> str:
    css = path.read_text()
    css = re.sub(r'@import url\("([^"]+)"\);', lambda m: inline_css((path.parent / m.group(1)).resolve()), css)
    return re.sub(r'url\("?(?!#|data:)([^")]+)"?\)', lambda m: f'url("{data_uri((path.parent / m.group(1)).resolve())}")', css)


def build(args: Single, files: set[str]) -> str:
    talk_dir = SITE / "slides" / args.talk
    page = (talk_dir / "index.html").read_text()
    module = re.search(r'<script type="module">(.*?)</script>', page, re.S)
    page = page.replace(module.group(0), f'<script type="module">{bundle(talk_dir, module.group(1))}</script>')
    page = re.sub(r'\s*<link rel="preload"[^>]*>', "", page)
    page = re.sub(r'<link rel="stylesheet" href="([^"]+)">',
                  lambda m: f"<style>{inline_css((talk_dir / m.group(1)).resolve())}</style>", page)

    def media(tag: re.Match) -> str:  # static image sources as data URIs, since the parser loads them before any script
        def attribute(m: re.Match) -> str:
            target = (talk_dir / m.group(2)).resolve()
            return f'{m.group(1)}="{data_uri(target)}"' if target.is_file() else m.group(0)
        return re.sub(r'\b(src|href)="(?!data:|https?:|#)([^"]+)"', attribute, tag.group(0))
    page = re.sub(r"<(img|image|source)\b[^>]*>", media, page)

    embedded = {f"/{f}": [mime(SITE / f), base64.b64encode((SITE / f).read_bytes()).decode()]
                for f in sorted(files) if (SITE / f).is_file() and not f.endswith((".js", ".mjs", ".css", ".html"))}
    head = (f'<base href="{ORIGIN}/slides/{args.talk}/">\n'
            f'<script type="application/json" id="single-file-data">{json.dumps(embedded)}</script>\n'
            f"<script>{SHIM}</script>\n")
    return page.replace("<head>\n", "<head>\n" + head, 1), len(embedded)


async def verify(out: Path, ids: list[str], wait_ms: int) -> list[str]:
    """Open the file from disk with the network off and walk it: every page error, unanswered request and missing slide."""
    problems: list[str] = []
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(channel="chrome")
        context = await browser.new_context(viewport={"width": 1280, "height": 720}, offline=True)
        page = await context.new_page()
        page.on("pageerror", lambda e: problems.append(f"page error: {e}"))
        page.on("console", lambda m: m.type == "error" and problems.append(f"console: {m.text[:200]}"))
        await page.goto(out.resolve().as_uri() + "?present")
        seen = await walk(page, wait_ms)
        problems += [f"slide #{i} missing" for i in ids if i not in seen]
        await browser.close()
    return problems


def main(args: Single) -> int:
    args.out.mkdir(parents=True, exist_ok=True)
    listing = args.out / "files.json"
    if args.reuse_walk and listing.exists():
        files = set(json.loads(listing.read_text()))
    else:
        files = asyncio.run(requested_files(args))
        listing.write_text(json.dumps(sorted(files), indent=1))
    page, count = build(args, files)
    out = args.out / f"{args.talk}.html"
    out.write_text(page)
    ids_local = [s for s in re.findall(r'<section id="([^"]+)"', page)]
    problems = asyncio.run(verify(out, ids_local, args.wait_ms))
    report = [f"{out}: {out.stat().st_size / 1e6:.1f} MB, {count} embedded files, {len(ids_local)} slides",
              *(problems or ["opened from disk offline: every slide and click walked, no errors"])]
    (args.out / "check.txt").write_text("\n".join(report) + "\n")
    print("\n".join(report))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(tyro.cli(Single)))
