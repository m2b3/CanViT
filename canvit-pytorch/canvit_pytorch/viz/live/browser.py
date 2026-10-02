"""<canvit-live> in headless Chrome against the parity reference: outputs, EG-C2F's choices, pointer input, timings.

Needs a page with one <canvit-live> served over HTTP, the reference `parity` wrote (<export>/parity/) served over
HTTP too, and Google Chrome installed (Playwright's channel "chrome"). The page may run a local export or the
published ones: the reference is the same as long as the graphs are.
The element is driven as a visitor drives it: its load button, clicks, the wheel, a drag, the keyboard; then the page
is reloaded, and the second load must read the graphs from the browser's cache.
"""

import asyncio
import json
import logging
import statistics
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from playwright.async_api import Page, async_playwright

from canvit_pytorch.viz.live.parity import MAX_REL_L2, MIN_ARGMAX_AGREEMENT

log = logging.getLogger(__name__)

Backend = Literal["webgpu", "wasm"]
BACKEND_NAMES: dict[Backend, str] = {"webgpu": "WebGPU", "wasm": "WebAssembly"}
# Without a GPU process Chrome offers no WebGPU adapter, so the page falls back as it does for such visitors.
CHROME_ARGS: dict[Backend, list[str]] = {"webgpu": [], "wasm": ["--disable-gpu"]}
POINTER_TOLERANCE = 0.02
"""Largest accepted difference between a pointer-chosen viewpoint and the one expected from the pointer's position,
in scene coordinates: pointer positions are rounded to CSS pixels, a few thousandths of the scene at this size."""

# Replays the reference viewpoints through the element, then lets its EG-C2F choose, then times whole episodes.
REPLAY_JS = r"""
async ({ timedEpisodes, referenceUrl }) => {
  const el = document.querySelector("canvit-live");
  const base = new URL(referenceUrl.replace(/\/?$/, "/"), document.baseURI);
  const fetchOk = async (url) => { const r = await fetch(url); if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`); return r; };
  const episode = await (await fetchOk(new URL("episode.json", base))).json();
  const reference = async (name, t) =>
    new Float32Array(await (await fetchOk(new URL(`reference/${name}_t${String(t).padStart(2, "0")}.bin`, base))).arrayBuffer());
  const errors = (actual, ref) => {
    if (actual.length !== ref.length) throw new Error(`length ${actual.length}, reference ${ref.length}`);
    let maxAbs = 0, maxRef = 0, diff2 = 0, ref2 = 0;
    for (let i = 0; i < ref.length; i++) {
      const d = actual[i] - ref[i];
      maxAbs = Math.max(maxAbs, Math.abs(d)); maxRef = Math.max(maxRef, Math.abs(ref[i]));
      diff2 += d * d; ref2 += ref[i] * ref[i];
    }
    return { max_abs: maxAbs, max_abs_over_max_ref: maxAbs / maxRef, rel_l2: Math.sqrt(diff2 / ref2) };
  };
  const numClasses = episode.num_classes;
  const argmax = (logits) => {
    const n = logits.length / numClasses, labels = new Uint8Array(n), best = logits.slice(0, n);
    for (let c = 1; c < numClasses; c++) for (let i = 0; i < n; i++) if (logits[c * n + i] > best[i]) { best[i] = logits[c * n + i]; labels[i] = c; }
    return labels;
  };

  await el.reset();
  const replay = [];
  for (const [t, [row, col, scale]] of episode.viewpoints.entries()) {
    const out = await el.look({ row, col, scale });
    const refLogits = await reference("logits", t);
    const refLabels = argmax(refLogits);
    const record = { t, step_ms: out.stepMs, run_ms: out.runMs, logits: errors(out.logits, refLogits),
                     entropy: errors(out.entropy, await reference("entropy", t)),
                     argmax_agreement: out.labels.filter((c, i) => c === refLabels[i]).length / refLabels.length };
    if (episode.full_steps.includes(t)) {
      record.canvas = errors(await out.readCanvas(), await reference("canvas", t));
      record.glimpse = errors(out.glimpse, await reference("glimpse", t));
    }
    replay.push(record);
  }

  await el.reset();
  const chosen = [];
  for (let t = 0; t < episode.egc2f_glimpses; t++) {
    const { viewpoint } = await el.policyStep();
    chosen.push([viewpoint.row, viewpoint.col, viewpoint.scale]);
  }
  const matches = chosen.map((v, t) => v.every((x, k) => x === episode.viewpoints[t][k]));

  const timed = [];
  for (let e = 0; e < timedEpisodes; e++) {
    await el.reset();
    for (let t = 0; t < episode.egc2f_glimpses; t++) {
      const out = await el.policyStep();
      timed.push({ step_ms: out.stepMs, run_ms: out.runMs });
    }
  }
  await el.reset();
  return { replay, closed_loop: { chosen, matches_reference: matches }, timed };
}
"""


def summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {"n": len(ordered), "min": ordered[0], "median": statistics.median(ordered),
            "p90": ordered[int(0.9 * (len(ordered) - 1))], "max": ordered[-1]}


async def last_viewpoint(page: Page, count: int) -> dict[str, float]:
    await page.wait_for_function(f"document.querySelector('canvit-live').history.length === {count}", timeout=120_000)
    return await page.evaluate("document.querySelector('canvit-live').history.at(-1)")


def inside_scene(row: float, col: float, scale: float) -> dict[str, float]:
    return {"row": min(max(row, scale - 1), 1 - scale), "col": min(max(col, scale - 1), 1 - scale), "scale": scale}


async def load(page: Page) -> float:
    """Press the element's load button; milliseconds until it is ready."""
    await page.locator("canvit-live").scroll_into_view_if_needed()
    await page.locator("canvit-live [data-load]").click()
    started = await page.evaluate("performance.now()")
    await page.evaluate("document.querySelector('canvit-live').ready")
    return await page.evaluate("performance.now()") - started


@dataclass(frozen=True)
class CheckBrowser:
    """Check <canvit-live> in headless Chrome against the PyTorch reference written by `parity`."""

    page_url: str
    """A page with one <canvit-live>, e.g. the project page at http://127.0.0.1:8020/."""
    reference_url: str
    """The parity directory of the export the page runs, e.g. http://127.0.0.1:8020/.live-model/parity/."""
    backend: Backend
    out: Path
    """Report (JSON); screenshots go next to it."""
    timed_episodes: int = 2
    """Whole EG-C2F episodes timed after the checks."""
    headless: bool = True

    def run(self) -> Path:
        report = asyncio.run(self._check())
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.out.write_text(json.dumps(report, indent=1) + "\n")
        worst, pointer = report["worst"], report["pointer"]
        log.info("%s: worst %s; closed loop %s; pointer %s; warm step ms %s", report["backend"], worst,
                 report["closed_loop"]["matches"], [p["ok"] for p in pointer], report["warm_step_ms"])
        assert report["backend"] == BACKEND_NAMES[self.backend], report["backend"]
        assert all(worst[key] <= MAX_REL_L2 for key in ("logits", "entropy", "canvas", "glimpse")), worst
        assert worst["min_argmax_agreement"] >= MIN_ARGMAX_AGREEMENT, worst
        assert report["closed_loop"]["matches"] == report["closed_loop"]["glimpses"], report["closed_loop"]
        assert all(p["ok"] for p in pointer), pointer
        assert report["graph_requests"], "the first load requested no .onnx file: the check cannot see downloads"
        cached = report["cached_load"]
        assert "cached in this browser" in cached["button"] and not cached["graph_requests"], cached
        assert not report["page_errors"], report["page_errors"]
        return self.out

    async def _check(self) -> dict[str, Any]:
        screens = self.out.with_suffix("")
        # A fresh profile on disk, as a visitor's: Chrome's in-memory profiles refuse a cache entry the size of the graph.
        with tempfile.TemporaryDirectory() as profile:
            async with async_playwright() as playwright:
                context = await playwright.chromium.launch_persistent_context(
                    profile, channel="chrome", headless=self.headless, args=CHROME_ARGS[self.backend],
                    viewport={"width": 1280, "height": 1000}, device_scale_factor=2)
                assert context.browser is not None
                page = await context.new_page()
                console: list[str] = []
                page_errors: list[str] = []
                page.on("console", lambda message: console.append(f"{message.type}: {message.text}"))
                page.on("pageerror", lambda error: page_errors.append(str(error)))
                graph_requests: list[str] = []
                page.on("request", lambda request: graph_requests.append(request.url) if ".onnx" in request.url else None)
                await page.goto(self.page_url)
                live = page.locator("canvit-live")
                load_ms = await load(page)
                backend = await page.evaluate("document.querySelector('canvit-live').backend")
                log.info("%s ready in %.0f ms; replaying the reference", backend, load_ms)

                result = await page.evaluate(REPLAY_JS, {"timedEpisodes": self.timed_episodes,
                                                         "referenceUrl": self.reference_url})
                pointer = await self._pointer(page)
                await live.screenshot(path=f"{screens}-pointer.png")
                await page.locator("canvit-live [data-policy-run]").click()
                await last_viewpoint(page, len(result["closed_loop"]["chosen"]))
                await page.locator("canvit-live [data-scene-frame]").hover()
                await live.screenshot(path=f"{screens}-egc2f.png")
                resources = await page.evaluate(
                    "performance.getEntriesByType('resource').map((e) => ({name: e.name, ms: e.duration, bytes: e.decodedBodySize}))")
                report = {
                    "page": self.page_url, "reference": self.reference_url, "chrome": context.browser.version, "headless": self.headless, "backend": backend,
                    "adapter": await page.evaluate(
                        "(async () => { const a = await navigator.gpu?.requestAdapter(); return a ? a.info.vendor + ' ' + a.info.architecture : null })()"),
                    "load_ms": load_ms,
                    "graph_requests": list(graph_requests),
                    # Cross-origin files report a size only when their server sends Timing-Allow-Origin.
                    "downloads": [r for r in resources if r["bytes"] > 1_000_000 and "/parity/" not in r["name"]],
                    "worst": {
                        key: max(r[key]["rel_l2"] for r in result["replay"] if key in r)
                        for key in ("logits", "entropy", "canvas", "glimpse")
                    } | {"max_abs_logits": max(r["logits"]["max_abs"] for r in result["replay"]),
                         "min_argmax_agreement": min(r["argmax_agreement"] for r in result["replay"])},
                    "closed_loop": {"matches": sum(result["closed_loop"]["matches_reference"]),
                                    "glimpses": len(result["closed_loop"]["matches_reference"])},
                    # The first glimpse after loading includes one-time setup.
                    "warm_step_ms": summary([r["step_ms"] for r in result["replay"][1:]] + [r["step_ms"] for r in result["timed"]]),
                    "warm_run_ms": summary([r["run_ms"] for r in result["replay"][1:]] + [r["run_ms"] for r in result["timed"]]),
                    "first_step_ms": result["replay"][0]["step_ms"],
                    "pointer": pointer,
                    "replay": result["replay"],
                }
                graph_requests.clear()
                await page.reload()
                button = await page.locator("canvit-live [data-load]").text_content()
                report["cached_load"] = {"button": button, "load_ms": await load(page), "graph_requests": graph_requests}
                log.info("reloaded: %r, ready in %.0f ms", button, report["cached_load"]["load_ms"])
                report |= {"page_errors": page_errors, "console": console}
                await context.close()
        return report

    async def _pointer(self, page: Page) -> list[dict[str, Any]]:
        """Clicks at arbitrary points and scales, a drag and the keyboard; each must glimpse where aimed."""
        frame = page.locator("canvit-live [data-scene-frame]")
        await page.locator("canvit-live [data-reset]").click()
        # Measured after the click, which may scroll a long page, and in view: the mouse works in viewport coordinates.
        await frame.scroll_into_view_if_needed()
        box = await frame.bounding_box()
        assert box is not None
        at = lambda x, y: (box["x"] + x * box["width"], box["y"] + y * box["height"])  # noqa: E731
        scale_now = lambda: page.evaluate(  # noqa: E731
            "Number(document.querySelector('canvit-live').shadowRoot.querySelector('.size output').textContent.split(/\\s+/)[1])")
        checks: list[dict[str, Any]] = []
        count = 0

        async def expect(name: str, expected: dict[str, float]) -> None:
            nonlocal count
            count += 1
            got = await last_viewpoint(page, count)
            ok = all(abs(got[k] - expected[k]) <= POINTER_TOLERANCE for k in ("row", "col", "scale"))
            checks.append({"input": name, "expected": expected, "got": got, "ok": ok})

        for (x, y), wheel in (((0.31, 0.62), 0), ((0.83, 0.17), -300), ((0.5, 0.45), 900), ((0.07, 0.93), -450)):
            await page.mouse.move(*at(x, y))
            if wheel:
                await page.mouse.wheel(0, wheel)
            scale = float(await scale_now())
            await page.mouse.click(*at(x, y))
            await expect(f"click at ({x}, {y}) after wheel {wheel}", inside_scene(2 * y - 1, 2 * x - 1, scale))

        start, end = (0.6, 0.35), (0.72, 0.41)
        await page.mouse.move(*at(*start))
        await page.mouse.down()
        await page.mouse.move(*at(*end), steps=8)
        await page.mouse.up()
        drag_scale = 2 * max(abs(end[0] - start[0]), abs(end[1] - start[1]))
        await expect(f"drag from {start} to {end}", inside_scene(2 * start[1] - 1, 2 * start[0] - 1, drag_scale))

        await frame.focus()
        previous = await page.evaluate("document.querySelector('canvit-live').history.at(-1)")
        await page.mouse.move(0, 0)  # the keyboard moves the box from where the pointer left it
        for key in ("ArrowRight", "ArrowRight", "ArrowUp", "Enter"):
            await page.keyboard.press(key)
        await expect("keyboard: right, right, up, enter from the scene's center",
                     inside_scene(-0.05, 0.1, previous["scale"]))
        return checks
