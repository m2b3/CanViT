// Step <canvit-*> elements with reveal.js fragments. A fragment
//   <span class="fragment" data-canvit-target="#cat" data-canvit-t="5"></span>
// sets t="5" on every element matching the selector within its slide while it is shown;
// with no such fragment shown, a target keeps the t it had in the markup.

const initial = new WeakMap();

export function bindFragments(deck) {
  const sync = () => {
    const slide = deck.getCurrentSlide();
    if (!slide) return;
    const fragments = [...slide.querySelectorAll("[data-canvit-target]")].sort(
      (a, b) => Number(a.dataset.fragmentIndex ?? 0) - Number(b.dataset.fragmentIndex ?? 0));
    const selectors = new Set(fragments.map((f) => f.dataset.canvitTarget));
    for (const selector of selectors) {
      const targets = slide.querySelectorAll(selector);
      if (targets.length === 0) console.error(`data-canvit-target="${selector}" matches nothing on this slide`);
      const shown = fragments.filter((f) => f.dataset.canvitTarget === selector && f.classList.contains("visible"));
      for (const target of targets) {
        if (!initial.has(target)) initial.set(target, target.getAttribute("t") ?? "0");
        target.setAttribute("t", shown.length ? shown.at(-1).dataset.canvitT : initial.get(target));
      }
    }
  };
  for (const event of ["ready", "slidechanged", "fragmentshown", "fragmenthidden"]) deck.on(event, sync);
  if (deck.isReady()) sync();
}
