// Behavior of index.html: the dot plots, the policy slot, the citation's copy button.

import { POLICIES } from "./canvit/policies.js";

// Dot plots: each dot sits at its own printed value on an axis zoomed to the group's values, with labeled ticks,
// so small differences show without bars whose lengths would misstate the ratios.
for (const group of document.querySelectorAll(".dots")) {
  const rows = [...group.querySelectorAll(".dot-row")];
  const values = rows.map((row) => parseFloat(row.querySelector(".num").textContent));
  const [lo, hi] = [Math.floor(Math.min(...values)) - 1, Math.ceil(Math.max(...values)) + 1];
  const position = (value) => (value - lo) / (hi - lo);
  rows.forEach((row, i) => row.style.setProperty("--position", String(position(values[i]))));
  const step = hi - lo > 6 ? 2 : 1;
  group.querySelector(".ticks").innerHTML = Array.from({ length: Math.floor((hi - lo) / step) + 1 }, (_, k) => lo + k * step)
    .map((tick) => `<span style="left: ${100 * position(tick)}%">${tick}%</span>`).join("");
  group.style.setProperty("--tick-fraction", String(step / (hi - lo)));
  new IntersectionObserver(([entry], observer) => {
    if (!entry.isIntersecting) return;
    group.classList.add("shown");
    observer.disconnect();
  }, { threshold: 0.5 }).observe(group);
}

const copy = document.querySelector(".copy");
copy.addEventListener("click", async () => {
  await navigator.clipboard.writeText(document.querySelector(".bibtex code").textContent);
  copy.textContent = "Copied";
  setTimeout(() => { copy.textContent = "Copy"; }, 1600);
});

// The policy slot cycles through the paper's viewing policies in their figure colors, slow then fast, and lands on
// "Anything you want".
const slot = document.querySelector(".cycle");
const final = slot.textContent;
if (!matchMedia("(prefers-reduced-motion: reduce)").matches) {
  new IntersectionObserver(([entry], observer) => {
    if (!entry.isIntersecting) return;
    observer.disconnect();
    const policies = Object.values(POLICIES);
    const names = [...policies, ...policies];
    let delay = 560;
    const show = (i) => {
      if (i === names.length) {
        slot.textContent = final;
        slot.style.color = "";
        slot.classList.add("landed");
        return;
      }
      slot.textContent = names[i].name[0].toUpperCase() + names[i].name.slice(1);
      slot.style.color = names[i].color;
      delay = Math.max(70, delay * 0.8);
      setTimeout(() => show(i + 1), delay);
    };
    slot.classList.add("cycling");
    show(0);
  }, { threshold: 0.6 }).observe(slot);
}
