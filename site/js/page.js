// Behavior of index.html: the policy slot, the claims' underlines, the citation's copy button.

import { POLICIES } from "./canvit/policies.js";

// Runs `action` once, the first time at least `threshold` of `element` is in view.
function onFirstView(element, threshold, action) {
  new IntersectionObserver(([entry], observer) => {
    if (!entry.isIntersecting) return;
    observer.disconnect();
    action();
  }, { threshold }).observe(element);
}

// Each claim's underlines draw in once it is in view.
for (const claim of document.querySelectorAll(".claim")) onFirstView(claim, 0.8, () => claim.classList.add("shown"));

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
  onFirstView(slot, 0.6, () => {
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
    show(0);
  });
}
