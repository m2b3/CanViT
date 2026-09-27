// Behavior of index.html: the result bars, the policy slot, the citation's copy button.

// Each bar's length is its own printed value, relative to the largest value in its group, from zero.
const bars = document.querySelectorAll(".bars");
for (const group of bars) {
  const rows = [...group.querySelectorAll(".bar")];
  const values = rows.map((row) => parseFloat(row.querySelector(".num").textContent));
  const max = Math.max(...values);
  rows.forEach((row, i) => row.style.setProperty("--fraction", String(values[i] / max)));
}
new IntersectionObserver((entries, observer) => {
  for (const entry of entries) if (entry.isIntersecting) { entry.target.classList.add("shown"); observer.unobserve(entry.target); }
}, { threshold: 0.3 }).observe(document.querySelector(".comparisons"));

const copy = document.querySelector(".copy");
copy.addEventListener("click", async () => {
  await navigator.clipboard.writeText(document.querySelector(".bibtex code").textContent);
  copy.textContent = "Copied";
  setTimeout(() => { copy.textContent = "Copy"; }, 1600);
});

// The policy slot cycles through the paper's viewing policies, slow then fast, and lands on "Anything you want".
const POLICY_NAMES = ["Coarse-to-fine", "Fine-to-coarse", "Entropy-guided", "Random", "Full, then random", "Repeated full view"];
const slot = document.querySelector(".cycle");
const final = slot.textContent;
if (!matchMedia("(prefers-reduced-motion: reduce)").matches) {
  new IntersectionObserver(([entry], observer) => {
    if (!entry.isIntersecting) return;
    observer.disconnect();
    const names = [...POLICY_NAMES, ...POLICY_NAMES];
    let delay = 560;
    const show = (i) => {
      if (i === names.length) {
        slot.textContent = final;
        slot.classList.add("landed");
        return;
      }
      slot.textContent = names[i];
      delay = Math.max(70, delay * 0.8);
      setTimeout(() => show(i + 1), delay);
    };
    slot.classList.add("cycling");
    show(0);
  }, { threshold: 0.6 }).observe(slot);
}
