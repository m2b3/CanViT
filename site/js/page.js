// Page behavior for index.html; the components live in js/canvit/.

import "./canvit/index.js";

// Scene switcher for the smooth-path demo.
const path = document.getElementById("path");
const sceneButtons = document.querySelectorAll("[data-scene]");
for (const button of sceneButtons) {
  button.addEventListener("click", () => {
    for (const other of sceneButtons) other.setAttribute("aria-pressed", String(other === button));
    path.setAttribute("src", button.dataset.scene);
  });
}

// Light/dark switch; the choice is a per-viewer convenience, so storage failures are ignored.
document.querySelector(".theme").addEventListener("click", () => {
  const root = document.documentElement;
  const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  root.dataset.theme = dark ? "light" : "dark";
  try { localStorage.setItem("theme", root.dataset.theme); } catch {}
});

for (const button of document.querySelectorAll("[data-copy]")) {
  button.addEventListener("click", async () => {
    const source = document.getElementById(button.dataset.copy);
    try {
      await navigator.clipboard.writeText(source.textContent);
      button.textContent = "Copied";
    } catch {
      getSelection().selectAllChildren(source);
      button.textContent = "Selected: press Ctrl/⌘ C";
    }
  });
}
