// <canvit-scene src t [trail="past|none"]>: the photograph with the current glimpse's
// box (animated from the previous one), earlier glimpses' boxes (trail="past", the
// default), and the region outside the glimpse dimmed.

import { CanvitView, define, frameCss, place, sheet } from "./view.js";

const TRAILS = ["past", "none"];

const styles = sheet(`
  ${frameCss}
  :host { aspect-ratio: 1; }
  img { object-fit: cover; user-select: none; -webkit-user-drag: none; }
  .past { border: 1px dashed rgb(255 255 255 / .7); opacity: 0; transition: opacity 300ms; }
  .past.seen { opacity: 1; }
  .current {
    border: 2px solid var(--canvit-glimpse, #4080d0);
    box-shadow: 0 0 0 1px rgb(0 0 0 / .55), 0 0 0 200vmax rgb(8 6 16 / var(--canvit-dim, .42));
  }
  .tag {
    position: absolute; left: -2px; bottom: 100%; padding: 1px 6px 2px;
    background: var(--canvit-glimpse, #4080d0); color: #fff; border-radius: 4px 4px 4px 0;
    font: 600 11px/1.3 var(--canvit-mono, ui-monospace, SFMono-Regular, Menlo, monospace);
    white-space: nowrap;
  }
  .current.top .tag { bottom: auto; top: 0; border-radius: 0 0 4px 0; }
`);

class CanvitScene extends CanvitView {
  static observedAttributes = [...CanvitView.observedAttributes, "trail"];
  static styles = [styles];

  get trail() {
    const trail = this.getAttribute("trail") ?? "past";
    if (!TRAILS.includes(trail)) throw new Error(`trail="${trail}" is not one of ${TRAILS.join(", ")}`);
    return trail;
  }

  build(bundle) {
    const frame = document.createElement("div");
    frame.className = "frame";
    const img = new Image();
    img.src = bundle.sceneUrl;
    img.alt = `${bundle.manifest.title}: the scene, with the model's glimpses outlined`;
    img.draggable = false;
    this.pastBoxes = bundle.glimpses.map((g) => {
      const box = document.createElement("div");
      box.className = "box past";
      place(box, g.box);
      return box;
    });
    this.current = document.createElement("div");
    this.current.className = "box current";
    this.tag = document.createElement("span");
    this.tag.className = "tag";
    this.current.append(this.tag);
    this.marker = document.createElement("div");
    this.marker.className = "pointer";
    frame.append(img, ...this.pastBoxes, this.current, this.marker);
    this.shadowRoot.replaceChildren(frame);
    this.trackPointer(frame);
  }

  render(t) {
    const { box } = this.bundle.glimpses[t];
    const trail = this.trail === "past";
    this.pastBoxes.forEach((el, i) => el.classList.toggle("seen", trail && i < t));
    place(this.current, box);
    this.current.classList.toggle("top", box.top < 0.06);
    this.tag.textContent = `t = ${t}`;
    const p = this.pointer;
    this.marker.hidden = !p;
    if (p) Object.assign(this.marker.style, { left: `${100 * p.x}%`, top: `${100 * p.y}%` });
  }
}

define("canvit-scene", CanvitScene);
