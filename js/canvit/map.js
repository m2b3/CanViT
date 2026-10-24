// <canvit-map src t layer="canvas|labels|entropy|change|crop|writeK" [domain="unit|frame|bundle"]>:
// one layer at glimpse t, upscaled with nearest-neighbor pixels. Canvas-grid layers
// outline the current glimpse; the segmentation names the class under the pointer.

import { CanvitView, define, frameCss, place, sheet } from "./view.js";
import { requireLayer } from "./bundle.js";
import { DOMAINS, layerSpec, paletteCss, renderLayer } from "./layers.js";

const styles = sheet(`
  ${frameCss}
  :host { aspect-ratio: 1; }
  .glimpse { border: 1.5px dashed var(--canvit-glimpse, #4080d0); filter: drop-shadow(0 0 1px rgb(0 0 0 / .8)); }
  .tip {
    position: absolute; transform: translate(-50%, calc(-100% - 12px)); pointer-events: none;
    padding: 3px 8px; border-radius: 6px; white-space: nowrap; z-index: 1;
    background: rgb(12 10 20 / .88); color: #fff;
    font: 600 12px/1.35 var(--canvit-sans, system-ui, sans-serif);
  }
  .tip.below { transform: translate(-50%, 14px); }
  .tip.left { translate: -40% 0; }
  .tip.right { translate: 40% 0; }
  .swatch { display: inline-block; width: .8em; height: .8em; margin-right: 6px; border-radius: 2px;
            vertical-align: -1px; box-shadow: 0 0 0 1px rgb(255 255 255 / .6); }
`);

class CanvitMap extends CanvitView {
  static observedAttributes = [...CanvitView.observedAttributes, "layer", "domain"];
  static styles = [styles];

  attributeChangedCallback(name, old, value) {
    if (name === "layer" && old !== value) this.invalidate();
    super.attributeChangedCallback(name, old, value);
  }

  get layer() { return requireLayer(this.bundle, this.getAttribute("layer")); }

  get domain() {
    const domain = this.getAttribute("domain");
    if (domain !== null && !DOMAINS.includes(domain)) {
      throw new Error(`domain="${domain}" is not one of ${DOMAINS.join(", ")}`);
    }
    return domain ?? undefined;
  }

  build(bundle) {
    const name = this.layer;
    this.spec = layerSpec(name);
    const side = bundle.sides[this.spec.size];
    const frame = document.createElement("div");
    frame.className = "frame";
    this.canvas = Object.assign(document.createElement("canvas"), { width: side, height: side, className: "pixels" });
    this.canvas.setAttribute("role", "img");
    this.context = this.canvas.getContext("2d");
    this.glimpse = document.createElement("div");
    this.glimpse.className = "box glimpse";
    this.glimpse.hidden = this.spec.size !== "canvas";
    this.marker = document.createElement("div");
    this.marker.className = "pointer";
    this.tip = document.createElement("div");
    this.tip.className = "tip";
    frame.append(this.canvas, this.glimpse, this.marker, this.tip);
    this.shadowRoot.replaceChildren(frame);
    this.drawn = null;
    // Glimpse and patch layers cover only the glimpse's box of the scene.
    this.trackPointer(frame, () => (this.spec.size === "canvas" ? undefined : this.bundle.glimpses[this.t].box));
  }

  render(t) {
    const { bundle, spec } = this;
    const name = this.layer;
    const box = spec.size === "canvas" ? { top: 0, left: 0, size: 1 } : bundle.glimpses[t].box;
    const p = this.pointer;
    // Pointer position within this map, if the map shows that part of the scene.
    const local = p && {
      u: (p.x - box.left) / box.size, v: (p.y - box.top) / box.size,
    };
    const inside = local && local.u >= 0 && local.u < 1 && local.v >= 0 && local.v < 1;

    let highlight = null;
    if (spec.kind === "labels" && inside) {
      const raster = bundle.glimpses[t].layers[name];
      highlight = raster.data[Math.floor(local.v * raster.height) * raster.width + Math.floor(local.u * raster.width)];
    }
    const key = `${t}|${name}|${this.domain}|${highlight}`;
    if (key !== this.drawn) {
      this.context.putImageData(renderLayer(bundle, t, name, { domain: this.domain, highlight }), 0, 0);
      this.drawn = key;
      this.canvas.setAttribute("aria-label", `${spec.title} at glimpse ${t}`);
    }

    if (spec.size === "canvas") place(this.glimpse, bundle.glimpses[t].box);
    this.marker.hidden = !inside;
    this.tip.hidden = highlight === null;
    if (inside) {
      const at = { left: `${100 * local.u}%`, top: `${100 * local.v}%` };
      Object.assign(this.marker.style, at);
      Object.assign(this.tip.style, at);
      this.tip.classList.toggle("below", local.v < 0.2);
      this.tip.classList.toggle("left", local.u > 0.75);
      this.tip.classList.toggle("right", local.u < 0.25);
    }
    if (highlight !== null) {
      const swatch = document.createElement("span");
      swatch.className = "swatch";
      swatch.style.background = paletteCss(highlight);
      this.tip.replaceChildren(swatch, bundle.classNames[highlight]);
    }
  }
}

define("canvit-map", CanvitMap);
