// <canvit-legend src t layer [max] [domain]>: how to read a map. Segmentation: the
// largest classes at glimpse t (at most `max`, default 5). Scalar maps: the colormap
// and its range (`domain` as on the map). RGB maps: what the colors are.

import { requireLayer } from "./bundle.js";
import { CanvitView, define, sheet } from "./view.js";
import { colormapGradient, layerSpec, paletteCss, scalarDomain } from "./layers.js";

const styles = sheet(`
  /* A fixed height keeps the layout still while the legend changes with t. */
  :host { height: var(--canvit-legend-height, 4.4em); overflow: hidden; font: 12px/1.4 var(--canvit-sans, system-ui, sans-serif); }
  .classes { display: flex; flex-wrap: wrap; gap: 2px 10px; }
  .class { display: inline-flex; align-items: center; gap: 5px; white-space: nowrap; }
  .swatch { width: 9px; height: 9px; border-radius: 2px; flex: none; }
  .more, .note, .range { color: var(--canvit-muted, color-mix(in srgb, currentColor 62%, transparent)); }
  .bar { height: 7px; border-radius: 4px; }
  .ends { display: flex; justify-content: space-between; gap: 8px; margin-top: 3px; }
  .range { margin-top: 1px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
           font-family: var(--canvit-mono, ui-monospace, SFMono-Regular, Menlo, monospace); font-size: 10.5px; }
`);

class CanvitLegend extends CanvitView {
  static observedAttributes = [...CanvitView.observedAttributes, "layer", "max", "domain"];
  static styles = [styles];

  attributeChangedCallback(name, old, value) {
    if (name === "layer" && old !== value) this.invalidate();
    super.attributeChangedCallback(name, old, value);
  }

  build(bundle) {
    this.spec = layerSpec(requireLayer(bundle, this.getAttribute("layer")));
    this.body = document.createElement("div");
    this.shadowRoot.replaceChildren(this.body);
    this.shown = null;
    if (this.spec.kind === "rgb") {
      const note = this.spec.note(bundle);
      if (!note) throw new Error(`no description of the "${this.getAttribute("layer")}" colors for this bundle's manifest`);
      this.body.className = "note";
      this.body.textContent = note;
    }
  }

  render(t) {
    const name = this.getAttribute("layer");
    const { spec, bundle } = this;
    const key = `${t}|${this.getAttribute("max")}|${this.getAttribute("domain")}`;
    if (key === this.shown) return; // pointer moves need no redraw
    this.shown = key;
    if (spec.kind === "labels") {
      const data = bundle.glimpses[t].layers[name].data;
      const counts = new Map();
      for (const c of data) counts.set(c, (counts.get(c) ?? 0) + 1);
      const ranked = [...counts.keys()].sort((a, b) => counts.get(b) - counts.get(a));
      const max = Number(this.getAttribute("max") ?? 5);
      if (!Number.isInteger(max) || max < 1) throw new Error(`max="${this.getAttribute("max")}" is not a positive integer`);
      const chips = ranked.slice(0, max).map((c) => {
        const chip = document.createElement("span");
        chip.className = "class";
        const swatch = document.createElement("span");
        swatch.className = "swatch";
        swatch.style.background = paletteCss(c);
        chip.append(swatch, bundle.classNames[c]);
        return chip;
      });
      if (ranked.length > max) {
        const more = document.createElement("span");
        more.className = "more";
        more.textContent = `+${ranked.length - max} more`;
        chips.push(more);
      }
      this.body.className = "classes";
      this.body.replaceChildren(...chips);
    } else if (spec.kind === "scalar") {
      const domain = this.getAttribute("domain") ?? spec.domain;
      const [lo, hi] = scalarDomain(bundle, t, name, domain);
      const fmt = (f) => spec.value(f).toFixed(spec.value(hi) < 0.1 ? 3 : 2);
      const quantity = spec.quantity(bundle);
      this.body.innerHTML = `
        <div class="bar" style="background:${colormapGradient(spec.colormap)}"></div>
        <div class="ends"><span>${spec.low}</span><span>${spec.high}</span></div>
        <div class="range" title="${quantity}: ${fmt(lo)} to ${fmt(hi)}">${quantity} ${fmt(lo)}–${fmt(hi)}</div>`;
    }
  }
}

define("canvit-legend", CanvitLegend);
