// <deck-sequence>: shows its children one at a time, in order, then holds the last and starts over. With data-play
// (deck.js) it plays while its slide is shown. Attributes: period (ms per child, default 1400), fast (ms per child at
// the end: the pace speeds up geometrically from period to fast, slow then fast; default: period throughout), hold (ms
// on the last child, default 3000). Every child is laid out in the same place; only the current one is visible. Each
// change dispatches deck-sequence-change with detail {index, count}.

class DeckSequence extends HTMLElement {
  #timer = null;
  #index = 0;

  connectedCallback() {
    if (this.children.length === 0) throw new Error("<deck-sequence> has no children to show");
    this.#show(0);
  }

  #ms(name, fallback) {
    const value = Number(this.getAttribute(name) ?? fallback);
    if (!(value > 0)) throw new Error(`<deck-sequence ${name}="${this.getAttribute(name)}">: expected a positive number`);
    return value;
  }

  #show(index) {
    this.#index = index;
    [...this.children].forEach((child, i) => child.toggleAttribute("data-current", i === index));
    this.dispatchEvent(new CustomEvent("deck-sequence-change", { detail: { index, count: this.children.length } }));
  }

  /** The time on child `index` before the next: period at the start, fast at the end, geometric in between. */
  #period(index) {
    const period = this.#ms("period", 1400);
    if (!this.hasAttribute("fast")) return period;
    const fraction = index / Math.max(1, this.children.length - 2);
    return period * (this.#ms("fast", period) / period) ** fraction;
  }

  #tick = () => {
    const last = this.children.length - 1;
    const wait = this.#index === last ? this.#ms("hold", 3000) : this.#period(this.#index);
    this.#timer = setTimeout(() => {
      this.#show(this.#index === last ? 0 : this.#index + 1);
      this.#tick();
    }, wait);
  };

  play() {
    if (this.#timer === null) this.#tick();
  }

  pause() {
    clearTimeout(this.#timer);
    this.#timer = null;
  }

  restart() {
    this.pause();
    this.#show(0);
    this.play();
  }
}

customElements.define("deck-sequence", DeckSequence);
