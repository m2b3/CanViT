// <deck-sequence>: shows its children one at a time, in order, then holds the last and starts over. With data-play
// (deck.js) it plays while its slide is shown. Attributes: period (ms per child, default 1400), hold (ms on the last
// child, default 3000). Every child is laid out in the same place; only the current one is visible.

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
  }

  #tick = () => {
    const last = this.children.length - 1;
    const wait = this.#index === last ? this.#ms("hold", 3000) : this.#ms("period", 1400);
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
