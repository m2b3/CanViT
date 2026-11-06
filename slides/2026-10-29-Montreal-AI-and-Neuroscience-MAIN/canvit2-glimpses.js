// <foveated-video> (#next): each frame of a video as CanViT 2's retina perceives it (rendered offline), unseen regions
// black, every sensor square outlined from per-frame geometry and the fixation ringed, looping. Data:
// data/talk/canvit2-glimpses/manifest.json. Follows the deck's data-play contract.

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`canvit2-glimpses: ${url} answered ${response.status}`);
  return response.json();
}

/** Defines <foveated-video> from the data at base (a URL ending in "/"). */
export async function drawCanvit2(base) {
  const manifest = await fetchJson(`${base}manifest.json`);
  const geometry = await fetchJson(base + manifest.canvit2.geometry);
  const frames = Array.from({ length: manifest.canvit2.count }, (_, i) =>
    Object.assign(new Image(), { src: base + manifest.canvit2.frames.replace("{i}", String(i).padStart(3, "0")) }));
  const fps = manifest.canvit2.fps;

  class FoveatedVideo extends HTMLElement {
    #timer = null;
    #index = 0;
    #canvas;

    connectedCallback() {
      this.#canvas = document.createElement("canvas");
      this.replaceChildren(this.#canvas);
      this.style.display = "contents";
      Promise.all(frames.map((frame) => frame.decode())).then(() => this.#draw(this.#index));
    }

    #draw(i) {
      this.#index = i;
      const canvas = this.#canvas;
      if (!canvas.offsetWidth) return; // a hidden slide: drawn when shown
      const scale = devicePixelRatio * Math.max(1, canvas.getBoundingClientRect().width / canvas.offsetWidth);
      const width = Math.round(canvas.offsetWidth * scale), height = Math.round(canvas.offsetHeight * scale);
      if (canvas.width !== width || canvas.height !== height) Object.assign(canvas, { width, height });
      const ctx = canvas.getContext("2d");
      const k = width / geometry.frame_hw[1], unit = width / canvas.offsetWidth; // scene px, one CSS px
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(frames[i], 0, 0, width, height);
      const { squares, gaze: [gy, gx] } = geometry.frames[i];
      // Every sensor square: a faint dark halo under a white line, legible on sky and ground alike.
      for (const [style, line] of [["rgb(0 0 0 / .35)", 2.2], ["rgb(255 255 255 / .65)", 1]]) {
        ctx.strokeStyle = style;
        ctx.lineWidth = line * unit;
        ctx.beginPath();
        for (const [y, x, side] of squares) ctx.rect(x * k, y * k, side * k, side * k);
        ctx.stroke();
      }
      for (const [style, line] of [["rgb(0 0 0 / .6)", 5], ["white", 2.5]]) {
        ctx.strokeStyle = style;
        ctx.lineWidth = line * unit;
        ctx.beginPath();
        ctx.arc(gx * k, gy * k, 9 * unit, 0, 2 * Math.PI);
        ctx.stroke();
      }
    }

    play() {
      if (this.#timer !== null) return;
      this.#draw(this.#index);
      this.#timer = setInterval(() => this.#draw((this.#index + 1) % frames.length), 1000 / fps);
    }

    pause() {
      clearInterval(this.#timer);
      this.#timer = null;
    }

    restart() {
      this.pause();
      this.#index = 0;
      this.play();
    }
  }
  if (!customElements.get("foveated-video")) customElements.define("foveated-video", FoveatedVideo);
}
