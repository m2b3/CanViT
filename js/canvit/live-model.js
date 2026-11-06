// CanViT-B and its ADE20K probe, one glimpse per call, in ONNX Runtime Web: the CanViT graph, then the probe graph on
// the new canvas. Both directories are written by `python -m canvit_pytorch.viz.live export`, and published apart
// (schemas below; canvit-pytorch/docs/viz.md, "Live model"). On WebGPU the scene, canvas and recurrent CLS token live
// in fixed GPU buffers: each step writes the next state to output buffers, which a GPU copy moves onto the input
// buffers, and the probe reads the new canvas there; only the maps shown are read back.

const CANVIT_SCHEMA = "canvit-live-canvit-0a49b16e-2f86-41ae-8b08-f83d94fb1732";
const PROBE_SCHEMA = "canvit-live-probe-23117c20-1e8e-4edb-a56d-21018d157b8d";
// The published exports, canvit_pytorch.hub.repos.LIVE_CANVIT and LIVE_PROBE on the Hub
// (`python -m canvit_pytorch.viz.live publish`).
export const PUBLISHED_CANVIT =
  "https://huggingface.co/canvit/canvitb16-add-vpe-pretrain-g128px-s512px-in21k-dv3b16-2026-02-02-c64-onnx-fp32/resolve/main";
export const PUBLISHED_PROBE = "https://huggingface.co/canvit/probe-ade20k-40k-s512-c64-in21k-onnx-fp32/resolve/main";
// The exported graph relies on this version's WebGPU kernels (docs/viz.md, "Live model"): after changing it,
// rerun `canvit_pytorch.viz.live check-browser` on both backends.
const ORT_VERSION = "1.30.0";

// The ONNX Runtime Web build that carries each backend. ort.webgpu is ORT's C++ WebGPU execution provider
// compiled to WebAssembly; the JavaScript kernels of ort.all (1.30.0) rejected this graph's GridSample.
const BACKENDS = {
  webgpu: { build: "ort.webgpu.min.mjs", name: "WebGPU" },
  wasm: { build: "ort.wasm.min.mjs", name: "WebAssembly" },
};

const loadOrt = (backend) =>
  import(`https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/${BACKENDS[backend].build}`);

const numel = (dims) => dims.reduce((a, b) => a * b, 1);

// Verified files stay in the browser's Cache Storage under their URL and SHA-256, so a later page load reads them from
// disk; a new export at the same URL has new hashes, misses, downloads once and replaces the old entries. Entries of
// URLs no page loads any more stay until the browser evicts them. Manifests are fetched from the server on every load
// and stored too: offline, the stored ones start the cached files. Without Cache Storage (an insecure origin), every
// load downloads.
const CACHE_NAME = "canvit-live-files";

let opened = null; // the page's cache, or null without one, opened once

const openCache = () => (opened ??= (async () => {
  try {
    if (!globalThis.caches) throw new Error("this origin has no Cache Storage");
    return await caches.open(CACHE_NAME);
  } catch (error) {
    console.warn("CanViT live model: downloading on every load", error);
    return null;
  }
})());

function cacheKey(url, sha256) {
  const key = new URL(url);
  key.searchParams.set("sha256", sha256);
  return key.href;
}

async function fetchOk(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Could not fetch ${url}: HTTP ${response.status}`);
  return response;
}

/** The body's bytes, reporting the count received as they arrive. */
async function readBody(response, onProgress) {
  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    onProgress(received);
  }
  const bytes = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.length; }
  return bytes;
}

async function sha256Hex(bytes) {
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
  return Array.from(digest, (b) => b.toString(16).padStart(2, "0")).join("");
}

async function verify(url, file, bytes) {
  if (bytes.length !== file.bytes) throw new Error(`${url}: ${bytes.length} bytes, the manifest says ${file.bytes}`);
  if ((await sha256Hex(bytes)) !== file.sha256) throw new Error(`${url}: SHA-256 differs from the manifest`);
  return bytes;
}

// Removes the entries of other versions of url before adding this one, which frees their quota first.
async function store(cache, url, file, bytes) {
  try {
    for (const request of await cache.keys()) {
      const key = new URL(request.url);
      key.searchParams.delete("sha256");
      if (key.href === url.href) await cache.delete(request);
    }
    await cache.put(cacheKey(url, file.sha256), new Response(bytes));
  } catch (error) {
    console.warn(`CanViT live model: could not cache ${url}; the next load downloads it again`, error);
  }
}

/** A manifest's file, read from the cache when an earlier load stored it, else downloaded; checked against the manifest either way. */
async function verifiedBytes(cache, { base, file }, onProgress) {
  const url = new URL(file.path, base);
  const cached = await cache?.match(cacheKey(url, file.sha256));
  if (cached) {
    try {
      return await verify(url, file, await readBody(cached, onProgress));
    } catch (error) {
      console.warn(`CanViT live model: the cached ${url} is damaged; downloading it again`, error);
      await cache.delete(cacheKey(url, file.sha256));
    }
  }
  const bytes = await verify(url, file, await readBody(await fetchOk(url), onProgress));
  if (cache) await store(cache, url, file, bytes);
  return bytes;
}

// The server's manifest, stored for offline loads; the stored one only when the network fails, not the server.
async function fetchManifest(cache, url) {
  let response;
  try {
    response = await fetch(url, { cache: "no-cache" });
  } catch (error) {
    const stored = await cache?.match(url);
    if (!stored) throw new Error(`Could not fetch ${url}: ${error.message}`);
    console.warn(`CanViT live model: ${url} is unreachable; using the manifest stored by an earlier load`, error);
    return stored.json();
  }
  if (!response.ok) throw new Error(`Could not fetch ${url}: HTTP ${response.status}`);
  const text = await response.text();
  try {
    await cache?.put(url, new Response(text, { headers: { "content-type": "application/json" } }));
  } catch (error) {
    console.warn(`CanViT live model: could not store ${url} for offline loads`, error);
  }
  return JSON.parse(text);
}

async function loadManifest(cache, url, schema) {
  const base = new URL(url.endsWith("/") ? url : `${url}/`, document.baseURI);
  const manifest = await fetchManifest(cache, new URL("manifest.json", base));
  if (manifest.schema !== schema) throw new Error(`${base}manifest.json: schema ${manifest.schema}, expected ${schema}`);
  return { base, manifest };
}

/** The manifests of a CanViT export and a probe export (URLs relative to the page), checked to fit together. */
export async function loadManifests(canvitUrl, probeUrl) {
  const cache = await openCache();
  const [canvit, probe] = await Promise.all([loadManifest(cache, canvitUrl, CANVIT_SCHEMA), loadManifest(cache, probeUrl, PROBE_SCHEMA)]);
  const a = canvit.manifest, b = probe.manifest;
  if (JSON.stringify(b.graph.inputs.canvas) !== JSON.stringify(a.graph.outputs.next_canvas)
      || b.canvas_grid !== a.canvas_grid || b.num_canvas_registers !== a.num_canvas_registers) {
    throw new Error(`${probe.base} reads a ${b.canvas_grid}² canvas ${JSON.stringify(b.graph.inputs.canvas)}; ` +
                    `${canvit.base} writes a ${a.canvas_grid}² one ${JSON.stringify(a.graph.outputs.next_canvas)}`);
  }
  return { canvit, probe };
}

// The files to fetch before the first glimpse.
const files = ({ canvit, probe }) => ({
  initialState: { base: canvit.base, file: canvit.manifest.initial_state },
  canvit: { base: canvit.base, file: canvit.manifest.graph },
  probe: { base: probe.base, file: probe.manifest.graph },
});

/** Bytes to download before the first glimpse: the files this browser has not cached. */
export async function downloadBytes(manifests) {
  const cache = await openCache();
  let bytes = 0;
  for (const { base, file } of Object.values(files(manifests))) {
    if (!(await cache?.match(cacheKey(new URL(file.path, base), file.sha256)))) bytes += file.bytes;
  }
  return bytes;
}

// The registers, then the canvas patch broadcast over the grid, as CanViT.init_state builds them.
function initialState(manifest, flat) {
  const [registers, patch, recurrentCls] = manifest.initial_state.parts.map((part) => numel(part.shape));
  if (flat.length !== registers + patch + recurrentCls) throw new Error(`initial state has ${flat.length} floats`);
  const cells = manifest.canvas_grid ** 2;
  const canvas = new Float32Array(registers + cells * patch);
  canvas.set(flat.subarray(0, registers));
  const patchInit = flat.subarray(registers, registers + patch);
  for (let i = 0; i < cells; i++) canvas.set(patchInit, registers + i * patch);
  return { canvas, recurrentCls: flat.slice(registers + patch) };
}

// A float32 tensor in a fixed GPU buffer. Sizes round up to 16 bytes: ORT's shaders bind at least that much,
// and the one-float `scales` input would fail validation.
class GpuTensor {
  constructor(ort, device, dims) {
    this.device = device;
    this.bytes = 4 * numel(dims);
    this.buffer = device.createBuffer({
      size: Math.ceil(this.bytes / 16) * 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this.tensor = ort.Tensor.fromGpuBuffer(this.buffer, { dataType: "float32", dims });
  }

  write(data) { this.device.queue.writeBuffer(this.buffer, 0, data); }
}

/** Copy GPU tensors to one staging buffer and map it once; resolves to one Float32Array per tensor. */
async function readBack(device, tensors) {
  const offsets = [];
  let size = 0;
  for (const t of tensors) { offsets.push(size); size += Math.ceil(t.bytes / 16) * 16; }
  const staging = device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const encoder = device.createCommandEncoder();
  tensors.forEach((t, i) => encoder.copyBufferToBuffer(t.buffer, 0, staging, offsets[i], t.bytes));
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const mapped = staging.getMappedRange();
  const out = tensors.map((t, i) => new Float32Array(mapped.slice(offsets[i], offsets[i] + t.bytes)));
  staging.destroy();
  return out;
}

async function webgpuAvailable() {
  try {
    return Boolean(await navigator.gpu?.requestAdapter());
  } catch {
    return false;
  }
}

export class LiveModel {
  /**
   * Fetch (from the browser's cache or the network), verify and start the CanViT export at canvitUrl and the probe
   * export at probeUrl: WebGPU when the browser has an adapter, WebAssembly otherwise. onProgress(stage, received,
   * total) reports the stages "fetch", with the bytes received of the total, then "start".
   * `fallback` names why WebGPU was not used, if it failed.
   */
  static async load(canvitUrl, probeUrl, { onProgress = () => {} } = {}) {
    const manifests = await loadManifests(canvitUrl, probeUrl);
    const cache = await openCache();
    const parts = Object.entries(files(manifests));
    const total = parts.reduce((sum, [, { file }]) => sum + file.bytes, 0);
    const received = Object.fromEntries(parts.map(([name]) => [name, 0]));
    const progress = (name) => (bytes) => {
      received[name] = bytes;
      onProgress("fetch", Object.values(received).reduce((a, b) => a + b, 0), total);
    };
    const fetched = Object.fromEntries(await Promise.all(
      parts.map(async ([name, part]) => [name, await verifiedBytes(cache, part, progress(name))])));
    const init = initialState(manifests.canvit.manifest, new Float32Array(fetched.initialState.buffer));
    const graphs = { canvit: fetched.canvit, probe: fetched.probe };
    let fallback = null;
    if (await webgpuAvailable()) {
      onProgress("start", 0, 0);
      try {
        return await LiveModel.#start("webgpu", manifests, graphs, init, null);
      } catch (error) {
        console.error("CanViT on WebGPU failed; using WebAssembly", error);
        fallback = `WebGPU failed: ${error.message}`;
      }
    }
    onProgress("start", 0, 0);
    return LiveModel.#start("wasm", manifests, graphs, init, fallback);
  }

  static async #start(backend, manifests, graphs, init, fallback) {
    const ort = await loadOrt(backend);
    const options = { executionProviders: [backend], graphOptimizationLevel: "all" };
    if (backend === "webgpu") options.preferredOutputLocation = "gpu-buffer";
    const started = performance.now();
    const step = await ort.InferenceSession.create(graphs.canvit, options);
    const probe = await ort.InferenceSession.create(graphs.probe, options);
    const model = new LiveModel(ort, backend, manifests, { step, probe }, init, fallback);
    model.sessionCreateMs = performance.now() - started;
    if (backend === "webgpu") await model.#allocateGpu();
    model.reset();
    return model;
  }

  #ort; #init; #gpu = null; #cpu = null; #scene = null;

  constructor(ort, backend, manifests, sessions, init, fallback) {
    this.#ort = ort;
    this.#init = init;
    this.backend = backend;
    this.backendName = BACKENDS[backend].name;
    this.fallback = fallback;
    this.manifest = manifests.canvit.manifest;
    this.probeManifest = manifests.probe.manifest;
    this.sessions = sessions;
  }

  async #allocateGpu() {
    const device = await this.#ort.env.webgpu.device;
    const make = (dims) => new GpuTensor(this.#ort, device, dims);
    const { inputs, outputs } = this.manifest.graph;
    this.#gpu = {
      device,
      inputs: Object.fromEntries(Object.entries(inputs).map(([name, dims]) => [name, make(dims)])),
      outputs: Object.fromEntries(Object.entries(outputs).map(([name, dims]) => [name, make(dims)])),
      readout: Object.fromEntries(Object.entries(this.probeManifest.graph.outputs).map(([name, dims]) => [name, make(dims)])),
      initCanvas: make(inputs.canvas),
      initCls: make(inputs.recurrent_cls),
    };
    this.#gpu.initCanvas.write(this.#init.canvas);
    this.#gpu.initCls.write(this.#init.recurrentCls);
  }

  #copy(pairs) {
    const encoder = this.#gpu.device.createCommandEncoder();
    for (const [src, dst] of pairs) encoder.copyBufferToBuffer(src.buffer, 0, dst.buffer, 0, src.bytes);
    this.#gpu.device.queue.submit([encoder.finish()]);
  }

  /** scene: Float32Array [3, S, S], normalized as manifest.normalization says. */
  setScene(scene) {
    if (scene.length !== numel(this.manifest.graph.inputs.scene)) throw new Error(`scene has ${scene.length} values`);
    if (this.#gpu) this.#gpu.inputs.scene.write(scene);
    else this.#scene = new this.#ort.Tensor("float32", scene, this.manifest.graph.inputs.scene);
  }

  /** Back to the state before any glimpse. */
  reset() {
    if (this.#gpu) {
      const { initCanvas, initCls, inputs } = this.#gpu;
      this.#copy([[initCanvas, inputs.canvas], [initCls, inputs.recurrent_cls]]);
    } else {
      const dims = this.manifest.graph.inputs;
      this.#cpu = {
        canvas: new this.#ort.Tensor("float32", this.#init.canvas, dims.canvas),
        cls: new this.#ort.Tensor("float32", this.#init.recurrentCls, dims.recurrent_cls),
      };
    }
  }

  /**
   * One glimpse at viewpoint {row, col, scale}. Resolves to the logits [C, G²] (class-major), the entropy [G²]
   * in nats and the glimpse [3, g²], all as Float32Array, with runMs, the time session.run took to return.
   */
  async step({ row, col, scale }) {
    const centers = Float32Array.of(row, col);
    const scales = Float32Array.of(scale);
    if (this.#gpu) {
      const { inputs, outputs, readout, device } = this.#gpu;
      inputs.centers.write(centers);
      inputs.scales.write(scales);
      const tensors = (buffers) => Object.fromEntries(Object.entries(buffers).map(([name, t]) => [name, t.tensor]));
      const started = performance.now();
      await this.sessions.step.run(tensors(inputs), tensors(outputs));
      this.#copy([[outputs.next_canvas, inputs.canvas], [outputs.next_recurrent_cls, inputs.recurrent_cls]]);
      await this.sessions.probe.run({ canvas: inputs.canvas.tensor }, tensors(readout));
      const runMs = performance.now() - started;
      const [logits, entropy, glimpse] = await readBack(device, [readout.logits, readout.entropy, outputs.glimpse]);
      return { logits, entropy, glimpse, runMs };
    }
    const ort = this.#ort;
    const started = performance.now();
    const out = await this.sessions.step.run({
      scene: this.#scene, canvas: this.#cpu.canvas, recurrent_cls: this.#cpu.cls,
      centers: new ort.Tensor("float32", centers, [1, 2]), scales: new ort.Tensor("float32", scales, [1]),
    });
    const read = await this.sessions.probe.run({ canvas: out.next_canvas });
    const runMs = performance.now() - started;
    this.#cpu = { canvas: out.next_canvas, cls: out.next_recurrent_cls };
    return { logits: read.logits.data, entropy: read.entropy.data, glimpse: out.glimpse.data, runMs };
  }

  /** The current canvas [R + G², D], for checks against PyTorch. */
  async readCanvas() {
    if (this.#gpu) return (await readBack(this.#gpu.device, [this.#gpu.inputs.canvas]))[0];
    return this.#cpu.canvas.data;
  }
}

/**
 * The image as the model sees it: the short side resized to `size` and the center cropped, as
 * canvit_pytorch.preprocess.preprocess(size) does with the browser's resampler in place of PIL's.
 * Returns the cropped picture (a canvas, for display) and the normalized [3, size, size] tensor.
 */
export function sceneFromImage(image, { size, mean, std }) {
  const width = image.naturalWidth ?? image.width;
  const height = image.naturalHeight ?? image.height;
  const side = Math.min(width, height);
  const picture = Object.assign(document.createElement("canvas"), { width: size, height: size });
  const ctx = picture.getContext("2d", { willReadFrequently: true });
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(image, (width - side) / 2, (height - side) / 2, side, side, 0, 0, size, size);
  const { data } = ctx.getImageData(0, 0, size, size);
  const n = size * size;
  const tensor = new Float32Array(3 * n);
  for (let c = 0; c < 3; c++) {
    for (let i = 0; i < n; i++) tensor[c * n + i] = (data[4 * i + c] / 255 - mean[c]) / std[c];
  }
  return { picture, tensor };
}
