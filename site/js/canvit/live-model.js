// CanViT-B with its ADE20K probe, one glimpse per call, in ONNX Runtime Web. The model directory is written by
// `python -m canvit_pytorch.viz.live export` (schema below; canvit-pytorch/docs/viz.md, "Live model").
// On WebGPU the scene, canvas and recurrent CLS token live in fixed GPU buffers: each step writes the next state
// to output buffers, which a GPU copy moves onto the input buffers; only the maps shown are read back.

const SCHEMA = "canvit-live-model-2ef08388-92e1-4d7d-b5ac-2922601b5aa0";
// The published export, canvit_pytorch.hub.repos.LIVE_MODEL on the Hub (`python -m canvit_pytorch.viz.live publish`).
export const PUBLISHED_MODEL = "https://huggingface.co/canvit/canvitb16-in21k-ade20k-s512-c64-onnx-fp32/resolve/main";
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

async function fetchOk(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Could not fetch ${url}: HTTP ${response.status}`);
  return response;
}

/** The bytes at url, reporting (received, total) as they arrive. */
async function fetchBytes(url, onProgress = () => {}) {
  const response = await fetchOk(url);
  const total = Number(response.headers.get("content-length")) || 0;
  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    onProgress(received, total);
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

async function verifiedBytes(base, file, onProgress) {
  const url = new URL(file.path, base);
  const bytes = await fetchBytes(url, onProgress);
  if (bytes.length !== file.bytes) throw new Error(`${url}: ${bytes.length} bytes, the manifest says ${file.bytes}`);
  if ((await sha256Hex(bytes)) !== file.sha256) throw new Error(`${url}: SHA-256 differs from the manifest`);
  return bytes;
}

/** The manifest of the model directory at url (relative to the page). */
export async function loadManifest(url) {
  const base = new URL(url.endsWith("/") ? url : `${url}/`, document.baseURI);
  const manifest = await (await fetchOk(new URL("manifest.json", base))).json();
  if (manifest.schema !== SCHEMA) throw new Error(`${base}manifest.json: schema ${manifest.schema}, expected ${SCHEMA}`);
  return { base, manifest };
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
   * Download, verify and start the model at url: WebGPU when the browser has an adapter, WebAssembly otherwise.
   * onProgress(stage, received, total) reports the download. `fallback` names why WebGPU was not used, if it failed.
   */
  static async load(url, { onProgress = () => {} } = {}) {
    const { base, manifest } = await loadManifest(url);
    const [stateBytes, graphBytes] = await Promise.all([
      verifiedBytes(base, manifest.initial_state),
      verifiedBytes(base, manifest.graph, (received, total) => onProgress("download", received, total)),
    ]);
    const init = initialState(manifest, new Float32Array(stateBytes.buffer));
    let fallback = null;
    if (await webgpuAvailable()) {
      onProgress("start", 0, 0);
      try {
        return await LiveModel.#start("webgpu", manifest, graphBytes, init, null);
      } catch (error) {
        console.error("CanViT on WebGPU failed; using WebAssembly", error);
        fallback = `WebGPU failed: ${error.message}`;
      }
    }
    onProgress("start", 0, 0);
    return LiveModel.#start("wasm", manifest, graphBytes, init, fallback);
  }

  static async #start(backend, manifest, graphBytes, init, fallback) {
    const ort = await loadOrt(backend);
    const options = { executionProviders: [backend], graphOptimizationLevel: "all" };
    if (backend === "webgpu") options.preferredOutputLocation = "gpu-buffer";
    const started = performance.now();
    const session = await ort.InferenceSession.create(graphBytes, options);
    const model = new LiveModel(ort, backend, manifest, session, init, fallback);
    model.sessionCreateMs = performance.now() - started;
    if (backend === "webgpu") await model.#allocateGpu();
    model.reset();
    return model;
  }

  #ort; #init; #gpu = null; #cpu = null; #scene = null;

  constructor(ort, backend, manifest, session, init, fallback) {
    this.#ort = ort;
    this.#init = init;
    this.backend = backend;
    this.backendName = BACKENDS[backend].name;
    this.fallback = fallback;
    this.manifest = manifest;
    this.session = session;
  }

  async #allocateGpu() {
    const device = await this.#ort.env.webgpu.device;
    const make = (dims) => new GpuTensor(this.#ort, device, dims);
    const { inputs, outputs } = this.manifest.graph;
    this.#gpu = {
      device,
      inputs: Object.fromEntries(Object.entries(inputs).map(([name, dims]) => [name, make(dims)])),
      outputs: Object.fromEntries(Object.entries(outputs).map(([name, dims]) => [name, make(dims)])),
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
      const { inputs, outputs, device } = this.#gpu;
      inputs.centers.write(centers);
      inputs.scales.write(scales);
      const feeds = Object.fromEntries(Object.entries(inputs).map(([name, t]) => [name, t.tensor]));
      const fetches = Object.fromEntries(Object.entries(outputs).map(([name, t]) => [name, t.tensor]));
      const started = performance.now();
      await this.session.run(feeds, fetches);
      const runMs = performance.now() - started;
      this.#copy([[outputs.next_canvas, inputs.canvas], [outputs.next_recurrent_cls, inputs.recurrent_cls]]);
      const [logits, entropy, glimpse] = await readBack(device, [outputs.logits, outputs.entropy, outputs.glimpse]);
      return { logits, entropy, glimpse, runMs };
    }
    const ort = this.#ort;
    const started = performance.now();
    const out = await this.session.run({
      scene: this.#scene, canvas: this.#cpu.canvas, recurrent_cls: this.#cpu.cls,
      centers: new ort.Tensor("float32", centers, [1, 2]), scales: new ort.Tensor("float32", scales, [1]),
    });
    const runMs = performance.now() - started;
    this.#cpu = { canvas: out.next_canvas, cls: out.next_recurrent_cls };
    return { logits: out.logits.data, entropy: out.entropy.data, glimpse: out.glimpse.data, runMs };
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
