// Serve site/ for editing, and reload every open page when a file under site/ changes. reveal.js keeps the slide
// and fragment in the URL, so a talk reloads where it was.
//
//   node site/serve.mjs [port]        # default 8000; then open http://127.0.0.1:8000/
//
// Static files only, as the published page is. The reload hook is added to HTML responses here and nowhere else.

import { createReadStream, watch } from "node:fs";
import { stat } from "node:fs/promises";
import { createServer } from "node:http";
import { extname, join, normalize, sep } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = fileURLToPath(new URL(".", import.meta.url));
const PORT = Number(process.argv[2] ?? 8000);
const TYPES = {
  ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".mjs": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8", ".json": "application/json", ".svg": "image/svg+xml", ".png": "image/png",
  ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp", ".woff2": "font/woff2", ".onnx": "application/octet-stream",
  ".bin": "application/octet-stream", ".txt": "text/plain; charset=utf-8", ".md": "text/plain; charset=utf-8",
  ".mp4": "video/mp4", ".webm": "video/webm", ".pdf": "application/pdf",
};
const RELOAD = `<script>new EventSource("/__reload").addEventListener("change", () => location.reload());</script>`;
// Generated or bulky trees whose changes never need a reload.
const IGNORED = ["node_modules", ".screens", ".live-model", "data"];

const clients = new Set();
let pending = null;
watch(ROOT, { recursive: true }, (_event, file) => {
  if (!file || IGNORED.some((dir) => file.split(sep).includes(dir))) return;
  clearTimeout(pending);
  pending = setTimeout(() => {
    console.log(`changed: ${file}; reloading ${clients.size} page(s)`);
    for (const client of clients) client.write("event: change\ndata: \n\n");
  }, 120);
});

createServer(async (request, response) => {
  const path = decodeURIComponent(new URL(request.url, "http://localhost").pathname);
  if (path === "/__reload") {
    response.writeHead(200, { "content-type": "text/event-stream", "cache-control": "no-store", connection: "keep-alive" });
    response.write(": connected\n\n");
    clients.add(response);
    request.on("close", () => clients.delete(response));
    return;
  }
  let file = normalize(join(ROOT, path));
  if (!file.startsWith(ROOT)) return void response.writeHead(403).end();
  try {
    if ((await stat(file)).isDirectory()) file = join(file, "index.html");
    const info = await stat(file);
    const type = TYPES[extname(file).toLowerCase()] ?? "application/octet-stream";
    if (type.startsWith("text/html")) {
      const { readFile } = await import("node:fs/promises");
      const html = (await readFile(file, "utf8")).replace("</body>", `${RELOAD}</body>`);
      response.writeHead(200, { "content-type": type, "cache-control": "no-store" }).end(html);
      return;
    }
    response.writeHead(200, { "content-type": type, "content-length": info.size, "cache-control": "no-store" });
    createReadStream(file).pipe(response);
  } catch {
    response.writeHead(404, { "content-type": "text/plain" }).end(`not found: ${path}\n`);
    console.error(`404 ${path}`);
  }
}).listen(PORT, "127.0.0.1", () => console.log(`serving ${ROOT} at http://127.0.0.1:${PORT}/`));
