// Exact PNG decoding for bundle layers. Canvas readback (drawImage + getImageData)
// is not exact: browsers may color-manage the image, and anti-fingerprinting
// (Brave, Safari private browsing, Firefox resistFingerprinting) perturbs or
// blanks the pixels, which would silently change class indices.

const SIGNATURE = [137, 80, 78, 71, 13, 10, 26, 10];
const CHANNELS = { 0: 1, 2: 3, 6: 4 }; // grayscale, RGB, RGBA

/** @returns {Promise<{width: number, height: number, channels: number, depth: 8|16, data: Uint8Array|Uint16Array}>} */
export async function decodePng(buffer, name = "PNG") {
  const bytes = new Uint8Array(buffer);
  const view = new DataView(buffer);
  if (!SIGNATURE.every((b, i) => bytes[i] === b)) throw new Error(`${name}: not a PNG file`);

  let header = null;
  const idat = [];
  for (let pos = 8; pos < bytes.length; ) {
    const length = view.getUint32(pos);
    const type = String.fromCharCode(...bytes.subarray(pos + 4, pos + 8));
    const data = bytes.subarray(pos + 8, pos + 8 + length);
    if (type === "IHDR") {
      header = { width: view.getUint32(pos + 8), height: view.getUint32(pos + 12),
                 depth: data[8], colorType: data[9], interlace: data[12] };
    } else if (type === "IDAT") idat.push(data);
    else if (type === "IEND") break;
    pos += 12 + length;
  }
  if (!header || idat.length === 0) throw new Error(`${name}: missing IHDR or IDAT (truncated file?)`);
  const { width, height, depth, colorType, interlace } = header;
  const channels = CHANNELS[colorType];
  const supported = channels && interlace === 0 && (depth === 8 || (depth === 16 && colorType === 0));
  if (!supported) {
    throw new Error(`${name}: unsupported PNG (color type ${colorType}, depth ${depth}, interlace ${interlace}); ` +
                    "bundle layers are 8-bit grayscale/RGB/RGBA or 16-bit grayscale, non-interlaced");
  }

  const inflated = new Uint8Array(await new Response(
    new Blob(idat).stream().pipeThrough(new DecompressionStream("deflate"))).arrayBuffer());
  const bpp = channels * (depth / 8);
  const stride = width * bpp;
  if (inflated.length !== height * (stride + 1)) {
    throw new Error(`${name}: expected ${height * (stride + 1)} bytes of scanlines, got ${inflated.length}`);
  }

  const out = new Uint8Array(height * stride);
  for (let y = 0; y < height; y++) {
    const filter = inflated[y * (stride + 1)];
    const src = inflated.subarray(y * (stride + 1) + 1, (y + 1) * (stride + 1));
    const row = out.subarray(y * stride, (y + 1) * stride);
    const up = y > 0 ? out.subarray((y - 1) * stride, y * stride) : new Uint8Array(stride);
    for (let i = 0; i < stride; i++) {
      const a = i >= bpp ? row[i - bpp] : 0;
      const b = up[i];
      const c = i >= bpp ? up[i - bpp] : 0;
      switch (filter) {
        case 0: row[i] = src[i]; break;
        case 1: row[i] = src[i] + a; break;
        case 2: row[i] = src[i] + b; break;
        case 3: row[i] = src[i] + ((a + b) >> 1); break;
        case 4: {
          const p = a + b - c, pa = Math.abs(p - a), pb = Math.abs(p - b), pc = Math.abs(p - c);
          row[i] = src[i] + (pa <= pb && pa <= pc ? a : pb <= pc ? b : c);
          break;
        }
        default: throw new Error(`${name}: invalid filter type ${filter} on row ${y}`);
      }
    }
  }
  if (depth === 8) return { width, height, channels, depth, data: out };
  const wide = new Uint16Array(width * height * channels);
  for (let i = 0; i < wide.length; i++) wide[i] = (out[2 * i] << 8) | out[2 * i + 1];
  return { width, height, channels, depth, data: wide };
}
