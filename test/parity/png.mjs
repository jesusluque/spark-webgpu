// Minimal PNG codec and image diff, on node:zlib only: screenshots are 8-bit
// RGB(A), non-interlaced, which is all this has to read.

import zlib from "node:zlib";

const SIGNATURE = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);

/** Decodes a PNG to { width, height, data: RGBA Uint8Array }. */
export function decodePng(buf) {
  if (!buf.subarray(0, 8).equals(SIGNATURE)) throw new Error("not a PNG");
  let pos = 8;
  let width = 0;
  let height = 0;
  let channels = 0;
  let palette = null;
  let trns = null;
  const idat = [];
  while (pos < buf.length) {
    const len = buf.readUInt32BE(pos);
    const type = buf.toString("latin1", pos + 4, pos + 8);
    const body = buf.subarray(pos + 8, pos + 8 + len);
    pos += 12 + len;
    if (type === "IHDR") {
      width = body.readUInt32BE(0);
      height = body.readUInt32BE(4);
      const depth = body[8];
      const color = body[9];
      if (depth !== 8 || body[12] !== 0) {
        throw new Error(
          `unsupported PNG: depth ${depth}, interlace ${body[12]}`,
        );
      }
      channels = { 0: 1, 2: 3, 3: 1, 4: 2, 6: 4 }[color];
      if (!channels) throw new Error(`unsupported PNG colour type ${color}`);
      if (color === 3) palette = [];
      if (color === 0 || color === 4) palette = "gray";
    } else if (type === "PLTE") {
      palette = body;
    } else if (type === "tRNS") {
      trns = body;
    } else if (type === "IDAT") {
      idat.push(body);
    } else if (type === "IEND") {
      break;
    }
  }
  const raw = zlib.inflateSync(Buffer.concat(idat));
  const stride = width * channels;
  const px = Buffer.alloc(stride * height);
  for (let y = 0; y < height; y++) {
    const filter = raw[y * (stride + 1)];
    const src = y * (stride + 1) + 1;
    const dst = y * stride;
    for (let x = 0; x < stride; x++) {
      const a = x >= channels ? px[dst + x - channels] : 0;
      const b = y > 0 ? px[dst - stride + x] : 0;
      const c = x >= channels && y > 0 ? px[dst - stride + x - channels] : 0;
      let v = raw[src + x];
      if (filter === 1) v += a;
      else if (filter === 2) v += b;
      else if (filter === 3) v += (a + b) >> 1;
      else if (filter === 4) {
        const p = a + b - c;
        const pa = Math.abs(p - a);
        const pb = Math.abs(p - b);
        const pc = Math.abs(p - c);
        v += pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
      }
      px[dst + x] = v;
    }
  }
  const data = new Uint8Array(width * height * 4);
  for (let i = 0; i < width * height; i++) {
    const s = i * channels;
    let r;
    let g;
    let b;
    let a = 255;
    if (channels >= 3) {
      r = px[s];
      g = px[s + 1];
      b = px[s + 2];
      if (channels === 4) a = px[s + 3];
    } else if (palette === "gray") {
      r = g = b = px[s];
      if (channels === 2) a = px[s + 1];
    } else {
      const k = px[s];
      r = palette[k * 3];
      g = palette[k * 3 + 1];
      b = palette[k * 3 + 2];
      if (trns && k < trns.length) a = trns[k];
    }
    data.set([r, g, b, a], i * 4);
  }
  return { width, height, data };
}

function chunk(type, body) {
  const head = Buffer.alloc(8);
  head.writeUInt32BE(body.length, 0);
  head.write(type, 4, "latin1");
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(zlib.crc32(Buffer.concat([head.subarray(4), body])) >>> 0);
  return Buffer.concat([head, body, crc]);
}

/** Encodes RGBA pixels as a PNG (filter "up", which keeps diffs small). */
export function encodePng({ width, height, data }) {
  const stride = width * 4;
  const raw = Buffer.alloc((stride + 1) * height);
  for (let y = 0; y < height; y++) {
    raw[y * (stride + 1)] = y > 0 ? 2 : 0;
    for (let x = 0; x < stride; x++) {
      const v = data[y * stride + x];
      raw[y * (stride + 1) + 1 + x] =
        y > 0 ? (v - data[(y - 1) * stride + x]) & 255 : v;
    }
  }
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(width, 0);
  ihdr.writeUInt32BE(height, 4);
  ihdr.set([8, 6, 0, 0, 0], 8);
  return Buffer.concat([
    SIGNATURE,
    chunk("IHDR", ihdr),
    chunk("IDAT", zlib.deflateSync(raw, { level: 6 })),
    chunk("IEND", Buffer.alloc(0)),
  ]);
}

/** Box-filters an image down by an integer factor (Safari's 2x screenshots). */
export function downscale(img, factor) {
  if (factor === 1) return img;
  const width = Math.floor(img.width / factor);
  const height = Math.floor(img.height / factor);
  const data = new Uint8Array(width * height * 4);
  const n = factor * factor;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      for (let c = 0; c < 4; c++) {
        let sum = 0;
        for (let dy = 0; dy < factor; dy++) {
          for (let dx = 0; dx < factor; dx++) {
            sum +=
              img.data[
                ((y * factor + dy) * img.width + x * factor + dx) * 4 + c
              ];
          }
        }
        data[(y * width + x) * 4 + c] = Math.round(sum / n);
      }
    }
  }
  return { width, height, data };
}

/**
 * Compares two RGBA images (alpha ignored, as the canvas is composited):
 * mean absolute difference per channel in /255 units, the % of pixels whose
 * largest channel difference is over `over`, the % of "lit" pixels in each
 * (some channel over 10, to catch a page that drew nothing), and a diff image
 * amplified 4x. A larger image an integer multiple of the other's size is
 * box-filtered down first.
 */
export function diffImages(imgA, imgB, { over = 16 } = {}) {
  let a = imgA;
  let b = imgB;
  if (a.width !== b.width || a.height !== b.height) {
    const big = a.width > b.width ? a : b;
    const small = big === a ? b : a;
    const f = big.width / small.width;
    if (!Number.isInteger(f) || big.height / small.height !== f) {
      throw new Error(
        `image sizes differ: ${a.width}x${a.height} vs ${b.width}x${b.height}`,
      );
    }
    if (big === a) a = downscale(a, f);
    else b = downscale(b, f);
  }
  const n = a.width * a.height;
  const out = new Uint8Array(n * 4);
  let sum = 0;
  let overCount = 0;
  let litA = 0;
  let litB = 0;
  for (let i = 0; i < n; i++) {
    let max = 0;
    let maxA = 0;
    let maxB = 0;
    for (let c = 0; c < 3; c++) {
      const va = a.data[i * 4 + c];
      const vb = b.data[i * 4 + c];
      const d = Math.abs(va - vb);
      sum += d;
      if (d > max) max = d;
      if (va > maxA) maxA = va;
      if (vb > maxB) maxB = vb;
      out[i * 4 + c] = Math.min(255, d * 4);
    }
    out[i * 4 + 3] = 255;
    if (max > over) overCount++;
    if (maxA > 10) litA++;
    if (maxB > 10) litB++;
  }
  return {
    mean: sum / (3 * n),
    pctOver: (overCount / n) * 100,
    litA: (litA / n) * 100,
    litB: (litB / n) * 100,
    width: a.width,
    height: a.height,
    a,
    b,
    diff: { width: a.width, height: a.height, data: out },
  };
}
