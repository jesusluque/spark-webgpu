// The Slang splat codecs (slang/core) against Spark's CPU encoders in
// src/utils.ts, on a real GPU.

import { describe, expect, it } from "vitest";
import {
  decodeExtRgb,
  decodeExtSplat,
  encodeExtRgb,
  encodeExtSplat,
  encodeSh1Rgb,
  encodeSh2Rgb,
  encodeSh3Rgb,
  setPackedSplat,
  unpackSplat,
} from "../../src/utils";
import { KernelRegistry } from "../../src/webgpu/KernelRegistry";
import codec from "../../src/webgpu/generated/tests/codec";
import { countParams, device, readBack, storage } from "./device";

const N = 4096;

// Deterministic pseudo-random numbers.
function rng(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 2 ** 32;
  };
}

interface Splat {
  center: number[];
  scales: number[];
  quat: number[];
  rgba: number[];
}

function randomSplats(n: number): Splat[] {
  const r = rng(1234);
  const out: Splat[] = [];
  for (let i = 0; i < n; i++) {
    let q = [r() - 0.5, r() - 0.5, r() - 0.5, r() - 0.5];
    const len = Math.hypot(...q);
    q = q.map((v) => v / len);
    out.push({
      center: [r() * 20 - 10, r() * 20 - 10, r() * 20 - 10],
      scales: [0, 1, 2].map(() => Math.exp(r() * 8 - 6)),
      quat: q,
      rgba: [r(), r(), r(), r()],
    });
  }
  return out;
}

function cpuEncode(splats: Splat[]) {
  const packed = new Uint32Array(N * 4);
  const extA = new Uint32Array(N * 4);
  const extB = new Uint32Array(N * 4);
  splats.forEach((s, i) => {
    const args = [
      ...s.center,
      ...s.scales,
      ...s.quat,
      s.rgba[3],
      ...s.rgba.slice(0, 3),
    ] as const;
    // biome-ignore lint/suspicious/noExplicitAny: positional encoder args
    setPackedSplat(packed, i, ...(args as any));
    // biome-ignore lint/suspicious/noExplicitAny: positional encoder args
    encodeExtSplat([extA, extB], i, ...(args as any));
  });
  return { packed, extA, extB };
}

const quatClose = (a: number[], b: number[]) =>
  Math.abs(a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]);

describe.skipIf(!device)("Slang splat codecs on the GPU", () => {
  const registry = new KernelRegistry(device as GPUDevice);
  const splats = randomSplats(N);
  const cpu = cpuEncode(splats);

  async function run(entry: string, encodedInit?: Uint32Array) {
    const input = new Float32Array(N * 16);
    splats.forEach((s, i) => {
      input.set([...s.center, 0, ...s.scales, 0, ...s.quat, ...s.rgba], i * 16);
    });
    const encoded = storage(encodedInit ?? N * 48);
    const decoded = storage(N * 128);
    registry.get(codec, entry).run({
      grid: [N],
      buffers: { input: storage(input), encoded, decoded },
      uniforms: countParams(N),
    });
    return {
      encoded: new Uint32Array(await readBack(encoded)),
      decoded: new Float32Array(await readBack(decoded)),
    };
  }

  it("decodes CPU-encoded packed and ext splats like the CPU decoders", async () => {
    const init = new Uint32Array(N * 12);
    for (let i = 0; i < N; i++) {
      init.set(cpu.packed.subarray(i * 4, i * 4 + 4), i * 12);
      init.set(cpu.extA.subarray(i * 4, i * 4 + 4), i * 12 + 4);
      init.set(cpu.extB.subarray(i * 4, i * 4 + 4), i * 12 + 8);
    }
    const { decoded } = await run("codecDecode", init);
    let worst = 0;
    for (let i = 0; i < N; i++) {
      const p = unpackSplat(cpu.packed, i);
      const pRef = [
        ...p.center.toArray(),
        1,
        ...p.scales.toArray(),
        0,
        ...p.quaternion.toArray(),
        p.color.r,
        p.color.g,
        p.color.b,
        p.opacity,
      ];
      const e = decodeExtSplat([cpu.extA, cpu.extB], i);
      const eRef = [
        ...e.center.toArray(),
        1,
        ...e.scales.toArray(),
        0,
        ...e.quaternion.toArray(),
        e.color.r,
        e.color.g,
        e.color.b,
        e.opacity,
      ];
      const ref = [...pRef, ...eRef];
      for (let k = 0; k < 32; k++) {
        if (k % 16 === 7) continue; // scales.w padding
        const g = decoded[i * 32 + k];
        const err = Math.abs(g - ref[k]) / Math.max(1, Math.abs(ref[k]));
        worst = Math.max(worst, err);
      }
    }
    expect(worst).toBeLessThan(1e-4);
  });

  it("encodes like the CPU and round-trips within quantization", async () => {
    const { encoded, decoded } = await run("codecRoundTrip");
    let sameWords = 0;
    let worstPackedQuat = 1;
    let worstExtQuat = 1;
    for (let i = 0; i < N; i++) {
      for (let k = 0; k < 4; k++) {
        if (encoded[i * 12 + k] === cpu.packed[i * 4 + k]) sameWords++;
        if (encoded[i * 12 + 4 + k] === cpu.extA[i * 4 + k]) sameWords++;
        if (encoded[i * 12 + 8 + k] === cpu.extB[i * 4 + k]) sameWords++;
      }
      const s = splats[i];
      const d = decoded.subarray(i * 32, i * 32 + 32);
      for (const [base, scaleTol] of [
        [0, 0.05],
        [16, 2e-3],
      ] as const) {
        for (let k = 0; k < 3; k++) {
          expect(Math.abs(d[base + k] - s.center[k])).toBeLessThan(0.01);
          expect(Math.abs(d[base + 4 + k] / s.scales[k] - 1)).toBeLessThan(
            scaleTol,
          );
        }
        expect(Math.abs(d[base + 15] - s.rgba[3])).toBeLessThan(
          base ? 1e-3 : 0.003,
        );
      }
      worstPackedQuat = Math.min(
        worstPackedQuat,
        quatClose([...d.subarray(8, 12)], s.quat),
      );
      worstExtQuat = Math.min(
        worstExtQuat,
        quatClose([...d.subarray(24, 28)], s.quat),
      );
    }
    expect(sameWords / (N * 12)).toBeGreaterThan(0.99);
    expect(worstPackedQuat).toBeGreaterThan(0.99);
    expect(worstExtQuat).toBeGreaterThan(0.9999);
  });

  it("evaluates packed and ext SH like a CPU reference", async () => {
    const r = rng(99);
    const M = 1024;
    const dirs = new Float32Array(M * 8);
    const words = new Uint32Array(M * 28);
    const ref: number[][] = [];
    const sh1 = new Uint32Array(2);
    const sh2 = new Uint32Array(4);
    const sh3 = new Uint32Array(4);
    for (let i = 0; i < M; i++) {
      let d = [r() - 0.5, r() - 0.5, r() - 0.5];
      const len = Math.hypot(...d);
      d = d.map((v) => v / len);
      const numSh = 1 + (i % 3);
      dirs.set([...d, numSh, 1, 1, 1, 0], i * 8);
      const c1 = Float32Array.from({ length: 9 }, () => r() * 2 - 1);
      const c2 = Float32Array.from({ length: 15 }, () => r() * 2 - 1);
      const c3 = Float32Array.from({ length: 21 }, () => r() * 2 - 1);
      sh1.fill(0);
      sh3.fill(0);
      encodeSh1Rgb(sh1, 0, c1);
      encodeSh2Rgb(sh2, 0, c2);
      encodeSh3Rgb(sh3, 0, c3);
      const ext = (c: Float32Array, k: number) =>
        encodeExtRgb(c[3 * k], c[3 * k + 1], c[3 * k + 2]);
      words.set([...sh1, 0, 0, ...sh2, ...sh3], i * 28);
      words.set([ext(c1, 0), ext(c1, 1), ext(c1, 2), ext(c2, 0)], i * 28 + 12);
      words.set([ext(c2, 1), ext(c2, 2), ext(c2, 3), ext(c2, 4)], i * 28 + 16);
      words.set([ext(c3, 0), ext(c3, 1), ext(c3, 2), ext(c3, 3)], i * 28 + 20);
      words.set([ext(c3, 4), ext(c3, 5), ext(c3, 6), 0], i * 28 + 24);

      // Reference from the dequantized coefficients.
      const [x, y, z] = d;
      const basis = [
        [-0.4886025 * y, 0.4886025 * z, -0.4886025 * x],
        [
          1.0925484 * x * y,
          -1.0925484 * y * z,
          0.3153915 * (2 * z * z - x * x - y * y),
          -1.0925484 * x * z,
          0.5462742 * (x * x - y * y),
        ],
        [
          -0.5900436 * y * (3 * x * x - y * y),
          2.8906114 * x * y * z,
          -0.4570458 * y * (4 * z * z - x * x - y * y),
          0.3731763 * z * (2 * z * z - 3 * x * x - 3 * y * y),
          -0.4570458 * x * (4 * z * z - x * x - y * y),
          1.4453057 * z * (x * x - y * y),
          -0.5900436 * x * (x * x - 3 * y * y),
        ],
      ];
      const signed = (w: Uint32Array, o: number, width: number) => {
        const wi = o >> 5;
        const bo = o & 31;
        let v = w[wi] >>> bo;
        if (bo + width > 32) v = (v | (w[wi + 1] << (32 - bo))) >>> 0;
        return (v << (32 - width)) >> (32 - width);
      };
      const packedRgb = [0, 0, 0];
      const extRgb = [0, 0, 0];
      const degrees = [
        [sh1, 7, 63, c1],
        [sh2, 8, 127, c2],
        [sh3, 6, 31, c3],
      ] as const;
      for (let deg = 0; deg < numSh; deg++) {
        const [w, width, scale, c] = degrees[deg];
        basis[deg].forEach((b, k) => {
          const e = decodeExtRgb(ext(c, k));
          const ec = [e.r, e.g, e.b];
          for (let ch = 0; ch < 3; ch++) {
            packedRgb[ch] +=
              (signed(w, (3 * k + ch) * width, width) / scale) * b;
            extRgb[ch] += ec[ch] * b;
          }
        });
      }
      ref.push([...packedRgb, ...extRgb]);
    }
    const out = storage(M * 32);
    registry.get(codec, "shEvaluate").run({
      grid: [M],
      buffers: { shDirs: storage(dirs), shWords: storage(words), shRgb: out },
      uniforms: countParams(M),
    });
    const got = new Float32Array(await readBack(out));
    let worst = 0;
    for (let i = 0; i < M; i++) {
      for (let ch = 0; ch < 3; ch++) {
        worst = Math.max(worst, Math.abs(got[i * 8 + ch] - ref[i][ch]));
        worst = Math.max(worst, Math.abs(got[i * 8 + 4 + ch] - ref[i][3 + ch]));
      }
    }
    expect(worst).toBeLessThan(1e-4);
  });
});
