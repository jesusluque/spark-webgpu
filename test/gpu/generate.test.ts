// kernels/generate.slang against a CPU reference built from src/utils.ts
// decoders and three.js transforms.

import * as THREE from "three";
import { describe, expect, it } from "vitest";
import {
  decodeExtSplat,
  encodeExtSplat,
  setPackedSplat,
  unpackSplat,
} from "../../src/utils";
import { KernelRegistry } from "../../src/webgpu/KernelRegistry";
import { kernelsGenerate } from "../../src/webgpu/generated/constants";
import generate from "../../src/webgpu/generated/kernels/generate";
import { UniformWriter } from "../../src/webgpu/uniforms";
import { device, readBack, storage } from "./device";

const {
  GEN_SRC_EXT: SRC_EXT,
  GEN_OUT_EXT: OUT_EXT,
  GEN_USE_LOD: USE_LOD,
  GEN_SORT_RADIAL: SORT_RADIAL,
} = kernelsGenerate;

function rng(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 2 ** 32;
  };
}

const N = 2000;
const r = rng(7);
const splats = Array.from({ length: N }, () => {
  const q = new THREE.Quaternion(
    r() - 0.5,
    r() - 0.5,
    r() - 0.5,
    r() - 0.5,
  ).normalize();
  return {
    center: new THREE.Vector3(r() * 10 - 5, r() * 10 - 5, r() * 10 - 5),
    scales: new THREE.Vector3(...[0, 1, 2].map(() => Math.exp(r() * 4 - 4))),
    quat: q,
    rgba: [r(), r(), r(), 0.2 + 0.6 * r()],
  };
});

// Interleaved GPU layouts, and the CPU-side arrays to decode from.
const packed = new Uint32Array(N * 4);
const extA = new Uint32Array(N * 4);
const extB = new Uint32Array(N * 4);
splats.forEach((s, i) => {
  const args = [
    s.center.x,
    s.center.y,
    s.center.z,
    s.scales.x,
    s.scales.y,
    s.scales.z,
    s.quat.x,
    s.quat.y,
    s.quat.z,
    s.quat.w,
    s.rgba[3],
    s.rgba[0],
    s.rgba[1],
    s.rgba[2],
  ] as const;
  setPackedSplat(packed, i, ...args);
  encodeExtSplat([extA, extB], i, ...args);
});
const extInterleaved = new Uint32Array(N * 8);
for (let i = 0; i < N; i++) {
  extInterleaved.set(extA.subarray(4 * i, 4 * i + 4), 8 * i);
  extInterleaved.set(extB.subarray(4 * i, 4 * i + 4), 8 * i + 4);
}

const rotate = new THREE.Quaternion().setFromEuler(
  new THREE.Euler(0.3, -0.7, 1.1),
);
const translate = new THREE.Vector3(1.5, -2, 3);
const scale = 1.7;
const recolor = [0.9, 0.8, 1.0, 0.75];
const viewCenter = new THREE.Vector3(0.5, 1, 20);

describe.skipIf(!device)("generate.slang", () => {
  const registry = new KernelRegistry(device as GPUDevice);

  async function run(
    flags: number,
    count: number,
    lod?: Uint32Array,
    origin = [0, 0, 0],
  ) {
    const outWords = flags & OUT_EXT ? 8 : 4;
    const out = storage(N * outWords * 4);
    const metric = storage(N * 4);
    const params = UniformWriter.for(generate).setAll({
      numSplats: count,
      outBase: 0,
      flags,
      numSh: 0,
      srcCount: N,
      rotate: [rotate.x, rotate.y, rotate.z, rotate.w],
      translateScale: [translate.x, translate.y, translate.z, scale],
      recolor,
      encoding: [0, 1, -12, 9],
      viewCenter: [viewCenter.x, viewCenter.y, viewCenter.z, 0],
      viewDir: [0, 0, -1, 0],
      outOrigin: [...origin, 0],
    });
    registry.get(generate, "generate").run({
      grid: [count],
      buffers: {
        src: storage(flags & SRC_EXT ? extInterleaved : packed),
        sh: storage(16),
        lodIndices: storage(lod ?? new Uint32Array(4)),
        outSplats: out,
        sortMetric: metric,
      },
      uniforms: params.data,
    });
    return {
      out: new Uint32Array(await readBack(out)),
      metric: new Float32Array(await readBack(metric)),
    };
  }

  // The source splat after decoding, transformed on the CPU.
  function expected(i: number, srcExt: boolean) {
    const d = srcExt ? decodeExtSplat([extA, extB], i) : unpackSplat(packed, i);
    const center = d.center
      .clone()
      .multiplyScalar(scale)
      .applyQuaternion(rotate)
      .add(translate);
    return {
      center,
      scales: d.scales.clone().multiplyScalar(scale),
      quat: rotate.clone().multiply(d.quaternion),
      rgba: [
        d.color.r * recolor[0],
        d.color.g * recolor[1],
        d.color.b * recolor[2],
        d.opacity * recolor[3],
      ],
    };
  }

  it("transforms ext splats to ext output, with radial metric", async () => {
    const { out, metric } = await run(SRC_EXT | OUT_EXT | SORT_RADIAL, N);
    const a = new Uint32Array(N * 4);
    const b = new Uint32Array(N * 4);
    for (let i = 0; i < N; i++) {
      a.set(out.subarray(8 * i, 8 * i + 4), 4 * i);
      b.set(out.subarray(8 * i + 4, 8 * i + 8), 4 * i);
    }
    for (let i = 0; i < N; i++) {
      const e = expected(i, true);
      const g = decodeExtSplat([a, b], i);
      expect(g.center.distanceTo(e.center)).toBeLessThan(1e-3);
      expect(g.scales.x / e.scales.x).toBeCloseTo(1, 2);
      expect(Math.abs(g.quaternion.dot(e.quat))).toBeGreaterThan(0.9999);
      expect(g.opacity).toBeCloseTo(e.rgba[3], 2);
      expect(g.color.g).toBeCloseTo(e.rgba[1], 2);
      expect(metric[i]).toBeCloseTo(e.center.distanceTo(viewCenter), 3);
    }
  });

  it("writes packed output relative to the origin with alpha halved", async () => {
    const origin = [viewCenter.x, viewCenter.y, viewCenter.z];
    const { out } = await run(0, N, undefined, origin);
    for (let i = 0; i < N; i++) {
      const e = expected(i, false);
      const g = unpackSplat(out, i);
      const rel = e.center.clone().sub(viewCenter);
      expect(g.center.distanceTo(rel)).toBeLessThan(0.05);
      expect(g.opacity * 2).toBeCloseTo(e.rgba[3], 1);
    }
  });

  it("remaps through the LOD index list and writes the z metric", async () => {
    const lod = Uint32Array.from({ length: 100 }, (_, k) => N - 1 - 3 * k);
    const { out, metric } = await run(
      SRC_EXT | OUT_EXT | USE_LOD,
      lod.length,
      lod,
    );
    const a = new Uint32Array(lod.length * 4);
    const b = new Uint32Array(lod.length * 4);
    for (let i = 0; i < lod.length; i++) {
      a.set(out.subarray(8 * i, 8 * i + 4), 4 * i);
      b.set(out.subarray(8 * i + 4, 8 * i + 8), 4 * i);
    }
    lod.forEach((src, i) => {
      const e = expected(src, true);
      expect(
        decodeExtSplat([a, b], i).center.distanceTo(e.center),
      ).toBeLessThan(1e-3);
      const z =
        e.center
          .clone()
          .sub(viewCenter)
          .dot(new THREE.Vector3(0, 0, -1)) + 100;
      expect(metric[i]).toBeCloseTo(z, 3);
    });
  });
});
