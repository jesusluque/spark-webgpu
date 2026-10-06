// WgpuSplatPager's pools read by generate: paged splats come out as the same
// splats uploaded as an ordinary source, for packed and ext (pagedExtSplats)
// pages, with the SH pool's stride sized for the SH degrees that arrived.

import { describe, expect, it } from "vitest";
import type { PagedSplats } from "../../src/PagedSplats";
import { encodeExtRgb, encodeExtSplat, setPackedSplat } from "../../src/utils";
import { GpuSplatSource } from "../../src/webgpu/GpuSplatSource";
import { KernelRegistry } from "../../src/webgpu/KernelRegistry";
import { WgpuSplatPager } from "../../src/webgpu/WgpuSplatPager";
import { specsFromRadMeta } from "../../src/webgpu/attributes/PagedAttribPool";
import {
  AttribPool,
  columnBits,
  decodeComponent,
  poolLayout,
} from "../../src/webgpu/attributes/schema";
import generate from "../../src/webgpu/generated/kernels/generate";
import { UniformWriter } from "../../src/webgpu/uniforms";
import { device, readBack, storage } from "./device";

const PAGE = 65536;
const SRC_EXT = 1;
const OUT_EXT = 2;
const USE_LOD = 4;
const COUNT = 300;

function rng(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 2 ** 32;
  };
}

type Upload = (
  page: number,
  packed: Uint32Array,
  sh: Uint32Array[],
  ext?: Uint32Array,
) => void;

const encoding = {
  rgbMin: 0,
  rgbMax: 1,
  lnScaleMin: -12,
  lnScaleMax: 9,
  sh1Max: 1,
  sh2Max: 1,
  sh3Max: 1,
  lodOpacity: false,
};

function pagedSplats(numSh: number) {
  return {
    numSh,
    maxSh: 3,
    splatEncoding: { ...encoding },
  } as unknown as PagedSplats;
}

describe.skipIf(!device)("WgpuSplatPager", () => {
  const d = device as GPUDevice;
  const registry = new KernelRegistry(d);

  // generate's ext output for `source` at `indices`, with SH up to `numSh`.
  async function run(
    source: GpuSplatSource,
    indices: Uint32Array,
    numSh: number,
  ) {
    const out = storage(indices.length * 32);
    const params = UniformWriter.for(generate).setAll({
      numSplats: indices.length,
      flags: USE_LOD | OUT_EXT | (source.format === "ext" ? SRC_EXT : 0),
      numSh,
      srcCount: source.count,
      shStride: source.shStride,
      rotate: [0, 0, 0, 1],
      translateScale: [0, 0, 0, 1],
      recolor: [1, 1, 1, 1],
      encoding: source.encoding,
      shMax: source.shMax,
      viewObject: [0.3, -2, 5, 0],
      viewDir: [0, 0, -1, 0],
    });
    registry.get(generate, "generate").run({
      grid: [indices.length],
      buffers: {
        src: source.src,
        sh: source.sh ?? storage(16),
        lodIndices: storage(indices),
        outSplats: out,
        sortMetric: storage(indices.length * 4),
      },
      uniforms: params.data,
    });
    return new Uint32Array(await readBack(out));
  }

  const range = (base: number) =>
    Uint32Array.from({ length: COUNT }, (_, i) => base + i);

  function packedPage(seed: number) {
    const r = rng(seed);
    const packed = new Uint32Array(COUNT * 4);
    for (let i = 0; i < COUNT; i++) {
      setPackedSplat(
        packed,
        i,
        r(),
        r(),
        r() - 2,
        0.02,
        0.02,
        0.02,
        0,
        0,
        0,
        1,
        0.8,
        r(),
        r(),
        r(),
      );
    }
    // Packed SH: sh1 2 words, sh2 and sh3 4 words, any bytes are valid.
    const words = (n: number) =>
      Uint32Array.from({ length: COUNT * n }, () => (r() * 2 ** 32) >>> 0);
    return { packed, sh: [words(2), words(4), words(4)] };
  }

  it("sizes the packed SH pool for the degrees uploaded, widening it", async () => {
    const pager = new WgpuSplatPager(d, {
      maxSplats: 2 * PAGE,
      onUpdate: () => {},
    });
    const upload = (pager as unknown as { uploadPage: Upload }).uploadPage.bind(
      pager,
    );
    const p0 = packedPage(1);
    const p1 = packedPage(2);

    // SH1 only: one uint4 a splat, a third of the SH3 pool.
    upload(0, p0.packed, p0.sh.slice(0, 1));
    expect(pager.pools.sh.buffer?.size).toBe(2 * PAGE * 16);
    const sh1 = pager.source(pagedSplats(1));
    expect(sh1.shStride).toBe(1);
    expect(sh1.numSh).toBe(1);
    const reference0 = GpuSplatSource.fromPacked(d, p0.packed, COUNT, {
      sh1: p0.sh[0],
    });
    const expected0 = await run(reference0, range(0), 1);
    // The SH colour shows in the output.
    expect(expected0).not.toEqual(await run(reference0, range(0), 0));
    expect(await run(sh1, range(0), 1)).toEqual(expected0);

    // An SH3 page widens the pool and keeps page 0.
    upload(1, p1.packed, p1.sh);
    expect(pager.pools.sh.buffer?.size).toBe(2 * PAGE * 48);
    const sh3 = pager.source(pagedSplats(3));
    expect(sh3.shStride).toBe(3);
    expect(await run(sh3, range(0), 1)).toEqual(expected0);
    const reference1 = GpuSplatSource.fromPacked(d, p1.packed, COUNT, {
      sh1: p1.sh[0],
      sh2: p1.sh[1],
      sh3: p1.sh[2],
    });
    expect(await run(sh3, range(PAGE), 3)).toEqual(
      await run(reference1, range(0), 3),
    );
    reference0.destroy();
    reference1.destroy();
    pager.dispose();
  });

  it("generates ext pages (pagedExtSplats) as the ext source would", async () => {
    const pager = new WgpuSplatPager(d, {
      extSplats: true,
      maxSplats: 2 * PAGE,
      onUpdate: () => {},
    });
    const r = rng(7);
    const a = new Uint32Array(COUNT * 4);
    const b = new Uint32Array(COUNT * 4);
    const sh = [0, 1].map(() => new Uint32Array(COUNT * 4));
    for (let i = 0; i < COUNT; i++) {
      encodeExtSplat(
        [a, b],
        i,
        100 + r(),
        r(),
        r() - 2,
        0.01,
        0.03,
        0.02,
        0.1,
        0.2,
        0.3,
        0.9,
        0.7,
        r(),
        r(),
        r(),
      );
      for (const s of sh) {
        for (let k = 0; k < 4; k++) {
          s[4 * i + k] = encodeExtRgb(r() - 0.5, r() - 0.5, r() - 0.5);
        }
      }
    }
    // SH2 on ext pages: two uint4s a splat (sh1, sh2).
    (pager as unknown as { uploadPage: Upload }).uploadPage(1, a, sh, b);
    const source = pager.source(pagedSplats(2));
    expect(source.format).toBe("ext");
    expect(source.shStride).toBe(2);
    expect(pager.pools.sh.buffer?.size).toBe(2 * PAGE * 32);
    const reference = GpuSplatSource.fromExt(d, a, b, COUNT, {
      sh1: sh[0],
      sh2: sh[1],
    });
    const got = await run(source, range(PAGE), 2);
    expect(got).toEqual(await run(reference, range(0), 2));
    // Centers far from the origin come through exactly.
    expect(new Float32Array(got.buffer, 0, 1)[0]).toBeGreaterThan(100);
    reference.destroy();
    pager.dispose();
  });

  it("pages a chunk's attributes into the schema of the .rad meta", async () => {
    const pager = new WgpuSplatPager(d, {
      maxSplats: 3 * PAGE,
      onUpdate: () => {},
    });
    const specs = specsFromRadMeta([
      { name: "label", format: "u8", components: 1, lodMerge: "mode" },
      {
        name: "normal",
        format: "snorm8",
        components: 3,
        lodMerge: "normalizeMean",
      },
      { name: "weight", format: "f16", components: 1 },
    ]);
    expect(specs[1].direction).toBe(true);
    pager.setAttribSchema(specs);
    const upload = (pager as unknown as { uploadPage: Upload }).uploadPage.bind(
      pager,
    );
    const { packed } = packedPage(3);
    // As a lone chunk decodes them: label as u32, no weight, and an extra
    // attribute the schema does not know.
    const chunk = new AttribPool(COUNT);
    chunk.setAttribute(
      "label",
      Array.from({ length: COUNT }, (_, i) => i % 200),
      "u32",
    );
    chunk.setAttribute(
      "normal",
      Array.from({ length: 3 * COUNT }, (_, k) => (k % 3 === 2 ? -1 : 0.5)),
      "f32",
      3,
    );
    chunk.setAttribute("other", new Float32Array(COUNT), "f32");
    // Through postMessage the pool arrives as plain data.
    upload(2, packed, [], undefined, JSON.parse(JSON.stringify(chunk)));

    const source = pager.source(pagedSplats(0));
    const pool = source.attribs as AttribPool;
    expect(pool.schema.map((s) => s.name)).toEqual([
      "label",
      "normal",
      "weight",
    ]);
    expect(pool.gpuBuffer).toBe(pager.attribs?.buffer);
    const words = new Uint32Array(await readBack(pool.gpuBuffer as GPUBuffer));
    const layout = poolLayout(specs);
    expect(Array.from(words.subarray(0, 4))).toEqual([
      layout.strideWords,
      3,
      layout.headerWords,
      3 * PAGE,
    ]);
    // Column k of splat `slot` read back from the pool.
    const value = (k: number, slot: number, c: number) => {
      const at =
        layout.headerWords + slot * layout.strideWords + layout.offsets[k];
      const col = { spec: specs[k], words: words.subarray(at, at + 4) };
      return decodeComponent(specs[k].format, columnBits(col, 0, c));
    };
    for (const i of [0, 7, COUNT - 1]) {
      expect(value(0, 2 * PAGE + i, 0)).toBe(i % 200);
      expect(value(1, 2 * PAGE + i, 0)).toBeCloseTo(0.5, 1);
      expect(value(1, 2 * PAGE + i, 2)).toBeCloseTo(-1, 2);
      expect(value(2, 2 * PAGE + i, 0)).toBe(0);
    }
    pager.dispose();
  });
});
