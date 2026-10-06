import { describe, expect, it } from "vitest";
import {
  AttribPool,
  attribGroupWords,
  fromHalf,
  toHalf,
  toHalfJs,
} from "../../src/webgpu/attributes/schema";

describe("attribute schema", () => {
  it("spans a comp4 group's words as the Slang side does", () => {
    const f32x7 = { format: "f32", components: 7 } as const;
    expect(attribGroupWords(f32x7, 0)).toEqual([0, 4]);
    expect(attribGroupWords(f32x7, 1)).toEqual([4, 3]);
    expect(attribGroupWords(f32x7, 2)).toEqual([8, 0]);
    expect(attribGroupWords({ format: "f16", components: 8 }, 1)).toEqual([
      2, 2,
    ]);
    expect(attribGroupWords({ format: "u16", components: 3 }, 0)).toEqual([
      0, 2,
    ]);
    expect(attribGroupWords({ format: "snorm8", components: 3 }, 0)).toEqual([
      0, 1,
    ]);
    expect(attribGroupWords({ format: "u8", components: 1 }, 1)).toEqual([
      1, 0,
    ]);
  });

  it("rounds to half floats like Float16Array", () => {
    const values = [
      0,
      -0,
      1,
      -1,
      0.1,
      1 / 3,
      65504,
      65519,
      65520,
      1e6,
      -1e6,
      6e-8,
      3e-8,
      2.9e-8,
      6.1e-5,
      1e-6,
      2049,
      2051,
      Number.POSITIVE_INFINITY,
      // Just past a tie, by less than float32 keeps: rounding through
      // float32 would make them ties and round them down.
      1 + 2 ** -11 + 2 ** -40,
      2 ** -25 + 2 ** -40,
      3 * 2 ** -25 + 2 ** -45,
      65519.99,
    ];
    for (let i = 0; i < 2000; i++)
      values.push((Math.random() - 0.5) * 2 ** ((i % 40) - 20));
    for (const v of values) expect(toHalfJs(v)).toBe(toHalf(v));
    expect(Number.isNaN(fromHalf(toHalfJs(Number.NaN)))).toBe(true);
  });

  it("packs columns and reads them back", () => {
    const pool = new AttribPool(3);
    pool.setAttribute("label", Uint8Array.of(1, 2, 250));
    pool.setAttribute("n", [1, 0, 0, 0, -1, 0, 0, 0, 0.5], "snorm8", 3, {
      direction: true,
    });
    pool.setAttribute(
      "f",
      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
      "f16",
      5,
    );
    expect(pool.schema.map((s) => [s.name, s.format, s.lodMerge])).toEqual([
      ["label", "u8", "mode"],
      ["n", "snorm8", "normalizeMean"],
      ["f", "f16", "weightedMean"],
    ]);
    expect(pool.getAttribute("label", 2)).toEqual([250]);
    expect(pool.getAttribute("n", 1)[1]).toBe(-1);
    expect(pool.getAttribute("n", 2)[2]).toBeCloseTo(0.5, 2);
    expect(pool.getAttribute("f", 2)).toEqual([11, 12, 13, 14, 15]);

    const { layout, words } = pool.pack((s) => s.name !== "n");
    expect(layout.strideWords).toBe(1 + 3);
    expect(Array.from(words.subarray(0, 4))).toEqual([4, 2, 12, 3]);
    // Survives structured cloning, as from the loader worker.
    const copy = AttribPool.from(structuredClone(pool));
    expect(copy.getAttribute("f", 1)).toEqual([6, 7, 8, 9, 10]);
  });
});

// The loader worker's form of the WASM decoders' attributes.
describe("AttribPool.from(AttribValues)", () => {
  it("packs each attribute, a normalizeMean 3-vector as a direction", () => {
    const pool = AttribPool.from({
      count: 2,
      specs: [
        {
          name: "normal",
          format: "f32",
          components: 3,
          lodMerge: "normalizeMean",
        },
        { name: "label", format: "u8", components: 1, lodMerge: "mode" },
      ],
      values: [new Float64Array([0, 0, 1, 1, 0, 0]), new Float64Array([3, 7])],
    });
    expect(
      pool.schema.map((s) => [
        s.name,
        s.format,
        s.lodMerge,
        !!s.direction,
        !!s.toDraw,
      ]),
    ).toEqual([
      ["normal", "f32", "normalizeMean", true, true],
      ["label", "u8", "mode", false, true],
    ]);
    expect(pool.getAttribute("normal", 1)).toEqual([1, 0, 0]);
    expect(pool.getAttribute("label", 1)).toEqual([7]);
  });

  it("drops an all-zero normal", () => {
    const pool = AttribPool.from({
      count: 2,
      specs: [
        {
          name: "normal",
          format: "f32",
          components: 3,
          lodMerge: "normalizeMean",
        },
      ],
      values: [new Float64Array(6)],
    });
    expect(pool.columns).toEqual([]);
  });
});
