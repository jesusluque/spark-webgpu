import { describe, expect, it } from "vitest";
import { PlyAttributeReader } from "../../src/webgpu/attributes/plyAttributes";
import {
  AttribPool,
  fromHalf,
  toHalf,
  toHalfJs,
} from "../../src/webgpu/attributes/schema";

describe("attribute schema", () => {
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

// A binary PLY with the standard 3DGS properties plus extras.
function makePly(n: number, zeroNormals: boolean) {
  const props: [string, string][] = [
    ["float", "x"],
    ["float", "y"],
    ["float", "z"],
    ["float", "nx"],
    ["float", "ny"],
    ["float", "nz"],
    ["float", "f_dc_0"],
    ["float", "f_dc_1"],
    ["float", "f_dc_2"],
    ["float", "opacity"],
    ["float", "scale_0"],
    ["float", "scale_1"],
    ["float", "scale_2"],
    ["float", "rot_0"],
    ["float", "rot_1"],
    ["float", "rot_2"],
    ["float", "rot_3"],
    ["uchar", "label"],
    ["float", "feat_0"],
    ["float", "feat_1"],
    ["float", "feat_2"],
    ["ushort", "instance"],
    ["int", "weird_0"],
  ];
  const header = `ply\nformat binary_little_endian 1.0\nelement vertex ${n}\n${props
    .map(([t, p]) => `property ${t} ${p}`)
    .join("\n")}\nend_header\n`;
  const size = props.reduce(
    (s, [t]) => s + (t === "uchar" ? 1 : t === "ushort" ? 2 : 4),
    0,
  );
  const head = new TextEncoder().encode(header);
  const bytes = new Uint8Array(head.length + n * size);
  bytes.set(head);
  const view = new DataView(bytes.buffer);
  for (let i = 0; i < n; i++) {
    let at = head.length + i * size;
    for (const [t, p] of props) {
      let v = i + 0.5;
      if (p === "nx") v = zeroNormals ? 0 : 1;
      if (p === "ny" || p === "nz") v = 0;
      if (p === "label") v = i % 7;
      if (p.startsWith("feat_")) v = i * 10 + Number(p.slice(5));
      if (p === "instance") v = 1000 + i;
      if (p === "weird_0") v = -i;
      if (t === "uchar") view.setUint8(at, v);
      else if (t === "ushort") view.setUint16(at, v, true);
      else if (t === "int") view.setInt32(at, v, true);
      else view.setFloat32(at, v, true);
      at += t === "uchar" ? 1 : t === "ushort" ? 2 : 4;
    }
  }
  return bytes;
}

describe("PLY attributes", () => {
  it("groups extra properties and reads them across chunk boundaries", () => {
    const n = 101;
    const bytes = makePly(n, false);
    const reader = new PlyAttributeReader();
    // Odd chunk sizes, splitting the header and records.
    for (let at = 0; at < bytes.length; at += 37) {
      reader.push(bytes.subarray(at, at + 37));
    }
    const pool = reader.finish() as AttribPool;
    expect(pool.count).toBe(n);
    expect(
      pool.schema.map((s) => [s.name, s.format, s.components, !!s.direction]),
    ).toEqual([
      ["normal", "f32", 3, true],
      ["label", "u8", 1, false],
      ["feat", "f32", 3, false],
      ["instance", "u16", 1, false],
      ["weird_0", "f32", 1, false],
    ]);
    for (const i of [0, 50, 100]) {
      expect(pool.getAttribute("normal", i)).toEqual([1, 0, 0]);
      expect(pool.getAttribute("label", i)).toEqual([i % 7]);
      expect(pool.getAttribute("feat", i)).toEqual([
        i * 10,
        i * 10 + 1,
        i * 10 + 2,
      ]);
      expect(pool.getAttribute("instance", i)).toEqual([1000 + i]);
      expect(pool.getAttribute("weird_0", i)).toEqual([i ? -i : 0]);
    }
  });

  it("drops all-zero normals and ignores other files", () => {
    const reader = new PlyAttributeReader();
    reader.push(makePly(10, true));
    expect(reader.finish()?.schema.map((s) => s.name)).not.toContain("normal");

    const other = new PlyAttributeReader();
    other.push(new TextEncoder().encode("NGSPLAT and more bytes"));
    expect(other.finish()).toBeNull();
  });
});
