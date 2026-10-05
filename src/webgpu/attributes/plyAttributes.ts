// The non-standard vertex properties of a binary PLY as per-Gaussian
// attributes. Runs beside the WASM decoder on the same byte stream (it never
// sees the splats, only the properties Spark ignores), so splat i of the
// decoded PackedSplats/ExtSplats is vertex i.
//
// Grouping: nx, ny, nz -> "normal" (a direction); name_0, name_1, ... of one
// type -> "name" with that many components; anything else alone. An
// all-zero normal is dropped: 3DGS trainers write nx = ny = nz = 0.

import {
  type AttribFormat,
  AttribPool,
  type AttributeSpec,
  defaultLodMerge,
} from "./schema";

// Properties the splat decoders read (rust/spark-lib/src/ply.rs).
const STANDARD = new Set([
  "x",
  "y",
  "z",
  "opacity",
  "red",
  "green",
  "blue",
  "alpha",
  "scale_0",
  "scale_1",
  "scale_2",
  "rot_0",
  "rot_1",
  "rot_2",
  "rot_3",
]);

const TYPES: Record<string, { bytes: number; format: AttribFormat }> = {
  char: { bytes: 1, format: "f32" },
  int8: { bytes: 1, format: "f32" },
  uchar: { bytes: 1, format: "u8" },
  uint8: { bytes: 1, format: "u8" },
  short: { bytes: 2, format: "f32" },
  int16: { bytes: 2, format: "f32" },
  ushort: { bytes: 2, format: "u16" },
  uint16: { bytes: 2, format: "u16" },
  int: { bytes: 4, format: "f32" },
  int32: { bytes: 4, format: "f32" },
  uint: { bytes: 4, format: "u32" },
  uint32: { bytes: 4, format: "u32" },
  float: { bytes: 4, format: "f32" },
  float32: { bytes: 4, format: "f32" },
  double: { bytes: 8, format: "f32" },
  float64: { bytes: 8, format: "f32" },
};

interface Property {
  name: string;
  type: string;
  offset: number;
}

interface Group {
  spec: AttributeSpec;
  props: Property[];
  values: Float64Array | Uint32Array;
}

function readValue(view: DataView, at: number, type: string): number {
  switch (type) {
    case "char":
    case "int8":
      return view.getInt8(at);
    case "uchar":
    case "uint8":
      return view.getUint8(at);
    case "short":
    case "int16":
      return view.getInt16(at, true);
    case "ushort":
    case "uint16":
      return view.getUint16(at, true);
    case "int":
    case "int32":
      return view.getInt32(at, true);
    case "uint":
    case "uint32":
      return view.getUint32(at, true);
    case "double":
    case "float64":
      return view.getFloat64(at, true);
    default:
      return view.getFloat32(at, true);
  }
}

/** Groups the extra properties of the vertex element into attributes. */
export function plyAttributeGroups(
  props: Property[],
): { spec: AttributeSpec; props: Property[] }[] {
  const extra = props.filter(
    (p) =>
      !STANDARD.has(p.name) &&
      !p.name.startsWith("f_dc_") &&
      !p.name.startsWith("f_rest_"),
  );
  const groups: { spec: AttributeSpec; props: Property[] }[] = [];
  const used = new Set<Property>();
  const byName = new Map(extra.map((p) => [p.name, p]));
  const normal = ["nx", "ny", "nz"].map((n) => byName.get(n));
  if (normal.every((p) => p && p.type === normal[0]?.type)) {
    const ps = normal as Property[];
    for (const p of ps) used.add(p);
    groups.push({
      spec: {
        name: "normal",
        format: TYPES[ps[0].type].format,
        components: 3,
        direction: true,
      },
      props: ps,
    });
  }
  for (const p of extra) {
    if (used.has(p)) continue;
    const m = /^(.*)_0$/.exec(p.name);
    const ps = [p];
    if (m) {
      for (let k = 1; ; k++) {
        const q = byName.get(`${m[1]}_${k}`);
        if (!q || used.has(q) || q.type !== p.type) break;
        ps.push(q);
      }
    }
    for (const q of ps) used.add(q);
    groups.push({
      spec: {
        name: m && ps.length > 1 ? m[1] : p.name,
        format: TYPES[p.type].format,
        components: ps.length,
      },
      props: ps,
    });
  }
  for (const g of groups) {
    g.spec.lodMerge = defaultLodMerge(g.spec);
    g.spec.toDraw = true;
  }
  return groups;
}

/**
 * Collects the extra attributes of a PLY from its bytes as they stream in;
 * `finish()` returns them, or null for a file that is not a plain binary
 * little-endian PLY or has none.
 */
export class PlyAttributeReader {
  private header: Uint8Array | null = new Uint8Array(0);
  private active = true;
  private leftover = new Uint8Array(0);
  private recordSize = 0;
  private count = 0;
  private next = 0;
  private groups: Group[] = [];
  private result: AttribPool | null | undefined;

  push = (chunk: Uint8Array) => {
    if (!this.active || chunk.length === 0) return;
    let data = chunk;
    if (this.header) {
      const joined = concat(this.header, chunk);
      if (joined.length >= 4 && !isPly(joined)) {
        this.active = false;
        return;
      }
      const end = indexOf(joined, END_HEADER);
      if (end < 0) {
        this.header = joined.length < 65536 ? joined : null;
        if (!this.header) this.active = false;
        return;
      }
      this.header = null;
      const text = new TextDecoder().decode(joined.subarray(0, end));
      if (!this.parseHeader(text)) {
        this.active = false;
        return;
      }
      data = joined.subarray(end + END_HEADER.length);
    }
    this.consume(data);
  };

  private parseHeader(text: string): boolean {
    const lines = text.split(/\r?\n/).map((l) => l.trim().split(/\s+/));
    if (
      !lines.some((l) => l[0] === "format" && l[1] === "binary_little_endian")
    )
      return false;
    let element: { name: string; count: number; size: number } | null = null;
    let fixed = true;
    let props: Property[] = [];
    let first = true;
    for (const l of lines) {
      if (l[0] === "element") {
        if (element?.name === "vertex") break;
        first = element === null;
        element = { name: l[1], count: Number(l[2]), size: 0 };
        props = [];
        fixed = true;
      } else if (l[0] === "property" && element) {
        if (l[1] === "list" || !TYPES[l[1]]) {
          fixed = false;
          continue;
        }
        props.push({ name: l[2], type: l[1], offset: element.size });
        element.size += TYPES[l[1]].bytes;
      }
    }
    // Vertices must come first: Supersplat's compressed PLY has its chunk
    // element before them and packs everything, nothing extra to read.
    if (element?.name !== "vertex" || !fixed || !first) return false;
    const groups = plyAttributeGroups(props);
    if (groups.length === 0) return false;
    this.count = element.count;
    this.recordSize = element.size;
    this.groups = groups.map((g) => ({
      ...g,
      values:
        g.spec.format === "f32"
          ? new Float64Array(this.count * g.spec.components)
          : new Uint32Array(this.count * g.spec.components),
    }));
    return true;
  }

  private consume(chunk: Uint8Array) {
    const bytes = this.leftover.length ? concat(this.leftover, chunk) : chunk;
    const size = this.recordSize;
    const records = Math.min(
      Math.floor(bytes.length / size),
      this.count - this.next,
    );
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    for (const g of this.groups) {
      const n = g.spec.components;
      for (let r = 0; r < records; r++) {
        const at = r * size;
        const i = (this.next + r) * n;
        for (let c = 0; c < n; c++) {
          const p = g.props[c];
          g.values[i + c] = readValue(view, at + p.offset, p.type);
        }
      }
    }
    this.next += records;
    this.leftover =
      this.next < this.count ? bytes.slice(records * size) : new Uint8Array(0);
  }

  finish(): AttribPool | null {
    if (this.result === undefined) this.result = this.build();
    return this.result;
  }

  private build(): AttribPool | null {
    if (!this.active || this.groups.length === 0 || this.next < this.count) {
      return null;
    }
    const pool = new AttribPool(this.count);
    for (const g of this.groups) {
      if (g.spec.direction && g.values.every((v) => v === 0)) continue;
      const { name, format, components, ...options } = g.spec;
      pool.setAttribute(name, g.values, format, components, options);
    }
    return pool.columns.length ? pool : null;
  }
}

const END_HEADER = new TextEncoder().encode("end_header\n");

function isPly(b: Uint8Array) {
  return b[0] === 0x70 && b[1] === 0x6c && b[2] === 0x79; // "ply"
}

function concat(a: Uint8Array, b: Uint8Array) {
  const out = new Uint8Array(a.length + b.length);
  out.set(a);
  out.set(b, a.length);
  return out;
}

function indexOf(hay: Uint8Array, needle: Uint8Array) {
  outer: for (let i = 0; i + needle.length <= hay.length; i++) {
    for (let k = 0; k < needle.length; k++) {
      if (hay[i + k] !== needle[k]) continue outer;
    }
    return i;
  }
  return -1;
}
