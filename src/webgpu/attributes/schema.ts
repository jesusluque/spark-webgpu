// Generic per-Gaussian attributes: a schema of named, typed vectors, stored
// per attribute on the CPU and packed into the one-buffer pool that
// slang/core/attrib.slang decodes at run time (layout documented there).
//
// AttribPool holds only plain data, so it survives postMessage from the
// loader worker; AttribPool.from() restores the methods on the other side.

export type AttribFormat =
  | "f32"
  | "f16"
  | "unorm8"
  | "snorm8"
  | "u8"
  | "u16"
  | "u32";

/** How LOD merges children into a parent (rust/spark-lib LOD builders). */
export type LodMerge =
  | "weightedMean"
  | "normalizeMean"
  | "max"
  | "mode"
  | "first";

export interface AttributeSpec {
  name: string;
  format: AttribFormat;
  /** Components per splat, 1 or more. */
  components: number;
  lodMerge?: LodMerge;
  /** Copied by generate into the draw pool, readable in the draw shaders. */
  toDraw?: boolean;
  /** A 3-vector that turns with its object (normals), rotated by generate. */
  direction?: boolean;
}

/** Format codes, as ATTRIB_* in slang/core/attrib.slang. */
export const ATTRIB_FORMATS: Record<AttribFormat, number> = {
  f32: 0,
  f16: 1,
  unorm8: 2,
  snorm8: 3,
  u8: 4,
  u16: 5,
  u32: 6,
};
export const ATTRIB_NONE = 0xffffffff;
export const ATTRIB_DIRECTION = 1;

export function formatBytes(format: AttribFormat): number {
  return format === "f32" || format === "u32"
    ? 4
    : format === "f16" || format === "u16"
      ? 2
      : 1;
}

export function isIntegerFormat(format: AttribFormat) {
  return format === "u8" || format === "u16" || format === "u32";
}

export function attribWords(
  spec: Pick<AttributeSpec, "format" | "components">,
) {
  return Math.ceil((spec.components * formatBytes(spec.format)) / 4);
}

export function defaultLodMerge(spec: AttributeSpec): LodMerge {
  if (spec.direction) return "normalizeMean";
  return isIntegerFormat(spec.format) ? "mode" : "weightedMean";
}

/** One component to its stored bits (not yet shifted into its word). */
export function encodeComponent(format: AttribFormat, v: number): number {
  switch (format) {
    case "f32":
      F32[0] = v;
      return U32[0];
    case "f16":
      return toHalf(v);
    case "unorm8":
      return Math.round(Math.min(Math.max(v, 0), 1) * 255);
    case "snorm8":
      return Math.round(Math.min(Math.max(v, -1), 1) * 127) & 0xff;
    case "u8":
      return Math.min(Math.max(Math.round(v), 0), 255);
    case "u16":
      return Math.min(Math.max(Math.round(v), 0), 65535);
    case "u32":
      return Math.max(Math.round(v), 0) >>> 0;
  }
}

/** Stored bits to the value readAttrib returns. */
export function decodeComponent(format: AttribFormat, bits: number): number {
  switch (format) {
    case "f32":
      U32[0] = bits;
      return F32[0];
    case "f16":
      return fromHalf(bits);
    case "unorm8":
      return bits / 255;
    case "snorm8":
      return Math.max(((bits << 24) >> 24) / 127, -1);
    default:
      return bits >>> 0;
  }
}

const F32 = new Float32Array(1);
const U32 = new Uint32Array(F32.buffer);

// Half floats without src/utils.ts, which would pull three.js into the
// loader worker. Float16Array where the browser has it.
const F16Array = (globalThis as { Float16Array?: Float32ArrayConstructor })
  .Float16Array;
const F16 = F16Array ? new F16Array(1) : null;
const U16 = F16 ? new Uint16Array(F16.buffer) : null;

export function toHalf(v: number): number {
  if (F16 && U16) {
    F16[0] = v;
    return U16[0];
  }
  return toHalfJs(v);
}

/** toHalf without Float16Array: round to nearest even. */
export function toHalfJs(v: number): number {
  F32[0] = v;
  const bits = U32[0];
  const sign = (bits >>> 16) & 0x8000;
  const exp = ((bits >>> 23) & 0xff) - 112;
  const mant = bits & 0x7fffff;
  if (exp >= 31) {
    // Overflow to infinity; NaN stays NaN.
    const nan = ((bits >>> 23) & 0xff) === 0xff && mant !== 0;
    return sign | 0x7c00 | (nan ? 0x200 : 0);
  }
  if (exp <= 0) {
    if (exp < -10) return sign;
    const m = (mant | 0x800000) >>> (1 - exp);
    return sign | ((m + 0xfff + ((m >>> 13) & 1)) >>> 13);
  }
  // Round to nearest even; a carry into the exponent is still correct.
  return (
    (sign | ((exp << 10) + ((mant + 0xfff + ((mant >>> 13) & 1)) >>> 13))) &
    0xffff
  );
}

export function fromHalf(h: number): number {
  if (F16 && U16) {
    U16[0] = h;
    return F16[0];
  }
  const sign = h & 0x8000 ? -1 : 1;
  const exp = (h >>> 10) & 0x1f;
  const mant = h & 0x3ff;
  if (exp === 0) return sign * mant * 2 ** -24;
  if (exp === 31) return mant ? Number.NaN : sign * Number.POSITIVE_INFINITY;
  return sign * (1 + mant / 1024) * 2 ** (exp - 15);
}

/** Where each attribute sits in a pool built from `specs`. */
export interface PoolLayout {
  specs: AttributeSpec[];
  offsets: number[];
  strideWords: number;
  headerWords: number;
}

export function poolLayout(specs: AttributeSpec[]): PoolLayout {
  const offsets: number[] = [];
  let stride = 0;
  for (const s of specs) {
    offsets.push(stride);
    stride += attribWords(s);
  }
  // Four words of sizes, then a uint4 descriptor per attribute.
  const headerWords = 4 + 4 * specs.length;
  return { specs, offsets, strideWords: stride, headerWords };
}

/** A pool's header words: sizes, then one descriptor per attribute. */
export function poolHeader(layout: PoolLayout, count: number): Uint32Array {
  const h = new Uint32Array(layout.headerWords);
  h.set([layout.strideWords, layout.specs.length, layout.headerWords, count]);
  layout.specs.forEach((s, k) => {
    h.set(
      [
        layout.offsets[k],
        ATTRIB_FORMATS[s.format],
        s.components,
        s.direction ? ATTRIB_DIRECTION : 0,
      ],
      4 + 4 * k,
    );
  });
  return h;
}

export function poolWords(layout: PoolLayout, count: number) {
  return layout.headerWords + Math.max(count, 1) * layout.strideWords;
}

/** An attribute's packed words, `attribWords(spec)` per splat. */
export interface AttributeColumn {
  spec: AttributeSpec;
  words: Uint32Array;
}

function inferFormat(data: ArrayLike<number>): AttribFormat {
  if (data instanceof Uint8Array) return "u8";
  if (data instanceof Uint16Array) return "u16";
  if (data instanceof Uint32Array) return "u32";
  if (data instanceof Int8Array) return "snorm8";
  return "f32";
}

/** Packs `count * components` values into a column of `spec`'s format. */
export function packColumn(
  spec: AttributeSpec,
  data: ArrayLike<number>,
  count: number,
): Uint32Array {
  const n = spec.components;
  if (data.length < count * n) {
    throw new Error(
      `attribute '${spec.name}': ${data.length} values for ${count} x ${n}`,
    );
  }
  const per = attribWords(spec);
  const bytes = formatBytes(spec.format);
  const words = new Uint32Array(count * per);
  // 32-bit formats: a typed-array copy (u32 only from integers, exact).
  if (
    spec.format === "f32" ||
    (spec.format === "u32" && data instanceof Uint32Array)
  ) {
    const values = ArrayBuffer.isView(data)
      ? (data as Float64Array).subarray(0, count * n)
      : Array.prototype.slice.call(data, 0, count * n);
    (spec.format === "f32" ? new Float32Array(words.buffer) : words).set(
      values,
    );
    return words;
  }
  for (let i = 0; i < count; i++) {
    for (let c = 0; c < n; c++) {
      const bits = encodeComponent(spec.format, data[i * n + c]);
      const byte = c * bytes;
      words[i * per + (byte >> 2)] |= bits << ((byte & 3) * 8);
    }
  }
  return words;
}

/** Raw bits of component `c` of splat `i` in a column. */
export function columnBits(col: AttributeColumn, i: number, c: number) {
  const bytes = formatBytes(col.spec.format);
  const byte = c * bytes;
  const w = col.words[i * attribWords(col.spec) + (byte >> 2)];
  if (bytes === 4) return w >>> 0;
  return (w >>> ((byte & 3) * 8)) & ((1 << (8 * bytes)) - 1);
}

/**
 * The extra attributes of a set of splats: columns by name, packed for the
 * GPU on demand. Index i is splat i of the PackedSplats/ExtSplats/source.
 */
export class AttribPool {
  columns: AttributeColumn[] = [];
  /** Bumped on every change, so GPU copies know to re-upload. */
  version = 0;

  constructor(public count: number) {}

  /** Restores the methods of a pool that went through postMessage. */
  static from(
    data: AttribPool | { count: number; columns: AttributeColumn[] },
  ) {
    if (data instanceof AttribPool) return data;
    const pool = new AttribPool(data.count);
    pool.columns = data.columns;
    return pool;
  }

  get schema(): AttributeSpec[] {
    return this.columns.map((c) => c.spec);
  }

  id(name: string): number {
    return this.columns.findIndex((c) => c.spec.name === name);
  }

  column(name: string): AttributeColumn | undefined {
    return this.columns[this.id(name)];
  }

  /**
   * Adds or replaces attribute `name` from `count * components` values:
   * floats for f32/f16, [0, 1] for unorm8, [-1, 1] for snorm8, integers for
   * the u formats. The format defaults from the array type.
   */
  setAttribute(
    name: string,
    data: ArrayLike<number>,
    format: AttribFormat = inferFormat(data),
    components = 1,
    options: Partial<
      Omit<AttributeSpec, "name" | "format" | "components">
    > = {},
  ): AttributeSpec {
    const spec: AttributeSpec = { name, format, components, ...options };
    spec.lodMerge ??= defaultLodMerge(spec);
    this.setColumn({ spec, words: packColumn(spec, data, this.count) });
    return spec;
  }

  /** Adds or replaces an already packed column. */
  setColumn(col: AttributeColumn) {
    if (col.words.length < this.count * attribWords(col.spec)) {
      throw new Error(`attribute '${col.spec.name}': column too short`);
    }
    const k = this.id(col.spec.name);
    if (k >= 0) this.columns[k] = col;
    else this.columns.push(col);
    this.version += 1;
  }

  removeAttribute(name: string) {
    const k = this.id(name);
    if (k >= 0) {
      this.columns.splice(k, 1);
      this.version += 1;
    }
  }

  /** The decoded components of splat `index`. */
  getAttribute(name: string, index: number): number[] {
    const col = this.column(name);
    if (!col) throw new Error(`no attribute '${name}'`);
    return Array.from({ length: col.spec.components }, (_, c) =>
      decodeComponent(col.spec.format, columnBits(col, index, c)),
    );
  }

  /** The interleaved pool of the columns `filter` keeps, with its header. */
  pack(filter: (spec: AttributeSpec) => boolean = () => true): {
    layout: PoolLayout;
    words: Uint32Array;
  } {
    const cols = this.columns.filter((c) => filter(c.spec));
    const layout = poolLayout(cols.map((c) => c.spec));
    const words = new Uint32Array(poolWords(layout, this.count));
    words.set(poolHeader(layout, this.count));
    const stride = layout.strideWords;
    cols.forEach((col, k) => {
      const per = attribWords(col.spec);
      let at = layout.headerWords + layout.offsets[k];
      for (let i = 0; i < this.count; i++, at += stride) {
        words.set(col.words.subarray(i * per, (i + 1) * per), at);
      }
    });
    return { layout, words };
  }
}
