// The vocabulary of aopenfx's effect model (sdk/include/aofx/{Types,
// Descriptor}.h), in TypeScript. Plain data, the same meaning as the C++, so
// an effect ported from an aofx bundle reads line for line like the original.

/** Half-open, in pixels, y up: (0,0) is the bottom left, as in aofx and OpenFX. */
export interface Rect {
  readonly x1: number;
  readonly y1: number;
  readonly x2: number;
  readonly y2: number;
}

export const EMPTY_RECT: Rect = { x1: 0, y1: 0, x2: 0, y2: 0 };

/**
 * Larger than any format; a generator's region of definition, which the
 * graph clamps to what it is asked for. Not infinite: a width that overflows
 * the moment anything subtracts corners is worse than a big number.
 */
export const EVERYWHERE: Rect = {
  x1: -1_000_000,
  y1: -1_000_000,
  x2: 1_000_000,
  y2: 1_000_000,
};

export const rect = (x1: number, y1: number, x2: number, y2: number): Rect => ({
  x1,
  y1,
  x2,
  y2,
});
export const rectWidth = (r: Rect) => r.x2 - r.x1;
export const rectHeight = (r: Rect) => r.y2 - r.y1;
export const isEmptyRect = (r: Rect) => r.x2 <= r.x1 || r.y2 <= r.y1;

export function intersectRect(a: Rect, b: Rect): Rect {
  const r = {
    x1: Math.max(a.x1, b.x1),
    y1: Math.max(a.y1, b.y1),
    x2: Math.min(a.x2, b.x2),
    y2: Math.min(a.y2, b.y2),
  };
  return isEmptyRect(r) ? EMPTY_RECT : r;
}

/** The union of the non-empty rectangles: aofx's default region of definition. */
export function unionRects(rects: readonly Rect[]): Rect {
  let out: Rect | null = null;
  for (const r of rects) {
    if (isEmptyRect(r)) continue;
    out = out
      ? {
          x1: Math.min(out.x1, r.x1),
          y1: Math.min(out.y1, r.y1),
          x2: Math.max(out.x2, r.x2),
          y2: Math.max(out.y2, r.y2),
        }
      : r;
  }
  return out ?? EMPTY_RECT;
}

export function equalRects(a: Rect, b: Rect) {
  return a.x1 === b.x1 && a.y1 === b.y1 && a.x2 === b.x2 && a.y2 === b.y2;
}

/**
 * A picture in device memory: float32 RGBA, linear, premultiplied, y up,
 * `stride` in pixels. `rect` is where it sits in the image, and `width` and
 * `height` always match it: effects relate two buffers by their rectangles
 * rather than assuming they line up.
 */
export interface FxBuffer {
  readonly buffer: GPUBuffer;
  readonly width: number;
  readonly height: number;
  readonly stride: number;
  readonly rect: Rect;
}

export const PIXEL_BYTES = 16;

export function isValidBuffer(b: FxBuffer | null | undefined): b is FxBuffer {
  return !!b && b.width > 0 && b.height > 0;
}

/** The same buffer, placed elsewhere in the picture (aofx's `scratch.rect = ...`). */
export function placed(b: FxBuffer, at: Rect): FxBuffer {
  return { ...b, rect: at };
}

/**
 * Parameter values as an effect sees them: a number per component, a
 * boolean, or text. Animation is the caller's business; an effect sees
 * numbers. A parameter absent here falls back to the effect's own default,
 * which matters where the default depends on the picture (Transform's centre).
 */
export type ParamValue = number | boolean | readonly number[] | string;
export type ParamValues = Readonly<Record<string, ParamValue>>;

export function paramNumbers(v: ParamValue | undefined): readonly number[] {
  if (v === undefined || typeof v === "string") return [];
  if (typeof v === "number") return [v];
  if (typeof v === "boolean") return [v ? 1 : 0];
  return v;
}

/** aofx's `ParamValue::number(at, fallback)`. */
export function paramNumber(
  params: ParamValues,
  name: string,
  fallback = 0,
  component = 0,
): number {
  const n = paramNumbers(params[name]);
  return component < n.length ? n[component] : fallback;
}

export function hasParam(params: ParamValues, name: string) {
  return paramNumbers(params[name]).length > 0;
}

export type ParamType =
  | "double"
  | "integer"
  | "boolean"
  | "choice"
  | "string"
  | "curve"
  | "button"
  | "colour"
  | "group";

/** What the numbers mean, so a host can draw a handle the plugin never draws. */
export type ParamRole =
  | "none"
  | "position"
  | "angle"
  | "scale"
  | "itemCount"
  | "itemIndex";

export interface ChoiceOption {
  /** What a script records. Indices are permanent: append, never reorder. */
  readonly value: string;
  readonly label: string;
}

export interface ParamDesc {
  readonly name: string;
  readonly label: string;
  readonly hint: string;
  readonly type: ParamType;
  readonly dimension?: number;
  readonly defaults: readonly number[];
  readonly textDefault?: string;
  readonly choices?: readonly ChoiceOption[];
  readonly role?: ParamRole;
  /** Defaults are fractions of the project size (Transform's centre). */
  readonly defaultsNormalised?: boolean;
  /** Shown only while another parameter has this value. */
  readonly shownWhen?: { readonly param: string; readonly is: string };
  readonly displayMin?: readonly number[];
  readonly displayMax?: readonly number[];
  readonly hardMin?: readonly number[];
  readonly hardMax?: readonly number[];
}

export interface ClipDesc {
  readonly name: string;
  readonly label: string;
  readonly optional?: boolean;
  readonly isMask?: boolean;
  /** The input an identity hands back and channel switches restore from. */
  readonly passThrough?: boolean;
}

export interface PlaneDesc {
  readonly id: string;
  readonly label: string;
  readonly channels?: readonly string[];
}

export interface EffectDesc {
  readonly identifier: string;
  readonly label: string;
  readonly grouping: string;
  readonly description: string;
  readonly versionMajor?: number;
  readonly versionMinor?: number;
  readonly inputs: readonly ClipDesc[];
  readonly params: readonly ParamDesc[];
  readonly outputs: readonly PlaneDesc[];
}

/** A Color output, which every picture-producing effect declares. */
export const COLOR_PLANE: PlaneDesc = {
  id: "Color",
  label: "Colour",
  channels: ["R", "G", "B", "A"],
};

/**
 * The parameter values an effect starts with, normalised defaults expanded
 * against the project size.
 */
export function defaultParams(
  desc: EffectDesc,
  projectWidth = 0,
  projectHeight = 0,
): Record<string, ParamValue> {
  const out: Record<string, ParamValue> = {};
  for (const p of desc.params) {
    if (p.type === "string") {
      out[p.name] = p.textDefault ?? "";
      continue;
    }
    let d = [...p.defaults];
    if (p.defaultsNormalised) {
      d = d.map((v, i) => v * (i === 0 ? projectWidth : projectHeight));
    }
    out[p.name] = p.type === "boolean" ? d[0] !== 0 : d.length === 1 ? d[0] : d;
  }
  return out;
}

/** The pass-through clip: the one marked so, else the first compulsory one. */
export function passThroughClip(desc: EffectDesc): ClipDesc | undefined {
  return (
    desc.inputs.find((c) => c.passThrough) ??
    desc.inputs.find((c) => !c.optional) ??
    desc.inputs[0]
  );
}
