// aopenfx/examples/transform/Transform.cpp: translate, rotate, scale and skew
// about a centre, with net.sf.openfx.TransformPlugin's names, defaults, order
// of composition and nine filters. No motion blur, as in the C++.
//
// The region of definition moves: all four corners go through the matrix,
// never two, which is right only until somebody rotates. The region of
// interest is the output's corners carried back through the inverse, grown
// by the filter's reach.

import { fxTransform } from "../../generated/constants";
import transformModule from "../../generated/fx/transform";
import { UniformWriter } from "../../uniforms";
import { Effect, type RegionQuery, type RenderRequest } from "../Effect";
import {
  COLOR_PLANE,
  type EffectDesc,
  type ParamValues,
  type Rect,
  hasParam,
  isEmptyRect,
  isValidBuffer,
  paramNumber,
  unionRects,
} from "../types";

/** A 2x3 affine, row major: x' = a x + b y + c, y' = d x + e y + f. */
export interface Affine {
  a: number;
  b: number;
  c: number;
  d: number;
  e: number;
  f: number;
}

const IDENTITY: Affine = { a: 1, b: 0, c: 0, d: 0, e: 1, f: 0 };

export function multiplyAffine(l: Affine, r: Affine): Affine {
  return {
    a: l.a * r.a + l.b * r.d,
    b: l.a * r.b + l.b * r.e,
    c: l.a * r.c + l.b * r.f + l.c,
    d: l.d * r.a + l.e * r.d,
    e: l.d * r.b + l.e * r.e,
    f: l.d * r.c + l.e * r.f + l.f,
  };
}

/** The identity where there is no inverse: a scale of zero shows the input. */
export function invertAffine(m: Affine): Affine {
  const det = m.a * m.e - m.b * m.d;
  if (Math.abs(det) < 1e-12) return { ...IDENTITY };
  const inv = 1 / det;
  const a = m.e * inv;
  const b = -m.b * inv;
  const d = -m.d * inv;
  const e = m.a * inv;
  return { a, b, c: -(a * m.c + b * m.f), d, e, f: -(d * m.c + e * m.f) };
}

export const applyAffine = (m: Affine, x: number, y: number) =>
  [m.a * x + m.b * y + m.c, m.d * x + m.e * y + m.f] as const;

export interface TransformSettings {
  translateX: number;
  translateY: number;
  rotate: number;
  scaleX: number;
  scaleY: number;
  skewX: number;
  skewY: number;
  skewOrderYX: boolean;
  centerX: number;
  centerY: number;
  invert: boolean;
}

/** The plugin's order: centre to origin, scale, skew, rotate, back, translate. */
export function forwardOf(s: TransformSettings): Affine {
  const r = (s.rotate * Math.PI) / 180;
  const cos = Math.cos(r);
  const sin = Math.sin(r);
  const toOrigin = { ...IDENTITY, c: -s.centerX, f: -s.centerY };
  const scale = { ...IDENTITY, a: s.scaleX, e: s.scaleY };
  const skew = s.skewOrderYX
    ? { a: s.skewX * s.skewY + 1, b: s.skewX, c: 0, d: s.skewY, e: 1, f: 0 }
    : { a: 1, b: s.skewX, c: 0, d: s.skewY, e: s.skewX * s.skewY + 1, f: 0 };
  // The sign is the plugin's, established by measurement in the C++.
  const rotation = { a: cos, b: -sin, c: 0, d: sin, e: cos, f: 0 };
  const back = {
    ...IDENTITY,
    c: s.centerX + s.translateX,
    f: s.centerY + s.translateY,
  };
  return multiplyAffine(
    back,
    multiplyAffine(
      rotation,
      multiplyAffine(skew, multiplyAffine(scale, toOrigin)),
    ),
  );
}

export function matrixOf(s: TransformSettings): Affine {
  const m = forwardOf(s);
  return s.invert ? invertAffine(m) : m;
}

export const TRANSFORM_FILTERS = [
  "impulse",
  "box",
  "bilinear",
  "cubic",
  "keys",
  "simon",
  "rifman",
  "mitchell",
  "parzen",
] as const;
const FILTER_BOX = fxTransform.kBox;
const BOX_REACH = fxTransform.kBoxReach;

export function settingsFrom(params: ParamValues): TransformSettings {
  const n = (name: string, fallback: number, c = 0) =>
    paramNumber(params, name, fallback, c);
  const scaleX = n("scale", 1, 0);
  return {
    translateX: n("translate", 0, 0),
    translateY: n("translate", 0, 1),
    rotate: n("rotate", 0),
    scaleX,
    scaleY: n("uniform", 0) !== 0 ? scaleX : n("scale", 1, 1),
    skewX: n("skewX", 0),
    skewY: n("skewY", 0),
    skewOrderYX: n("skewOrder", 0) !== 0,
    centerX: n("center", 0, 0),
    centerY: n("center", 0, 1),
    invert: n("invert", 0) !== 0,
  };
}

/** Lengths into this render's pixels; ratios and angles mean the same at any size. */
function bringToScale(s: TransformSettings, scaleX: number, scaleY: number) {
  const sx = scaleX > 0 ? scaleX : 1;
  const sy = scaleY > 0 ? scaleY : 1;
  s.translateX *= sx;
  s.translateY *= sy;
  s.centerX *= sx;
  s.centerY *= sy;
}

function boundsOf(m: Affine, r: Rect) {
  const pts = [
    applyAffine(m, r.x1, r.y1),
    applyAffine(m, r.x2, r.y1),
    applyAffine(m, r.x1, r.y2),
    applyAffine(m, r.x2, r.y2),
  ];
  return {
    minX: Math.min(...pts.map((p) => p[0])),
    minY: Math.min(...pts.map((p) => p[1])),
    maxX: Math.max(...pts.map((p) => p[0])),
    maxY: Math.max(...pts.map((p) => p[1])),
  };
}

/** The settings a region question means, centred on the input's middle when unset. */
function scaledSettings(q: RegionQuery, input: Rect) {
  const s = settingsFrom(q.params);
  bringToScale(s, q.scaleX, q.scaleY);
  if (!hasParam(q.params, "center")) {
    s.centerX = (input.x1 + input.x2) * 0.5;
    s.centerY = (input.y1 + input.y2) * 0.5;
  }
  return s;
}

export class Transform extends Effect {
  protected describeEffect(): EffectDesc {
    const bool = (name: string, label: string, hint: string, on = 0) => ({
      name,
      label,
      hint,
      type: "boolean" as const,
      defaults: [on],
    });
    return {
      identifier: "org.aopenfx.transform",
      label: "Transform",
      grouping: "Transform",
      description:
        "Translate, rotate, scale and skew about a centre, with a choice of resampling filter. No motion blur.",
      inputs: [{ name: "Source", label: "Source" }],
      outputs: [COLOR_PLANE],
      params: [
        {
          name: "translate",
          label: "Translate",
          hint: "Translation along x and y, in pixels.",
          type: "double",
          role: "position",
          dimension: 2,
          defaults: [0, 0],
        },
        {
          name: "rotate",
          label: "Rotate",
          hint: "Rotation in degrees about the centre.",
          type: "double",
          role: "angle",
          defaults: [0],
          displayMin: [-180],
          displayMax: [180],
        },
        {
          name: "scale",
          label: "Scale",
          hint: "Scale factor along x and y.",
          type: "double",
          role: "scale",
          dimension: 2,
          defaults: [1, 1],
        },
        bool("uniform", "Uniform", "Use the X scale for both directions."),
        {
          name: "skewX",
          label: "Skew X",
          hint: "Skew along the axis, about the centre.",
          type: "double",
          defaults: [0],
        },
        {
          name: "skewY",
          label: "Skew Y",
          hint: "Skew along the axis, about the centre.",
          type: "double",
          defaults: [0],
        },
        {
          name: "skewOrder",
          label: "Skew Order",
          hint: "Which skew is applied first.",
          type: "choice",
          choices: [
            { value: "xy", label: "XY" },
            { value: "yx", label: "YX" },
          ],
          defaults: [0],
        },
        {
          name: "center",
          label: "Center",
          hint: "The point rotation and scale turn about, in pixels.",
          type: "double",
          role: "position",
          dimension: 2,
          // Half the frame at any format; unset, the input's middle.
          defaultsNormalised: true,
          defaults: [0.5, 0.5],
        },
        bool("invert", "Invert", "Apply the inverse transform."),
        {
          name: "filter",
          label: "Filter",
          hint: "How the source is resampled. Keys, Simon and Rifman overshoot on purpose.",
          type: "choice",
          choices: TRANSFORM_FILTERS.map((value) => ({ value, label: value })),
          defaults: [3],
        },
        bool("clamp", "Clamp", "Clamp the result to [0, 1]."),
        bool(
          "black_outside",
          "Black outside",
          "Outside the source is transparent black. Off, the edge pixels repeat outwards.",
          1,
        ),
      ],
    };
  }

  regionOfDefinition(q: RegionQuery): Rect {
    const input = unionRects(q.inputRods);
    if (isEmptyRect(input)) return input;
    const b = boundsOf(matrixOf(scaledSettings(q, input)), input);
    // Outwards, and one pixel more for the filter's tail, as the plugin.
    return {
      x1: Math.floor(b.minX) - 1,
      y1: Math.floor(b.minY) - 1,
      x2: Math.ceil(b.maxX) + 1,
      y2: Math.ceil(b.maxY) + 1,
    };
  }

  regionOfInterest(q: RegionQuery, output: Rect): Rect[] {
    const wanted = q.inputRods.map(() => output);
    if (q.inputRods.length === 0 || isEmptyRect(output)) return wanted;
    const back = invertAffine(matrixOf(scaledSettings(q, q.inputRods[0])));
    const b = boundsOf(back, output);
    // A cubic reads two pixels either side; Box reads its footprint, which
    // under minification is as wide as an output pixel is in the source.
    let margin = 3;
    if (Math.trunc(paramNumber(q.params, "filter", 3)) === FILTER_BOX) {
      const wu = Math.max(1, Math.abs(back.a) + Math.abs(back.b));
      const wv = Math.max(1, Math.abs(back.d) + Math.abs(back.e));
      const reach = Math.min(BOX_REACH, Math.ceil(Math.max(wu, wv) * 0.5));
      margin = Math.max(margin, reach + 1);
    }
    wanted[0] = {
      x1: Math.floor(b.minX) - margin,
      y1: Math.floor(b.minY) - margin,
      x2: Math.ceil(b.maxX) + margin,
      y2: Math.ceil(b.maxY) + margin,
    };
    return wanted;
  }

  isIdentity(request: RenderRequest) {
    const n = (name: string, f: number, c = 0) => request.number(name, f, c);
    return (
      n("translate", 0, 0) === 0 &&
      n("translate", 0, 1) === 0 &&
      n("rotate", 0) === 0 &&
      n("scale", 1, 0) === 1 &&
      n("scale", 1, 1) === 1 &&
      n("skewX", 0) === 0 &&
      n("skewY", 0) === 0
    );
  }

  process(request: RenderRequest): boolean {
    const input = request.input("Source");
    const source = input?.buffer;
    const target = request.output()?.buffer;
    if (!isValidBuffer(source) || !isValidBuffer(target)) return false;
    const s = settingsFrom(request.params);
    // Unset, the centre is the middle of the input's whole picture -- not of
    // the piece delivered, which follows what this node asked for.
    const frame = input && !isEmptyRect(input.rod) ? input.rod : source.rect;
    const sx = request.scaleX > 0 ? request.scaleX : 1;
    const sy = request.scaleY > 0 ? request.scaleY : 1;
    s.centerX = request.number("center", ((frame.x1 + frame.x2) * 0.5) / sx, 0);
    s.centerY = request.number("center", ((frame.y1 + frame.y2) * 0.5) / sy, 1);
    bringToScale(s, sx, sy);
    // The kernel maps output pixels backwards.
    const back = invertAffine(matrixOf(s));
    const filter = Math.trunc(request.number("filter", 3));
    const u = UniformWriter.for(transformModule).setAll({
      width: target.width,
      height: target.height,
      srcStride: source.stride,
      dstStride: target.stride,
      srcX1: source.rect.x1,
      srcY1: source.rect.y1,
      srcWidth: source.width,
      srcHeight: source.height,
      dstX1: target.rect.x1,
      dstY1: target.rect.y1,
      m0: back.a,
      m1: back.b,
      m2: back.c,
      m3: back.d,
      m4: back.e,
      m5: back.f,
      filter: Math.min(Math.max(filter, 0), TRANSFORM_FILTERS.length - 1),
      blackOutside: request.number("black_outside", 1) !== 0,
      doClamp: request.number("clamp", 0) !== 0,
    });
    return request.gpu.run(
      request.gpu.load(transformModule, "transformMain"),
      [target.width, target.height, 1],
      [source, target],
      u,
    );
  }
}
