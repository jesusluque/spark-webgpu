// aopenfx/examples/cornerpin/CornerPin.cpp: the picture pinned to four
// corners with perspective, mapped backwards (output pixel to source) so it
// never leaves holes. `crown` bows the middle through a 3x3 mesh of offsets,
// kept on the GPU by value. Corners and motion hung on the Track (or Source)
// picture win over the parameters, as in the C++; a GeometricTrack's mesh
// plane is not read, since WebGPU has no blocking readback.

import cornerpinModule from "../../generated/fx/cornerpin";
import { UniformWriter } from "../../uniforms";
import { Effect, type RegionQuery, type RenderRequest } from "../Effect";
import {
  COLOR_PLANE,
  type EffectDesc,
  type ParamDesc,
  type Rect,
  isEmptyRect,
  isValidBuffer,
  paramNumber,
  paramNumbers,
  unionRects,
} from "../types";

type Point = readonly [number, number];
/** Row major, nine numbers. */
export type Matrix3 = number[];

const IDENTITY3 = [1, 0, 0, 0, 1, 0, 0, 0, 1];

/** Heckbert's map from the unit square's corners (0,0),(1,0),(1,1),(0,1) to a quad. */
export function squareToQuad(q: readonly Point[]): Matrix3 {
  const sx = q[0][0] - q[1][0] + q[2][0] - q[3][0];
  const sy = q[0][1] - q[1][1] + q[2][1] - q[3][1];
  if (Math.abs(sx) < 1e-12 && Math.abs(sy) < 1e-12) {
    return [
      q[1][0] - q[0][0],
      q[3][0] - q[0][0],
      q[0][0],
      q[1][1] - q[0][1],
      q[3][1] - q[0][1],
      q[0][1],
      0,
      0,
      1,
    ];
  }
  const dx1 = q[1][0] - q[2][0];
  const dx2 = q[3][0] - q[2][0];
  const dy1 = q[1][1] - q[2][1];
  const dy2 = q[3][1] - q[2][1];
  const den = dx1 * dy2 - dx2 * dy1;
  if (Math.abs(den) < 1e-12) return [...IDENTITY3];
  const g = (sx * dy2 - dx2 * sy) / den;
  const h = (dx1 * sy - sx * dy1) / den;
  return [
    q[1][0] - q[0][0] + g * q[1][0],
    q[3][0] - q[0][0] + h * q[3][0],
    q[0][0],
    q[1][1] - q[0][1] + g * q[1][1],
    q[3][1] - q[0][1] + h * q[3][1],
    q[0][1],
    g,
    h,
    1,
  ];
}

export function multiply3(a: Matrix3, b: Matrix3): Matrix3 {
  const out = new Array(9).fill(0);
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      out[r * 3 + c] =
        a[r * 3] * b[c] + a[r * 3 + 1] * b[3 + c] + a[r * 3 + 2] * b[6 + c];
    }
  }
  return out;
}

export function invert3(m: Matrix3): Matrix3 {
  const det =
    m[0] * (m[4] * m[8] - m[5] * m[7]) -
    m[1] * (m[3] * m[8] - m[5] * m[6]) +
    m[2] * (m[3] * m[7] - m[4] * m[6]);
  if (Math.abs(det) < 1e-12) return [...IDENTITY3];
  const s = 1 / det;
  return [
    (m[4] * m[8] - m[5] * m[7]) * s,
    (m[2] * m[7] - m[1] * m[8]) * s,
    (m[1] * m[5] - m[2] * m[4]) * s,
    (m[5] * m[6] - m[3] * m[8]) * s,
    (m[0] * m[8] - m[2] * m[6]) * s,
    (m[2] * m[3] - m[0] * m[5]) * s,
    (m[3] * m[7] - m[4] * m[6]) * s,
    (m[1] * m[6] - m[0] * m[7]) * s,
    (m[0] * m[4] - m[1] * m[3]) * s,
  ];
}

/** Bottom right one, which the kernel assumes so eight floats are enough. */
export function normalised3(m: Matrix3): Matrix3 {
  return Math.abs(m[8]) < 1e-12 ? m : m.map((v) => v / m[8]);
}

/** The corners of a rectangle in pin order: anticlockwise from the bottom left. */
export const rectCorners = (r: Rect): Point[] => [
  [r.x1, r.y1],
  [r.x2, r.y1],
  [r.x2, r.y2],
  [r.x1, r.y2],
];

/** The backward map (output to source) of `from` pinned to `to`. */
export function pinMatrix(
  from: readonly Point[],
  to: readonly Point[],
  invert = false,
) {
  const square = squareToQuad(from);
  const quad = squareToQuad(to);
  return normalised3(
    invert
      ? multiply3(quad, invert3(square))
      : multiply3(square, invert3(quad)),
  );
}

export class CornerPin extends Effect {
  /** CornerPinSS: the same pin with the supersampled filter, under its own name. */
  constructor(readonly supersampled = false) {
    super();
  }

  protected describeEffect(): EffectDesc {
    const ss = this.supersampled;
    const labels = ["Bottom left", "Bottom right", "Top right", "Top left"];
    const defaults = [
      [0, 0],
      [1920, 0],
      [1920, 1080],
      [0, 1080],
    ];
    const corners: ParamDesc[] = labels.map((label, i) => ({
      name: `corner${i + 1}`,
      label,
      type: "double",
      dimension: 2,
      role: "position",
      defaults: defaults[i],
      hint: "Where this corner of the picture ends up, in the project's pixels.",
    }));
    return {
      identifier: ss ? "org.aopenfx.cornerpinss" : "org.aopenfx.cornerpin",
      label: ss ? "CornerPinSS" : "Corner Pin",
      grouping: "Track",
      description:
        "Pins the picture to four corners, with perspective. Crown bows the middle, as a road is domed.",
      inputs: [
        { name: "Source", label: "Source", passThrough: true },
        // Nothing is read from its pixels; it carries tracked corners.
        { name: "Track", label: "Track", optional: true },
      ],
      outputs: [COLOR_PLANE],
      params: [
        ...corners,
        {
          name: "crown",
          label: "Crown",
          type: "double",
          defaults: [0],
          displayMin: [-40],
          displayMax: [40],
          hardMin: [-400],
          hardMax: [400],
          hint: "Bow the middle of the pin, in source pixels. Zero is a true plane.",
        },
        {
          name: "softness",
          label: "Edge softness",
          type: "double",
          defaults: [1],
          displayMin: [0],
          displayMax: [20],
          hardMin: [0],
          hardMax: [200],
          hint: "How wide the ramp at the edge of the pinned picture is, in source pixels.",
        },
        {
          name: "filter",
          label: "Filter",
          type: "choice",
          choices: ss
            ? [
                { value: "supersampled", label: "Supersampled" },
                { value: "bilinear", label: "Bilinear" },
                { value: "nearest", label: "Nearest" },
              ]
            : [
                { value: "nearest", label: "Nearest" },
                { value: "bilinear", label: "Bilinear" },
              ],
          defaults: [ss ? 0 : 1],
          hint: "How the source is read between its pixels.",
        },
        {
          name: "black_outside",
          label: "Black outside",
          type: "boolean",
          defaults: [1],
          hint: "Outside the source is transparent. Off, the edge pixels repeat outwards.",
        },
        {
          name: "motion_blur",
          label: "Motion blur",
          type: "boolean",
          defaults: [1],
          hint: "Blur along the motion a tracker hung on Track as `motion`.",
        },
        {
          name: "shutter",
          label: "Shutter",
          type: "double",
          defaults: [0.5],
          hardMin: [0],
          displayMax: [1],
          hint: "The fraction of a frame the exposure lasts.",
        },
        {
          name: "invert",
          label: "Invert",
          type: "boolean",
          defaults: [0],
          hint: "Map the other way: straighten what is inside the corners into the frame.",
        },
      ],
    };
  }

  /** The box round the four corners, one pixel more for the filter's tail. */
  regionOfDefinition(q: RegionQuery): Rect {
    if (q.inputRods.length >= 2 && !isEmptyRect(q.inputRods[1])) {
      return q.inputRods[1];
    }
    const input = unionRects(q.inputRods.slice(0, 1));
    if (isEmptyRect(input)) return input;
    if (paramNumber(q.params, "invert") !== 0) return input;
    const sx = q.scaleX > 0 ? q.scaleX : 1;
    const sy = q.scaleY > 0 ? q.scaleY : 1;
    const pts = rectCorners(input).map(([x, y], i) => {
      const v = paramNumbers(q.params[`corner${i + 1}`]);
      return v.length >= 2 ? [v[0] * sx, v[1] * sy] : [x, y];
    });
    return {
      x1: Math.floor(Math.min(...pts.map((p) => p[0]))) - 1,
      y1: Math.floor(Math.min(...pts.map((p) => p[1]))) - 1,
      x2: Math.ceil(Math.max(...pts.map((p) => p[0]))) + 1,
      y2: Math.ceil(Math.max(...pts.map((p) => p[1]))) + 1,
    };
  }

  /** All of the source, always: where the quad is says nothing of what it reads. */
  regionOfInterest(q: RegionQuery, output: Rect): Rect[] {
    return q.inputRods.map((rod) => (isEmptyRect(rod) ? output : rod));
  }

  process(request: RenderRequest): boolean {
    const sourcePlane = request.input("Source");
    const source = sourcePlane?.buffer;
    const target = request.output("Color")?.buffer;
    if (!sourcePlane || !isValidBuffer(source) || !isValidBuffer(target)) {
      return false;
    }
    const gpu = request.gpu;
    const sx = request.scaleX > 0 ? request.scaleX : 1;
    const sy = request.scaleY > 0 ? request.scaleY : 1;
    const ss = this.supersampled;
    const chosen = Math.min(
      Math.max(Math.trunc(request.number("filter", ss ? 0 : 1)), 0),
      2,
    );
    const u = UniformWriter.for(cornerpinModule).setAll({
      srcWidth: source.width,
      srcHeight: source.height,
      srcStride: source.stride,
      srcOriginX: source.rect.x1,
      srcOriginY: source.rect.y1,
      dstWidth: target.width,
      dstHeight: target.height,
      dstStride: target.stride,
      dstOriginX: target.rect.x1,
      dstOriginY: target.rect.y1,
      softness: Math.max(request.number("softness", 1), 0),
      // The kernel numbers nearest 0, bilinear 1, supersampled 2; CornerPinSS
      // lists its choices the other way round.
      filter: ss ? 2 - chosen : chosen,
      blackOutside: request.number("black_outside", 1) !== 0,
      shutter: 0.5,
    });

    const input = source.rect;
    // Track first, then Source, then the parameters.
    const track = request.input("Track");
    const trackSilent =
      !!track && isValidBuffer(track.buffer) && !track.values?.get("corners");
    const carrier = track?.values?.get("corners") ? track : sourcePlane;
    let to = rectCorners(input).map(([x, y], i) => {
      const name = `corner${i + 1}`;
      return [request.number(name, x, 0), request.number(name, y, 1)] as const;
    });
    const typed = request.has("corner1");
    const tracked = carrier.values?.get("corners");
    if (tracked && tracked.length === 8) {
      to = to.map((_, i) => [tracked[i * 2], tracked[i * 2 + 1]] as const);
    }
    // Typed or tracked corners are in the document's pixels.
    if (typed || tracked?.length === 8) {
      to = to.map(([x, y]) => [x * sx, y * sy] as const);
    }
    let back = pinMatrix(
      rectCorners(input),
      to,
      request.number("invert") !== 0,
    );
    if (trackSilent) {
      // A tracker that lost its target: every pixel fetches from far outside.
      back = [0, 0, -1e5, 0, 0, -1e5, 0, 0, 1];
      u.set("blackOutside", 1);
    }
    for (let i = 0; i < 8; i++) u.set(`h${i}`, back[i]);

    // One kept buffer per crown value, not per node: two pins with one crown
    // want the same nine offsets, and a drag wants a new buffer rather than
    // a rewrite of one a frame in flight may still read.
    const crown = request.number("crown", 0);
    const flat = Math.abs(crown) <= 1e-6;
    const knots = new Float32Array(flat ? 4 : 20);
    if (!flat) knots[4 * 2 + 1] = -crown;
    u.setAll({ meshX: flat ? 1 : 3, meshY: flat ? 1 : 3 });
    const mesh = gpu.keep(`cornerpin.mesh.${Math.round(crown * 100)}`, knots);
    if (!mesh) return false;

    // Motion blur from the corners' travel since the last frame, hung on
    // the carrier by a tracker.
    const shutter = request.number("shutter", 0.5);
    const motion =
      request.number("motion_blur", 1) >= 0.5 && shutter > 0
        ? carrier.values?.get("motion")
        : undefined;
    let blurOn = false;
    if (motion && motion.length === 8) {
      let travel = 0;
      for (let i = 0; i < 4; i++) {
        const mx = motion[i * 2] * sx;
        const my = motion[i * 2 + 1] * sy;
        u.set(`m${i}x`, mx).set(`m${i}y`, my);
        travel += Math.hypot(mx, my);
      }
      u.set("shutter", shutter);
      blurOn = (travel * shutter) / 4 >= 0.75;
    }
    u.set("blurOn", blurOn);
    const warp = gpu.load(cornerpinModule, "pinWarp");
    const grid = [target.width, target.height, 1] as const;
    if (!blurOn) return gpu.run(warp, grid, [source, mesh, target], u);
    const warped = gpu.scratch(target.stride, target.height);
    return (
      gpu.run(warp, grid, [source, mesh, warped], u) &&
      gpu.run(
        gpu.load(cornerpinModule, "pinBlur"),
        grid,
        [warped, mesh, target],
        u,
      )
    );
  }
}

export class CornerPinSS extends CornerPin {
  constructor() {
    super(true);
  }
}
