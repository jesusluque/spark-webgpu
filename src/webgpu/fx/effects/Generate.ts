// aopenfx/examples/generate/Generate.cpp: six nodes that make a picture out
// of nothing (Constant, CheckerBoard, ColorBars, ColorWheel), pass one
// through (NoOp) or measure one (ImageStatistics).
//
// A generator has no input, so the default region of definition -- the union
// of the inputs -- is empty and it would render nothing. Each answers its own:
// EVERYWHERE, which the graph clamps to what it is asked for, or a box of
// `size` at the origin.

import generateModule from "../../generated/fx/generate";
import statsModule from "../../generated/fx/imagestats";
import { UniformWriter } from "../../uniforms";
import { Effect, type RegionQuery, type RenderRequest } from "../Effect";
import {
  COLOR_PLANE,
  EVERYWHERE,
  type EffectDesc,
  type FxBuffer,
  type ParamDesc,
  type ParamValues,
  type Rect,
  equalRects,
  isValidBuffer,
  paramNumber,
} from "../types";

export function sizedRod(
  params: ParamValues,
  scaleX: number,
  scaleY: number,
): Rect {
  const sx = scaleX > 0 ? scaleX : 1;
  const sy = scaleY > 0 ? scaleY : 1;
  const w = Math.round(paramNumber(params, "size", 0, 0) * sx);
  const h = Math.round(paramNumber(params, "size", 0, 1) * sy);
  return w <= 0 || h <= 0 ? EVERYWHERE : { x1: 0, y1: 0, x2: w, y2: h };
}

const SIZE_PARAM: ParamDesc = {
  name: "size",
  label: "Size",
  hint: "Zero means the project format; anything else is a box of that many pixels at the origin.",
  type: "integer",
  dimension: 2,
  defaults: [0, 0],
};

const colour = (
  name: string,
  label: string,
  hint: string,
  value: number[],
): ParamDesc => ({
  name,
  label,
  hint,
  type: "colour",
  dimension: 4,
  defaults: value,
});

/** The uniforms every pattern shares: where the buffer is, and the whole frame. */
function targetUniforms(request: RenderRequest, target: FxBuffer) {
  // The pattern's own frame, not the window asked for: a blur downstream
  // asking for a margin must not move every bar boundary.
  let frame = sizedRod(request.params, request.scaleX, request.scaleY);
  if (equalRects(frame, EVERYWHERE)) {
    frame =
      request.projectWidth > 0 && request.projectHeight > 0
        ? {
            x1: 0,
            y1: 0,
            x2: Math.round(request.projectWidth * request.scaleX),
            y2: Math.round(request.projectHeight * request.scaleY),
          }
        : request.outputRod;
  }
  return UniformWriter.for(generateModule).setAll({
    dstWidth: target.width,
    dstHeight: target.height,
    dstStride: target.stride,
    originX: target.rect.x1,
    originY: target.rect.y1,
    frameX1: frame.x1,
    frameY1: frame.y1,
    frameX2: frame.x2,
    frameY2: frame.y2,
  });
}

abstract class Generator extends Effect {
  constructor(private readonly entry: string) {
    super();
  }

  regionOfDefinition(q: RegionQuery): Rect {
    return sizedRod(q.params, q.scaleX, q.scaleY);
  }

  protected abstract fill(u: UniformWriter, request: RenderRequest): void;

  process(request: RenderRequest): boolean {
    const target = request.output("Color")?.buffer;
    if (!isValidBuffer(target)) return false;
    const u = targetUniforms(request, target);
    this.fill(u, request);
    return request.gpu.run(
      request.gpu.load(generateModule, this.entry),
      [target.width, target.height, 1],
      [target],
      u,
    );
  }
}

const colourOf = (request: RenderRequest, name: string, fallback: number[]) =>
  fallback.map((v, i) => request.number(name, v, i));

export class Constant extends Generator {
  constructor() {
    super("constantMain");
  }

  protected describeEffect(): EffectDesc {
    return {
      identifier: "org.aopenfx.constant",
      label: "Constant",
      grouping: "Image/Source",
      description: "A rectangle of one colour, alpha included.",
      inputs: [],
      outputs: [COLOR_PLANE],
      params: [
        colour("color", "Color", "The colour, alpha included.", [0, 0, 0, 1]),
        SIZE_PARAM,
      ],
    };
  }

  protected fill(u: UniformWriter, request: RenderRequest) {
    u.set("colour", colourOf(request, "color", [0, 0, 0, 1]));
  }
}

const DARK = [0.1, 0.1, 0.1, 1];
const LIGHT = [0.5, 0.5, 0.5, 1];
const WHITE = [1, 1, 1, 1];

export class CheckerBoard extends Generator {
  constructor() {
    super("checkerBoardMain");
  }

  protected describeEffect(): EffectDesc {
    const hint =
      "One square of the repeating 2x2 cell, left to right and then upward.";
    return {
      identifier: "org.aopenfx.checkerboard",
      label: "CheckerBoard",
      grouping: "Image/Source",
      description:
        "A checker of four colours, for seeing what a transform did: two cannot show a half-cell shift.",
      inputs: [],
      outputs: [COLOR_PLANE],
      params: [
        colour("color0", "Color 0", hint, DARK),
        colour("color1", "Color 1", hint, LIGHT),
        colour("color2", "Color 2", hint, LIGHT),
        colour("color3", "Color 3", hint, DARK),
        {
          name: "boxsize",
          label: "Box size",
          hint: "How big one square is, in pixels.",
          type: "double",
          dimension: 2,
          defaults: [64, 64],
        },
        colour(
          "linecolor",
          "Line color",
          "The grid drawn over the squares.",
          WHITE,
        ),
        {
          name: "linewidth",
          label: "Line width",
          hint: "How thick the grid is, in pixels. Zero draws none.",
          type: "double",
          defaults: [0],
        },
        SIZE_PARAM,
      ],
    };
  }

  protected fill(u: UniformWriter, request: RenderRequest) {
    const sx = request.scaleX > 0 ? request.scaleX : 1;
    const sy = request.scaleY > 0 ? request.scaleY : 1;
    u.setAll({
      checker0: colourOf(request, "color0", DARK),
      checker1: colourOf(request, "color1", LIGHT),
      checker2: colourOf(request, "color2", LIGHT),
      checker3: colourOf(request, "color3", DARK),
      lineColour: colourOf(request, "linecolor", WHITE),
      boxSize: [
        request.number("boxsize", 64, 0) * sx,
        request.number("boxsize", 64, 1) * sy,
      ],
      lineWidth: request.number("linewidth", 0) * sx,
    });
  }
}

export class ColorBars extends Generator {
  constructor() {
    super("colorBarsMain");
  }

  protected describeEffect(): EffectDesc {
    return {
      identifier: "org.aopenfx.colorbars",
      label: "ColorBars",
      grouping: "Image/Source",
      description:
        "SMPTE colour bars from the numbers: seven bars, the reversed blue strip, -I, +Q and PLUGE.",
      inputs: [],
      outputs: [COLOR_PLANE],
      params: [
        {
          name: "level",
          label: "Level",
          hint: "75% is what broadcast means by colour bars; 100% is for testing clipping.",
          type: "choice",
          choices: [
            { value: "75", label: "75%" },
            { value: "100", label: "100%" },
          ],
          defaults: [0],
        },
        SIZE_PARAM,
      ],
    };
  }

  protected fill(u: UniformWriter, request: RenderRequest) {
    u.set("barLevel", request.number("level", 0) >= 0.5 ? 1 : 0.75);
  }
}

const WHEEL_KNOBS = [
  [
    "centersaturation",
    "Center saturation",
    "How saturated the middle is.",
    0,
    "centreSaturation",
  ],
  [
    "edgesaturation",
    "Edge saturation",
    "How saturated the rim is.",
    1,
    "edgeSaturation",
  ],
  [
    "centervalue",
    "Center value",
    "How bright the middle is.",
    1,
    "centreValue",
  ],
  ["edgevalue", "Edge value", "How bright the rim is.", 1, "edgeValue"],
  ["gamma", "Gamma", "Bends the brightness across the disc.", 1, "wheelGamma"],
  [
    "rotate",
    "Rotate",
    "Turns the wheel, in degrees. Red at three o'clock at zero.",
    0,
    "rotate",
  ],
] as const;

export class ColorWheel extends Generator {
  constructor() {
    super("colorWheelMain");
  }

  protected describeEffect(): EffectDesc {
    return {
      identifier: "org.aopenfx.colorwheel",
      label: "ColorWheel",
      grouping: "Image/Source",
      description:
        "Hue around, saturation outward. Outside the disc is transparent, so it composites without a square.",
      inputs: [],
      outputs: [COLOR_PLANE],
      params: [
        ...WHEEL_KNOBS.map(([name, label, hint, value]) => ({
          name,
          label,
          hint,
          type: "double" as const,
          defaults: [value],
        })),
        SIZE_PARAM,
      ],
    };
  }

  protected fill(u: UniformWriter, request: RenderRequest) {
    for (const [name, , , value, field] of WHEEL_KNOBS) {
      u.set(field, request.number(name, value));
    }
  }
}

/** Passes the picture through: always an identity, so it is never rendered. */
export class NoOp extends Effect {
  protected describeEffect(): EffectDesc {
    return {
      identifier: "org.aopenfx.noop",
      label: "NoOp",
      grouping: "Other",
      description: "Passes the picture through untouched.",
      inputs: [{ name: "Source", label: "Source" }],
      outputs: [COLOR_PLANE],
      params: [],
    };
  }

  isIdentity() {
    return true;
  }

  process() {
    return true;
  }
}

/** Minimum, maximum, mean and standard deviation per channel, in that order. */
export type ImageStats = number[];

/**
 * Measures the picture and passes it through. The sixteen numbers are
 * attached as `stats.<instance>`; WebGPU cannot read back inside a frame, so
 * they arrive as a promise that resolves once the frame is flushed.
 */
export class ImageStatistics extends Effect {
  protected describeEffect(): EffectDesc {
    return {
      identifier: "org.aopenfx.imagestatistics",
      label: "ImageStatistics",
      grouping: "Measure",
      description:
        "Minimum, maximum, mean and standard deviation of each channel, over the frame or a rectangle.",
      inputs: [{ name: "Source", label: "Source" }],
      outputs: [COLOR_PLANE],
      params: [
        {
          name: "restrict",
          label: "Restrict to region",
          hint: "Measures inside the rectangle below instead of the whole frame.",
          type: "boolean",
          defaults: [0],
        },
        {
          name: "region",
          label: "Region",
          hint: "The rectangle to measure, in pixels: left, bottom, right, top.",
          type: "double",
          dimension: 4,
          defaults: [0, 0, 0, 0],
        },
      ],
    };
  }

  process(request: RenderRequest): boolean {
    const input = request.input("Source")?.buffer;
    const target = request.output("Color")?.buffer;
    if (!isValidBuffer(input) || !isValidBuffer(target)) return false;
    const clampTo = (v: number, hi: number) => Math.min(Math.max(v, 0), hi);
    let [x1, y1, x2, y2] = [0, 0, input.width, input.height];
    if (request.number("restrict") >= 0.5) {
      x1 = clampTo(
        Math.floor(request.number("region", 0, 0)) - input.rect.x1,
        input.width,
      );
      y1 = clampTo(
        Math.floor(request.number("region", 0, 1)) - input.rect.y1,
        input.height,
      );
      x2 = clampTo(
        Math.ceil(request.number("region", 0, 2)) - input.rect.x1,
        input.width,
      );
      y2 = clampTo(
        Math.ceil(request.number("region", 0, 3)) - input.rect.y1,
        input.height,
      );
    }
    const rows = Math.max(0, y2 - y1);
    const u = UniformWriter.for(statsModule).setAll({
      srcWidth: input.width,
      srcHeight: input.height,
      srcStride: input.stride,
      x1,
      y1,
      x2,
      y2,
      rows,
      dstWidth: target.width,
      dstHeight: target.height,
      dstStride: target.stride,
      srcOffsetX: target.rect.x1 - input.rect.x1,
      srcOffsetY: target.rect.y1 - input.rect.y1,
    });
    const gpu = request.gpu;
    // A row of accumulators per thread, folded by a second dispatch: no
    // groupshared, so nothing depends on the group's shape.
    const sums = gpu.scratch(4 + rows * 4, 1);
    const bound = [input, sums, target];
    gpu.run(
      gpu.load(statsModule, "statsCopy"),
      [target.width, target.height, 1],
      bound,
      u,
    );
    if (rows === 0 || x2 <= x1) return true;
    gpu.run(gpu.load(statsModule, "statsRow"), [rows, 1, 1], bound, u);
    gpu.run(gpu.load(statsModule, "statsFold"), [4, 1, 1], bound, u);
    const count = (x2 - x1) * (y2 - y1);
    request.attach(
      `stats.${request.instance}`,
      gpu.read(sums, 64).then((data) => {
        const a = new Float32Array(data);
        const out = Array.from(a.subarray(0, 8));
        const means = [0, 1, 2, 3].map((c) => a[8 + c] / count);
        out.push(...means);
        for (let c = 0; c < 4; c++) {
          out.push(Math.sqrt(Math.max(0, a[12 + c] / count - means[c] ** 2)));
        }
        return out;
      }),
    );
    return true;
  }
}
