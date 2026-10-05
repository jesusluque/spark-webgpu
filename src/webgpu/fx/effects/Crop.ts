// aopenfx/examples/crop/Crop.cpp: a box, and a tick that says whether it
// cuts. Blacked, the frame keeps its size; cut, the region of definition is
// the box and everything downstream works on that many pixels.

import cropModule from "../../generated/fx/crop";
import { UniformWriter } from "../../uniforms";
import { Effect, type RegionQuery, type RenderRequest } from "../Effect";
import {
  COLOR_PLANE,
  type EffectDesc,
  type ParamValues,
  type Rect,
  isEmptyRect,
  isValidBuffer,
  paramNumber,
  paramNumbers,
} from "../types";

interface Box {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

/**
 * The box, ordered. Both corners left at zero is a node just made: the whole
 * picture, so a crop does nothing until it is dragged.
 */
export function cropBox(params: ParamValues, source: Rect): Box {
  const one = paramNumbers(params.corner1);
  const two = paramNumbers(params.corner2);
  if (
    one.length < 2 ||
    two.length < 2 ||
    (one[0] === 0 && one[1] === 0 && two[0] === 0 && two[1] === 0)
  ) {
    return { ...source };
  }
  return {
    x1: Math.min(one[0], two[0]),
    y1: Math.min(one[1], two[1]),
    x2: Math.max(one[0], two[0]),
    y2: Math.max(one[1], two[1]),
  };
}

const cuts = (params: ParamValues) => paramNumber(params, "cut") >= 0.5;

export class Crop extends Effect {
  protected describeEffect(): EffectDesc {
    const corner = {
      type: "double",
      dimension: 2,
      role: "position",
      defaults: [0, 0],
    } as const;
    return {
      identifier: "org.aopenfx.crop",
      label: "Crop",
      grouping: "Transform",
      description:
        "A rectangle. With Cut off, outside it goes black and the frame keeps its size; on, the picture is cut down to it.",
      inputs: [{ name: "Source", label: "Source", passThrough: true }],
      outputs: [COLOR_PLANE],
      params: [
        {
          ...corner,
          name: "corner1",
          label: "Corner 1",
          hint: "One corner of the box, in pixels.",
        },
        {
          ...corner,
          name: "corner2",
          label: "Corner 2",
          hint: "The opposite corner. Both at zero means the whole picture.",
        },
        {
          name: "cut",
          label: "Cut",
          type: "boolean",
          defaults: [0],
          hint: "Cut the picture down to the box instead of blacking what is outside it.",
        },
        {
          name: "softness",
          label: "Softness",
          type: "double",
          defaults: [0],
          displayMin: [0],
          displayMax: [100],
          hardMin: [0],
          hardMax: [2000],
          hint: "How many pixels the edge fades over. No effect with Cut on.",
        },
      ],
    };
  }

  regionOfDefinition(q: RegionQuery): Rect {
    const input = q.inputRods[0] ?? { x1: 0, y1: 0, x2: 0, y2: 0 };
    if (!cuts(q.params) || isEmptyRect(input)) return input;
    const box = cropBox(q.params, input);
    if (box.x2 <= box.x1 || box.y2 <= box.y1) return input;
    // Intersected with the input, never grown past it: that would be a pad.
    const out = {
      x1: Math.max(input.x1, Math.floor(box.x1)),
      y1: Math.max(input.y1, Math.floor(box.y1)),
      x2: Math.min(input.x2, Math.ceil(box.x2)),
      y2: Math.min(input.y2, Math.ceil(box.y2)),
    };
    return isEmptyRect(out) ? input : out;
  }

  isIdentity(request: RenderRequest) {
    const source = request.input("Source")?.buffer;
    if (!source) return false;
    const box = cropBox(request.params, source.rect);
    const r = source.rect;
    return (
      !cuts(request.params) &&
      box.x1 <= r.x1 &&
      box.y1 <= r.y1 &&
      box.x2 >= r.x2 &&
      box.y2 >= r.y2
    );
  }

  process(request: RenderRequest): boolean {
    const source = request.input("Source")?.buffer;
    const target = request.output("Color")?.buffer;
    if (!isValidBuffer(source) || !isValidBuffer(target)) return false;
    const box = cropBox(request.params, source.rect);
    const u = UniformWriter.for(cropModule).setAll({
      srcWidth: source.width,
      srcHeight: source.height,
      srcStride: source.stride,
      srcOffsetX: target.rect.x1 - source.rect.x1,
      srcOffsetY: target.rect.y1 - source.rect.y1,
      dstWidth: target.width,
      dstHeight: target.height,
      dstStride: target.stride,
      dstOriginX: target.rect.x1,
      dstOriginY: target.rect.y1,
      x1: box.x1,
      y1: box.y1,
      x2: box.x2,
      y2: box.y2,
      softness: Math.max(request.number("softness"), 0),
      // Cut, the output is the box and there is nothing outside to black;
      // blacking anyway would eat the box's own edge by the softness.
      keepOutside: cuts(request.params),
    });
    return request.gpu.run(
      request.gpu.load(cropModule, "cropMain"),
      [target.width, target.height, 1],
      [source, target],
      u,
    );
  }
}
