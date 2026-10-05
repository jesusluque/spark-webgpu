// aopenfx/examples/blur/Blur.cpp and BlurMath.h: a separable gaussian, along
// x into a scratch and along y out of it. CImgBlur's convention, sigma =
// size / 2.4, cut at three sigmas. The picture grows by the reach: a blur
// makes picture outside its input, which is what the soft edge is.

import blurModule from "../../generated/fx/blur";
import { UniformWriter } from "../../uniforms";
import { Effect, type RegionQuery, type RenderRequest } from "../Effect";
import {
  COLOR_PLANE,
  type EffectDesc,
  type FxBuffer,
  type ParamValues,
  type Rect,
  isEmptyRect,
  isValidBuffer,
  paramNumber,
  placed,
  unionRects,
} from "../types";

export const RADIUS_IN_SIGMAS = 3;
export const SIGMA_PER_SIZE = 1 / 2.4;

export const sigmaFrom = (size: number) => Math.max(size, 0) * SIGMA_PER_SIZE;

/** The allowance is for floating point: 8 / 2.4 * 3 is ten, not eleven. */
export const radiusFrom = (sigma: number) =>
  sigma <= 0 ? 0 : Math.ceil(sigma * RADIUS_IN_SIGMAS - 1e-9);

export interface Reach {
  sigmaX: number;
  sigmaY: number;
  radiusX: number;
  radiusY: number;
}

export function reachAt(
  sizeX: number,
  sizeY: number,
  scaleX: number,
  scaleY: number,
): Reach {
  const sigmaX = sigmaFrom(sizeX) * Math.max(scaleX, 0);
  const sigmaY = sigmaFrom(sizeY) * Math.max(scaleY, 0);
  return {
    sigmaX,
    sigmaY,
    radiusX: radiusFrom(sigmaX),
    radiusY: radiusFrom(sigmaY),
  };
}

/** A single number means both axes. */
function sizeOf(params: ParamValues): [number, number] {
  const x = paramNumber(params, "size", 0, 0);
  return [x, paramNumber(params, "size", x, 1)];
}

export function grown(r: Rect, reach: Reach): Rect {
  if (isEmptyRect(r)) return r;
  return {
    x1: r.x1 - reach.radiusX,
    y1: r.y1 - reach.radiusY,
    x2: r.x2 + reach.radiusX,
    y2: r.y2 + reach.radiusY,
  };
}

export class Blur extends Effect {
  protected describeEffect(): EffectDesc {
    return {
      identifier: "org.aopenfx.blur",
      label: "Blur",
      grouping: "Filter",
      description:
        "A gaussian blur on the GPU. Separable, so its cost is linear in the radius.",
      inputs: [{ name: "Source", label: "Source" }],
      outputs: [COLOR_PLANE],
      params: [
        {
          name: "size",
          label: "Size",
          hint: "Blur diameter in pixels, horizontally and vertically; sigma is size/2.4, as CImgBlur's.",
          type: "double",
          dimension: 2,
          defaults: [0, 0],
          hardMin: [0, 0],
          displayMax: [100, 100],
        },
      ],
    };
  }

  regionOfDefinition(q: RegionQuery): Rect {
    const [sx, sy] = sizeOf(q.params);
    return grown(unionRects(q.inputRods), reachAt(sx, sy, q.scaleX, q.scaleY));
  }

  /** Every output pixel sums neighbours outside its own rectangle. */
  regionOfInterest(q: RegionQuery, output: Rect): Rect[] {
    const [sx, sy] = sizeOf(q.params);
    const r = grown(output, reachAt(sx, sy, q.scaleX, q.scaleY));
    return Array.from({ length: Math.max(q.inputRods.length, 1) }, () => r);
  }

  isIdentity(request: RenderRequest) {
    const [sx, sy] = sizeOf(request.params);
    const reach = reachAt(sx, sy, request.scaleX, request.scaleY);
    return reach.radiusX === 0 && reach.radiusY === 0;
  }

  process(request: RenderRequest): boolean {
    const source = request.input("Source")?.buffer;
    const target = request.output()?.buffer;
    if (!isValidBuffer(source) || !isValidBuffer(target)) return false;
    const gpu = request.gpu;
    const kernel = gpu.load(blurModule, "blurMain");
    const [sx, sy] = sizeOf(request.params);
    const reach = reachAt(sx, sy, request.scaleX, request.scaleY);

    // Taller than the output by the vertical reach, so the second pass reads
    // picture above and below the window rather than transparent black.
    const t = target.rect;
    const scratch: FxBuffer = placed(
      gpu.scratch(target.width, target.height + 2 * reach.radiusY),
      {
        x1: t.x1,
        y1: t.y1 - reach.radiusY,
        x2: t.x2,
        y2: t.y2 + reach.radiusY,
      },
    );

    const pass = (
      from: FxBuffer,
      into: FxBuffer,
      stepX: number,
      sigma: number,
      radius: number,
    ) =>
      gpu.run(
        kernel,
        [into.width, into.height, 1],
        [from, into],
        UniformWriter.for(blurModule).setAll({
          width: into.width,
          height: into.height,
          srcWidth: from.width,
          srcHeight: from.height,
          srcOffsetX: into.rect.x1 - from.rect.x1,
          srcOffsetY: into.rect.y1 - from.rect.y1,
          srcStride: from.stride,
          dstStride: into.stride,
          stepX,
          stepY: 1 - stepX,
          sigma,
          radius,
        }),
      );
    // One kernel per axis, never averaged: a horizontal-only blur is real.
    return (
      pass(source, scratch, 1, reach.sigmaX, reach.radiusX) &&
      pass(scratch, target, 0, reach.sigmaY, reach.radiusY)
    );
  }
}
