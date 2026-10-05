// aopenfx/examples/invert/Invert.cpp: one minus the picture, with InvertOFX's
// controls. Which channels are inverted is the host's R/G/B/A switches
// (FxNode.channels), not a parameter.

import invertModule from "../../generated/fx/invert";
import { UniformWriter } from "../../uniforms";
import { Effect, type RenderRequest } from "../Effect";
import { COLOR_PLANE, type EffectDesc, isValidBuffer } from "../types";

const clamp = (v: number, lo: number, hi: number) =>
  Math.min(hi, Math.max(lo, v));

export class Invert extends Effect {
  protected describeEffect(): EffectDesc {
    return {
      identifier: "org.aopenfx.invert",
      label: "Invert",
      grouping: "Colour",
      description:
        "One minus the picture. (Un)premult divides by a channel first, so a soft edge does not invert brighter than the middle of its shape.",
      inputs: [
        { name: "Source", label: "Source", passThrough: true },
        // The alpha of what is wired here says how much of the invert each
        // pixel receives.
        { name: "Mask", label: "Mask", optional: true, isMask: true },
      ],
      outputs: [COLOR_PLANE],
      params: [
        {
          name: "premult",
          label: "(Un)premult",
          type: "boolean",
          defaults: [0],
          hint: "Divide the image by a channel before inverting and multiply it back afterwards.",
        },
        {
          name: "premultChannel",
          label: "By",
          type: "choice",
          choices: [
            { value: "r", label: "R" },
            { value: "g", label: "G" },
            { value: "b", label: "B" },
            { value: "a", label: "A" },
          ],
          defaults: [3],
          shownWhen: { param: "premult", is: "1" },
          hint: "Which channel to divide by and multiply back.",
        },
        {
          name: "maskInvert",
          label: "Invert Mask",
          type: "boolean",
          defaults: [0],
          hint: "The invert lands where the mask is dark instead of where it is bright.",
        },
        {
          name: "mix",
          label: "Mix",
          type: "double",
          defaults: [1],
          displayMin: [0],
          displayMax: [1],
          hardMin: [0],
          hardMax: [1],
          hint: "How much of the invert to keep. Zero is the input, and is not rendered at all.",
        },
      ],
    };
  }

  isIdentity(request: RenderRequest) {
    return request.number("mix", 1) <= 0;
  }

  process(request: RenderRequest): boolean {
    const source = request.input("Source")?.buffer;
    const target = request.output("Color")?.buffer;
    if (!isValidBuffer(source) || !isValidBuffer(target)) return false;
    const gpu = request.gpu;
    const kernel = gpu.load(invertModule, "invertMain");

    const mask = request.input("Mask")?.buffer;
    const hasMask = isValidBuffer(mask);
    const u = UniformWriter.for(invertModule).setAll({
      srcWidth: source.width,
      srcHeight: source.height,
      srcStride: source.stride,
      srcOffsetX: target.rect.x1 - source.rect.x1,
      srcOffsetY: target.rect.y1 - source.rect.y1,
      dstWidth: target.width,
      dstHeight: target.height,
      dstStride: target.stride,
      // The mask is always bound -- the kernel declares it -- and `hasMask`
      // is what stops it being read.
      hasMask,
      maskInvert: request.number("maskInvert") >= 0.5,
      premult: request.number("premult") >= 0.5,
      premultChannel: clamp(
        Math.floor(request.number("premultChannel", 3) + 0.5),
        0,
        3,
      ),
      mix: clamp(request.number("mix", 1), 0, 1),
    });
    if (hasMask) {
      u.setAll({
        maskWidth: mask.width,
        maskHeight: mask.height,
        maskStride: mask.stride,
        maskOffsetX: target.rect.x1 - mask.rect.x1,
        maskOffsetY: target.rect.y1 - mask.rect.y1,
      });
    }
    return gpu.run(
      kernel,
      [target.width, target.height, 1],
      [source, hasMask ? mask : source, target],
      u,
    );
  }
}
