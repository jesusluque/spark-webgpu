// aopenfx/examples/grade/Grade.cpp: two points in, two points out, then
// multiply, offset and gamma -- net.sf.openfx.GradePlugin's arithmetic and
// names. Declares a second plane, Clip: a matte of what the grade pushed out
// of range, produced only when somebody asks for it.

import gradeModule from "../../generated/fx/grade";
import { UniformWriter } from "../../uniforms";
import { Effect, type RenderRequest } from "../Effect";
import { COLOR_PLANE, type EffectDesc, isValidBuffer } from "../types";

export const GRADE_KNOBS = [
  [
    "blackPoint",
    "Black Point",
    "The colour of the darkest pixels in the image.",
    0,
  ],
  [
    "whitePoint",
    "White Point",
    "The colour of the brightest pixels in the image.",
    1,
  ],
  ["black", "Lift", "Colours at the black point are set to this.", 0],
  ["white", "Gain", "Colours at the white point are set to this.", 1],
  ["multiply", "Multiply", "Multiplies the result by this.", 1],
  [
    "offset",
    "Offset",
    "Added to the result, after the black and white points.",
    0,
  ],
  ["gamma", "Gamma", "Final gamma. Negative values are left alone.", 1],
] as const;

const SWITCHES = [
  [
    "clampBlack",
    "Clamp Black",
    "Everything below zero on output becomes zero.",
    true,
  ],
  [
    "clampWhite",
    "Clamp White",
    "Everything above one on output becomes one.",
    false,
  ],
  [
    "reverse",
    "Reverse",
    "Apply the exact inverse correction, gamma first and then the line.",
    false,
  ],
] as const;

export class Grade extends Effect {
  protected describeEffect(): EffectDesc {
    return {
      identifier: "org.aopenfx.grade",
      label: "Grade",
      grouping: "Colour",
      description:
        "Two points in, two points out, then multiply, offset and gamma. The Grade plugin's arithmetic, on the GPU.",
      inputs: [{ name: "Source", label: "Source" }],
      outputs: [
        COLOR_PLANE,
        { id: "Clip", label: "Clipping matte", channels: ["R", "G", "B", "A"] },
      ],
      params: [
        // Colours, alpha included: a grade that could not touch alpha could
        // not pull a matte back.
        ...GRADE_KNOBS.map(([name, label, hint, value]) => ({
          name,
          label,
          hint,
          type: "colour" as const,
          dimension: 4,
          defaults: [value, value, value, value],
        })),
        ...SWITCHES.map(([name, label, hint, on]) => ({
          name,
          label,
          hint,
          type: "boolean" as const,
          defaults: [on ? 1 : 0],
        })),
      ],
    };
  }

  /** What a node holds when it is made. A clamp is not nothing, though. */
  isIdentity(request: RenderRequest) {
    if (request.number("clampBlack", 1) !== 0) return false;
    if (request.number("clampWhite", 0) !== 0) return false;
    return GRADE_KNOBS.every(([name, , , value]) =>
      [0, 1, 2, 3].every((c) => request.number(name, value, c) === value),
    );
  }

  process(request: RenderRequest): boolean {
    const source = request.input("Source")?.buffer;
    const target = request.output()?.buffer;
    if (!isValidBuffer(source) || !isValidBuffer(target)) return false;
    const gpu = request.gpu;
    const u = UniformWriter.for(gradeModule).setAll({
      width: target.width,
      height: target.height,
      srcStride: source.stride,
      dstStride: target.stride,
      srcOffsetX: target.rect.x1 - source.rect.x1,
      srcOffsetY: target.rect.y1 - source.rect.y1,
      srcWidth: source.width,
      srcHeight: source.height,
      clampBlack: request.number("clampBlack", 1) !== 0,
      clampWhite: request.number("clampWhite", 0) !== 0,
      reverse: request.number("reverse", 0) !== 0,
    });
    for (const [name, , , value] of GRADE_KNOBS) {
      u.set(
        name,
        [0, 1, 2, 3].map((c) => request.number(name, value, c)),
      );
    }
    // The matte only if somebody asked. Unasked, the placeholder takes its
    // binding: aofx binds the picture there, which WebGPU refuses for a
    // buffer that is written.
    const clip = request.output("Clip")?.buffer;
    const haveClip = isValidBuffer(clip);
    u.set("wantClip", haveClip);
    return gpu.run(
      gpu.load(gradeModule, "gradeMain"),
      [target.width, target.height, 1],
      [source, target, haveClip ? clip : gpu.placeholder],
      u,
    );
  }
}
