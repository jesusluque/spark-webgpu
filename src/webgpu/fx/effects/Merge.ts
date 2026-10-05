// aopenfx/examples/merge/Merge.cpp: A onto B with net.sf.openfx.MergePlugin's
// thirty-nine operations, at its indices. B is input 0 and the pass-through
// one. The default region, the union of the inputs, is right for a merge.

import mergeModule from "../../generated/fx/merge";
import { UniformWriter } from "../../uniforms";
import { Effect, type RenderRequest } from "../Effect";
import { COLOR_PLANE, type EffectDesc, isValidBuffer } from "../types";

/** The plugin's own list and order. Appended to, never reordered. */
export const MERGE_OPERATIONS = [
  "atop",
  "average",
  "color",
  "color-burn",
  "color-dodge",
  "conjoint-over",
  "copy",
  "difference",
  "disjoint-over",
  "divide",
  "exclusion",
  "freeze",
  "from",
  "geometric",
  "grain-extract",
  "grain-merge",
  "hard-light",
  "hue",
  "hypot",
  "in",
  "luminosity",
  "mask",
  "matte",
  "max",
  "min",
  "minus",
  "multiply",
  "out",
  "over",
  "overlay",
  "pinlight",
  "plus",
  "reflect",
  "saturation",
  "screen",
  "soft-light",
  "stencil",
  "under",
  "xor",
] as const;

export const MERGE_OVER = 28;

const HSL = new Set(["hue", "saturation", "color", "luminosity"]);

export class Merge extends Effect {
  protected describeEffect(): EffectDesc {
    return {
      identifier: "org.aopenfx.merge",
      label: "Merge",
      grouping: "Merge",
      description:
        "Composites A onto B. Thirty-nine operations, at the indices the Merge plugin uses.",
      inputs: [
        {
          name: "B",
          label: "B (background)",
          optional: true,
          passThrough: true,
        },
        { name: "A", label: "A (foreground)", optional: true },
      ],
      outputs: [COLOR_PLANE],
      params: [
        {
          name: "operation",
          label: "Operation",
          hint: "How A is combined with B. `over` is A on top of B, weighted by A's alpha.",
          type: "choice",
          // The four HSL modes are labelled as differing from the plugin
          // where A partly covers, as the C++ says in its menu.
          choices: MERGE_OPERATIONS.map((value) => ({
            value,
            label: HSL.has(value)
              ? `${value} (differs from the Merge plugin)`
              : value,
          })),
          defaults: [MERGE_OVER],
        },
        {
          name: "screenAlpha",
          label: "Alpha masking",
          hint: "Output alpha is a + b - a*b rather than the operation applied to alpha.",
          type: "boolean",
          defaults: [0],
        },
      ],
    };
  }

  process(request: RenderRequest): boolean {
    const a = request.input("A")?.buffer;
    const b = request.input("B")?.buffer;
    const target = request.output()?.buffer;
    if (!isValidBuffer(target)) return false;
    const haveA = isValidBuffer(a);
    const haveB = isValidBuffer(b);
    if (!haveA && !haveB) return false;
    // A missing input is bound to the other and given a window of nothing,
    // so every sample of it falls outside and reads transparent.
    const present = haveA ? a : (b as NonNullable<typeof b>);
    const aBuf = haveA ? a : present;
    const bBuf = haveB ? b : present;
    const op = Math.trunc(request.number("operation", MERGE_OVER));
    const u = UniformWriter.for(mergeModule).setAll({
      width: target.width,
      height: target.height,
      dstStride: target.stride,
      aStride: aBuf.stride,
      bStride: bBuf.stride,
      aOffsetX: target.rect.x1 - aBuf.rect.x1,
      aOffsetY: target.rect.y1 - aBuf.rect.y1,
      bOffsetX: target.rect.x1 - bBuf.rect.x1,
      bOffsetY: target.rect.y1 - bBuf.rect.y1,
      aWidth: haveA ? aBuf.width : 0,
      aHeight: haveA ? aBuf.height : 0,
      bWidth: haveB ? bBuf.width : 0,
      bHeight: haveB ? bBuf.height : 0,
      operation: Math.min(Math.max(op, 0), MERGE_OPERATIONS.length - 1),
      screenAlpha: request.number("screenAlpha") !== 0,
    });
    return request.gpu.run(
      request.gpu.load(mergeModule, "mergeMain"),
      [target.width, target.height, 1],
      [aBuf, bBuf, target],
      u,
    );
  }
}
