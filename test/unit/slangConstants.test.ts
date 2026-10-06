import { describe, expect, it } from "vitest";
import { MERGE_OPERATIONS } from "../../src/webgpu/fx/effects/Merge";
import { TRANSFORM_FILTERS } from "../../src/webgpu/fx/effects/Transform";
import { fxMerge, fxTransform } from "../../src/webgpu/generated/constants";
// @ts-expect-error: a plain .mjs build script, untyped
import { findConstants } from "../../tools/slang-build/index.mjs";

describe("slang-build findConstants", () => {
  it("reads integer constants and expressions over earlier ones", () => {
    const source = `
      static const uint A = 4u; // a flag
      public static const uint B = 0xffffffffu;
      static const int C = -3;
      static const uint D = A * 2 + (1 << 3);
      /* static const uint E = 9u; */
      static const float F = 1.5;
      static const uint G = UNKNOWN + 1;
      static const uint kX = 0u;   static const uint kY = 1u;
    `;
    expect(findConstants(source)).toEqual({
      A: 4,
      B: 0xffffffff,
      C: -3,
      D: 16,
      kX: 0,
      kY: 1,
    });
  });
});

// The host's lists of a kernel's enumerations, against the kernel's values.
const squash = (s: string) => s.replace(/^k(?=[A-Z])|-/g, "").toLowerCase();

describe("fx enumerations match their kernels", () => {
  it("Merge operations", () => {
    expect(MERGE_OPERATIONS.map(squash)).toEqual(
      Object.entries(fxMerge)
        .sort((a, b) => a[1] - b[1])
        .map(([k]) => squash(k)),
    );
  });

  it("Transform filters", () => {
    const filters = Object.entries(fxTransform).filter(
      ([k]) => k !== "kBoxReach",
    );
    expect(TRANSFORM_FILTERS.map(squash)).toEqual(
      filters.sort((a, b) => a[1] - b[1]).map(([k]) => squash(k)),
    );
  });
});
