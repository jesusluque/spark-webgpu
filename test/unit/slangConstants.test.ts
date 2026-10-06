import { describe, expect, it } from "vitest";
import { LN_SCALE_MAX, LN_SCALE_MIN } from "../../src/defines";
import { WGSL_PRELUDE, f32Literal } from "../../src/dyno/wgsl/prelude";
import { MERGE_OPERATIONS } from "../../src/webgpu/fx/effects/Merge";
import { TRANSFORM_FILTERS } from "../../src/webgpu/fx/effects/Transform";
import {
  coreMath,
  fxMerge,
  fxTransform,
} from "../../src/webgpu/generated/constants";
// @ts-expect-error: a plain .mjs build script, untyped
import {
  findConstants,
  workgroupStorageBytes,
} from "../../tools/slang-build/index.mjs";

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
      F: 1.5,
    });
  });

  it("reads float constants", () => {
    const source = `
      static const float A = -12.0;
      public static const float B = 1.5e3f;
      static const float C = A * 0.5 + .25;
      static const float INF = 1.0 / 0.0;
      static const float NEG = -1.0 / 0.0;
      static const float NAN = 0.0 / 0.0;
      static const float4 V = float4(1.0);
      static const float X = sqrt(2.0);
    `;
    expect(findConstants(source)).toEqual({
      A: -12,
      B: 1500,
      C: -5.75,
      INF: Number.POSITIVE_INFINITY,
      NEG: Number.NEGATIVE_INFINITY,
    });
  });
});

describe("slang-build workgroupStorageBytes", () => {
  it("rounds each workgroup variable up to 16 bytes, as WebGPU counts", () => {
    const wgsl = `
      var<workgroup> a : array<atomic<u32>, i32(16)>;
      var<workgroup> b : array<vec4<u32>, i32(128)>;
      var<workgroup> c : u32;
      var<workgroup> d : array<vec3<f32>, 3>;
      var<private> e : array<u32, 64>;
    `;
    expect(workgroupStorageBytes(wgsl)).toBe(64 + 2048 + 16 + 48);
  });
});

describe("Slang and JS constants agree", () => {
  it("scale range", () => {
    expect(coreMath.LN_SCALE_MIN).toBe(LN_SCALE_MIN);
    expect(coreMath.LN_SCALE_MAX).toBe(LN_SCALE_MAX);
    expect(coreMath.PI).toBe(Math.PI);
  });

  it("the dyno WGSL prelude takes them as f32 literals", () => {
    expect(f32Literal(-12)).toBe("-12.0");
    expect(f32Literal(0.5)).toBe("0.5");
    expect(f32Literal(1e-7)).toBe("1e-7");
    expect(WGSL_PRELUDE).toContain("const LN_SCALE_MIN: f32 = -12.0;");
    expect(WGSL_PRELUDE).toContain("const LN_SCALE_MAX: f32 = 9.0;");
    expect(WGSL_PRELUDE).toContain("const SPLAT_TEX_WIDTH_MASK: u32 = 2047u;");
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
