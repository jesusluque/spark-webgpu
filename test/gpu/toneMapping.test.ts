// The inverse tone mappings SrgbComposite writes back through: three's
// output pass, tone-mapping the value written, must give the display colour
// composited. And the forward ports against three's formulas at a few
// points.

import * as THREE from "three";
import { describe, expect, it } from "vitest";
import {
  INVERTIBLE_TONE_MAPPINGS,
  toneMappingWgsl,
} from "../../src/webgpu/toneMapping";
import { device, readBack, storage } from "./device";

const NAMES: Record<number, string> = {
  [THREE.NoToneMapping]: "none",
  [THREE.LinearToneMapping]: "linear",
  [THREE.ReinhardToneMapping]: "reinhard",
  [THREE.CineonToneMapping]: "cineon",
  [THREE.ACESFilmicToneMapping]: "aces",
  [THREE.AgXToneMapping]: "agx",
  [THREE.NeutralToneMapping]: "neutral",
};

describe.skipIf(!device)("tone mapping inverses", () => {
  const d = device as GPUDevice;

  // out[i] = f(in[i]) for each colour, f "toneMap" or "roundTrip".
  async function run(mode: number, f: string, input: number[][], exposure = 1) {
    const module = d.createShaderModule({
      code: `${toneMappingWgsl(mode)}
@group(0) @binding(0) var<storage, read> src: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec4f>;
fn roundTrip(y: vec3f, e: f32) -> vec3f { return toneMap(inverseToneMap(y, e), e); }
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= arrayLength(&src)) { return; }
  dst[id.x] = vec4f(${f}(src[id.x].rgb, ${exposure.toFixed(4)}), 0.0);
}`,
    });
    const pipeline = d.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint: "main" },
    });
    const src = storage(new Float32Array(input.flatMap((c) => [...c, 0])));
    const dst = storage(input.length * 16);
    const encoder = d.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(
      0,
      d.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: src } },
          { binding: 1, resource: { buffer: dst } },
        ],
      }),
    );
    pass.dispatchWorkgroups(Math.ceil(input.length / 64));
    pass.end();
    d.queue.submit([encoder.finish()]);
    const out = new Float32Array(await readBack(dst));
    return input.map((_, i) => Array.from(out.subarray(4 * i, 4 * i + 3)));
  }

  // Scene colours: greys over the operators' range, and colours from
  // saturated to pale, dark to bright.
  const scene: number[][] = [];
  for (const v of [0, 0.002, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 20]) {
    scene.push([v, v, v]);
  }
  for (const r of [0.02, 0.3, 1.5]) {
    for (const g of [0.01, 0.4, 3]) {
      for (const b of [0.05, 0.7]) scene.push([r, g, b]);
    }
  }
  // Display colours, including ones some operators never produce.
  const display: number[][] = [];
  for (let v = 0; v <= 1.0001; v += 0.05) display.push([v, v, v]);
  for (const r of [0.1, 0.4, 0.7, 0.9]) {
    for (const g of [0.15, 0.45, 0.75]) {
      for (const b of [0.2, 0.5, 0.8]) display.push([r, g, b]);
    }
  }
  // Operators that produce every display colour in [0, 1]^3 (AgX and ACES
  // desaturate bright colours: their range is smaller).
  const onto = [
    THREE.NoToneMapping,
    THREE.LinearToneMapping,
    THREE.ReinhardToneMapping,
    THREE.CineonToneMapping,
    THREE.NeutralToneMapping,
  ];

  const worst = (a: number[][], b: number[][]) =>
    Math.max(...a.flatMap((c, i) => c.map((v, k) => Math.abs(v - b[i][k]))));

  for (const mode of INVERTIBLE_TONE_MAPPINGS) {
    it(`${NAMES[mode]}: tone-mapping the inverse gives the colour back`, async () => {
      for (const exposure of [1, 1.7]) {
        // Every colour the operator produces comes back...
        const produced = await run(mode, "toneMap", scene, exposure);
        const back = await run(mode, "roundTrip", produced, exposure);
        // ...within a level of the 8-bit canvas.
        expect(worst(back, produced)).toBeLessThan(0.5 / 255);
        if (onto.includes(mode)) {
          const all = await run(mode, "roundTrip", display, exposure);
          expect(worst(all, display)).toBeLessThan(0.5 / 255);
        }
      }
    });
  }

  // three r180's operators (nodes/display/ToneMappingFunctions), on the CPU.
  const rows = (...m: number[]) => new THREE.Matrix3(...(m as [])).elements;
  const cols = (...m: number[]) => new THREE.Matrix3().fromArray(m).elements;
  const apply = (m: number[], c: number[]) =>
    [0, 1, 2].map((r) => m[r] * c[0] + m[r + 3] * c[1] + m[r + 6] * c[2]);
  const clamp01 = (c: number[]) => c.map((v) => Math.min(Math.max(v, 0), 1));
  const reference: Record<number, (c: number[]) => number[]> = {
    [THREE.ACESFilmicToneMapping]: (c) => {
      const input = rows(
        0.59719,
        0.35458,
        0.04823,
        0.076,
        0.90834,
        0.01566,
        0.0284,
        0.13383,
        0.83777,
      );
      const output = rows(
        1.60475,
        -0.53108,
        -0.07367,
        -0.10208,
        1.10813,
        -0.00605,
        -0.00327,
        -0.07276,
        1.07602,
      );
      const v = apply(
        input,
        c.map((x) => x / 0.6),
      ).map(
        (u) =>
          (u * (u + 0.0245786) - 0.000090537) /
          (u * (u + 0.432951) * 0.983729 + 0.238081),
      );
      return clamp01(apply(output, v));
    },
    [THREE.AgXToneMapping]: (c) => {
      const toRec2020 = cols(
        0.6274,
        0.0691,
        0.0164,
        0.3293,
        0.9195,
        0.088,
        0.0433,
        0.0113,
        0.8956,
      );
      const toSrgb = cols(
        1.6605,
        -0.1246,
        -0.0182,
        -0.5876,
        1.1329,
        -0.1006,
        -0.0728,
        -0.0083,
        1.1187,
      );
      const inset = cols(
        0.856627153315983,
        0.137318972929847,
        0.11189821299995,
        0.0951212405381588,
        0.761241990602591,
        0.0767994186031903,
        0.0482516061458583,
        0.101439036467562,
        0.811302368396859,
      );
      const outset = cols(
        1.1271005818144368,
        -0.1413297634984383,
        -0.14132976349843826,
        -0.11060664309660323,
        1.157823702216272,
        -0.11060664309660294,
        -0.016493938717834573,
        -0.016493938717834257,
        1.2519364065950405,
      );
      const t = apply(inset, apply(toRec2020, c)).map((v) => {
        const x = Math.min(
          Math.max((Math.log2(Math.max(v, 1e-10)) + 12.47393) / 16.499999, 0),
          1,
        );
        return (
          15.5 * x ** 6 -
          40.14 * x ** 5 +
          31.96 * x ** 4 -
          6.868 * x ** 3 +
          0.4298 * x ** 2 +
          0.1191 * x -
          0.00232
        );
      });
      const p = apply(outset, t).map((v) => Math.max(v, 0) ** 2.2);
      return clamp01(apply(toSrgb, p));
    },
    [THREE.NeutralToneMapping]: (c) => {
      const x = Math.min(...c);
      const offset = x < 0.08 ? x - 6.25 * x * x : 0.04;
      let color = c.map((v) => v - offset);
      const peak = Math.max(...color);
      if (peak < 0.76) return color;
      const d = 0.24;
      const newPeak = 1 - (d * d) / (peak + d - 0.76);
      color = color.map((v) => (v * newPeak) / peak);
      const g = 1 - 1 / (0.15 * (peak - newPeak) + 1);
      return color.map((v) => v + (newPeak - v) * g);
    },
    [THREE.CineonToneMapping]: (c) =>
      c.map((v) => {
        const e = Math.max(v - 0.004, 0);
        return ((e * (6.2 * e + 0.5)) / (e * (6.2 * e + 1.7) + 0.06)) ** 2.2;
      }),
    [THREE.ReinhardToneMapping]: (c) => clamp01(c.map((v) => v / (1 + v))),
    [THREE.LinearToneMapping]: (c) => clamp01(c),
  };

  it("tone-maps as three's operators", async () => {
    for (const [mode, f] of Object.entries(reference)) {
      const out = await run(Number(mode), "toneMap", scene);
      expect(worst(out, scene.map(f))).toBeLessThan(1e-4);
    }
  });
});
