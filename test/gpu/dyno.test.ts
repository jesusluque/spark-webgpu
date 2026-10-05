// Every dyno op compiled to WGSL (src/dyno/wgsl) and run on the GPU, checked
// against the GLSL semantics computed in JS (dynoCases.ts).

import { describe, expect, it } from "vitest";
import { device } from "./device";
import {
  type Expected,
  cases,
  graph,
  matrixCases,
  splatCases,
  textureCases,
} from "./dynoCases";
import { evalDyno } from "./dynoHarness";

function check(actual: number[], expected: Expected) {
  if (typeof expected === "function") {
    expected(actual);
    return;
  }
  expect(actual.length).toBe(expected.length);
  actual.forEach((a, i) => {
    const e = expected[i];
    expect(Math.abs(a - e), `component ${i}: ${a} vs ${e}`).toBeLessThan(
      1e-4 * Math.max(1, Math.abs(e)),
    );
  });
}

describe.skipIf(!device)("dyno ops in WGSL", () => {
  for (const [name, type, f, expected] of [
    ...cases,
    ...splatCases,
    ...matrixCases,
    ...textureCases,
  ]) {
    it(name, async () => {
      check(await evalDyno(graph(type, f)), expected);
    });
  }
});
