// GpuSorter against the CPU sort (sortBackToFront) on the same metrics.

import { describe, expect, it } from "vitest";
import { GpuSorter } from "../../src/webgpu/GpuSorter";
import { KernelRegistry } from "../../src/webgpu/KernelRegistry";
import { sortBackToFront } from "../../src/webgpu/WgpuSplatRenderer";
import { device, readBack, storage } from "./device";

function metrics(n: number, seed: number) {
  const m = new Float32Array(n);
  let s = seed;
  for (let i = 0; i < n; i++) {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    const r = s / 2 ** 32;
    // Some inactive splats, some exact ties, a wide range of magnitudes.
    m[i] =
      i % 53 === 0
        ? Number.POSITIVE_INFINITY
        : i % 7 === 0
          ? 2.5
          : 0.001 + r * r * 5000;
  }
  return m;
}

describe.skipIf(!device)("GpuSorter", () => {
  const registry = new KernelRegistry(device as GPUDevice);
  const sorter = new GpuSorter(registry);

  it.each([1, 1000, 100003, 1_200_000])(
    "matches the CPU order for %i splats",
    async (n) => {
      const m = metrics(n, n);
      const d = device as GPUDevice;
      const enc = d.createCommandEncoder();
      const pass = enc.beginComputePass();
      sorter.encode(pass, storage(m), n);
      pass.end();
      registry.submit(enc.finish());

      const expected = sortBackToFront(m, n);
      const args = new Uint32Array(await readBack(sorter.drawArgs));
      expect(args[0]).toBe(4);
      expect(args[1]).toBe(expected.length);
      const got = new Uint32Array(await readBack(sorter.ordering)).subarray(
        0,
        expected.length,
      );
      let firstBad = -1;
      for (let i = 0; i < expected.length; i++) {
        if (got[i] !== expected[i]) {
          firstBad = i;
          break;
        }
      }
      expect(firstBad).toBe(-1);
    },
  );
});
