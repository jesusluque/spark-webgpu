import { describe, expect, it } from "vitest";
import { sortBackToFront } from "../../src/webgpu/WgpuSplatRenderer";

describe("sortBackToFront", () => {
  it("orders finite metrics largest first and drops infinities", () => {
    const n = 100000;
    const metric = new Float32Array(n);
    let s = 1;
    for (let i = 0; i < n; i++) {
      s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
      metric[i] =
        i % 97 === 0 ? Number.POSITIVE_INFINITY : (s / 2 ** 32) * 1000;
    }
    const order = sortBackToFront(metric, n);
    const finite = metric.filter(Number.isFinite).length;
    expect(order.length).toBe(finite);
    for (let k = 1; k < order.length; k++) {
      expect(metric[order[k - 1]]).toBeGreaterThanOrEqual(metric[order[k]]);
    }
    expect(new Set(order).size).toBe(order.length);
  });
});
