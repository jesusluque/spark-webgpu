// capabilitiesOf on a real device: GPUSupportedLimits keeps its values on
// the prototype, which a plain Object.keys would miss.

import { describe, expect, it } from "vitest";
import { capabilitiesOf } from "../../src/webgpu/capabilities";
import { device } from "./device";

describe.skipIf(!device)("capabilitiesOf(device)", () => {
  it("reads the device's limits", () => {
    const d = device as GPUDevice;
    const c = capabilitiesOf(d);
    expect(c.limits.maxStorageBufferBindingSize).toBe(
      d.limits.maxStorageBufferBindingSize,
    );
    expect(c.maxSplats).toBe(
      Math.floor(
        Math.min(d.limits.maxStorageBufferBindingSize, d.limits.maxBufferSize) /
          32,
      ),
    );
    expect(c.gpuSort).toBe(true);
  });
});
