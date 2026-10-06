import { describe, expect, it } from "vitest";
import {
  SORT_WORKGROUP_BYTES,
  capabilitiesOf,
  splatRequiredLimits,
} from "../../src/webgpu/capabilities";

// The WebGPU spec's default limits, which a device gets unless it asks.
const DEFAULTS = {
  maxStorageBuffersPerShaderStage: 8,
  maxComputeWorkgroupStorageSize: 16384,
  maxComputeInvocationsPerWorkgroup: 256,
  maxComputeWorkgroupSizeX: 256,
  maxStorageBufferBindingSize: 134217728,
  maxBufferSize: 268435456,
};

const gpu = (limits: Record<string, number>, features: string[] = []) => ({
  limits: limits as unknown as GPUSupportedLimits,
  features: new Set(features) as unknown as GPUSupportedFeatures,
});

describe("capabilitiesOf", () => {
  it("takes the GPU sort on default limits and sizes by the storage binding", () => {
    const c = capabilitiesOf(gpu(DEFAULTS));
    // radixScatter: sScan (128 x vec4<u32>) + sBase (16 x u32).
    expect(SORT_WORKGROUP_BYTES).toBe(2112);
    expect(c.gpuSort).toBe(true);
    expect(c.maxSplats).toBe(4194304);
    expect(c.maxSplatsPacked).toBe(8388608);
    expect(c.vertexStorage).toBe(true);
    expect(c.subgroups).toBe(false);
  });

  it("falls back to the CPU sort when workgroups are too small", () => {
    const c = capabilitiesOf(
      gpu({ ...DEFAULTS, maxComputeInvocationsPerWorkgroup: 128 }),
    );
    expect(c.gpuSort).toBe(false);
    expect(c.gpuSortReason).toMatch(/maxComputeInvocationsPerWorkgroup/);
  });

  it("falls back to the CPU sort when workgroup storage is too small", () => {
    const ok = capabilitiesOf(
      gpu({ ...DEFAULTS, maxComputeWorkgroupStorageSize: 2112 }),
    );
    expect(ok.gpuSort).toBe(true);
    const c = capabilitiesOf(
      gpu({ ...DEFAULTS, maxComputeWorkgroupStorageSize: 2048 }),
    );
    expect(c.gpuSort).toBe(false);
    expect(c.gpuSortReason).toMatch(/maxComputeWorkgroupStorageSize/);
  });

  it("reads compatibility mode's vertex-stage storage limit", () => {
    const c = capabilitiesOf(
      gpu({ ...DEFAULTS, maxStorageBuffersInVertexStage: 0 }),
    );
    expect(c.vertexStorage).toBe(false);
  });

  it("reports optional features without assuming them", () => {
    const c = capabilitiesOf(
      gpu(DEFAULTS, ["subgroups", "timestamp-query", "bgra8unorm-storage"]),
    );
    expect(c.subgroups && c.timestampQuery && c.bgra8unormStorage).toBe(true);
    expect(c.float32Filterable).toBe(false);
  });

  it("asks for the adapter's storage sizes", () => {
    const limits = splatRequiredLimits(
      gpu({ ...DEFAULTS, maxStorageBufferBindingSize: 4294967292 }),
    );
    expect(limits.maxStorageBufferBindingSize).toBe(4294967292);
    expect(limits.maxBufferSize).toBe(268435456);
  });
});
