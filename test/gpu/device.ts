// A WebGPU device for tests, from Dawn's Node bindings (the `webgpu`
// package). On a machine without a usable adapter `device` is null and the
// GPU suites skip.

import { create, globals } from "webgpu";

Object.assign(globalThis, globals);

// Dawn tears down its instance when the GPU object is garbage-collected, and
// a device still in use then crashes the process (SIGSEGV, at random). Keep
// both alive for the whole run.
let gpu: GPU | null = null;
let adapter: GPUAdapter | null = null;

async function open(): Promise<GPUDevice | null> {
  try {
    gpu = create([]);
    adapter = await gpu.requestAdapter();
    return adapter ? await adapter.requestDevice() : null;
  } catch {
    return null;
  }
}

export const device = await open();

export function storage(data: ArrayBufferView | number): GPUBuffer {
  const d = device as GPUDevice;
  const size = typeof data === "number" ? data : data.byteLength;
  const buffer = d.createBuffer({
    size: Math.max(16, Math.ceil(size / 16) * 16),
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST,
  });
  if (typeof data !== "number") {
    d.queue.writeBuffer(
      buffer,
      0,
      data.buffer,
      data.byteOffset,
      data.byteLength,
    );
  }
  return buffer;
}

export async function readBack(buffer: GPUBuffer): Promise<ArrayBuffer> {
  const d = device as GPUDevice;
  const staging = d.createBuffer({
    size: buffer.size,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  const encoder = d.createCommandEncoder();
  encoder.copyBufferToBuffer(buffer, 0, staging, 0, buffer.size);
  d.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const copy = staging.getMappedRange().slice(0);
  staging.unmap();
  staging.destroy();
  return copy;
}

/** Uniform block of `count` followed by padding, as the test kernels use. */
export function countParams(count: number): Uint32Array {
  return new Uint32Array([count, 0, 0, 0]);
}
