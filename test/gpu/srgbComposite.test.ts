// SrgbComposite: an empty splat layer (L = 0, T = 1) leaves a linear target
// as it was, through three's sRGB transfer functions and back.

import { describe, expect, it } from "vitest";
import { SrgbComposite } from "../../src/webgpu/SrgbComposite";
import { device } from "./device";

const srgb = (c: number) =>
  c <= 0.0031308 ? c * 12.92 : 1.055 * c ** 0.41666 - 0.055;
const linear = (c: number) =>
  c <= 0.04045 ? c * 0.0773993808 : (c * 0.9478672986 + 0.0521327014) ** 2.4;

describe.skipIf(!device)("SrgbComposite", () => {
  const d = device as GPUDevice;

  it("leaves pixels no splat covers unchanged", async () => {
    const color = d.createTexture({
      size: [2, 1],
      format: "rgba32float",
      usage:
        GPUTextureUsage.RENDER_ATTACHMENT |
        GPUTextureUsage.COPY_SRC |
        GPUTextureUsage.COPY_DST,
    });
    const dst = 0.2;
    d.queue.writeTexture(
      { texture: color },
      new Float32Array([dst, dst, dst, 1, dst, dst, dst, 1]),
      { bytesPerRow: 32 },
      [2, 1],
    );
    const composite = new SrgbComposite(d);
    const encoder = d.createCommandEncoder();
    composite.beginLayer(encoder, color, undefined).end();
    composite.composite(encoder, color, { view: color.createView() });
    d.queue.submit([encoder.finish()]);
    const read = d.createBuffer({
      size: 256,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const enc2 = d.createCommandEncoder();
    enc2.copyTextureToBuffer({ texture: color }, { buffer: read }, [2, 1]);
    d.queue.submit([enc2.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const px = new Float32Array(read.getMappedRange().slice(0, 32));
    read.unmap();
    expect(px[0]).toBeCloseTo(linear(srgb(dst)), 4);
    expect(px[0]).toBeCloseTo(dst, 3);
    expect(px[3]).toBeCloseTo(1, 5);
    composite.dispose();
  });
});
