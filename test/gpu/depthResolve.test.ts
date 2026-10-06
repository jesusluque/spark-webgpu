// DepthResolve: a multisampled depth texture copied to a single-sample one.

import { describe, expect, it } from "vitest";
import { DepthResolve } from "../../src/webgpu/DepthResolve";
import { device } from "./device";

describe.skipIf(!device)("DepthResolve", () => {
  it("copies sample 0 of a multisampled depth texture", async () => {
    const d = device as GPUDevice;
    const source = d.createTexture({
      size: [8, 4],
      format: "depth24plus-stencil8",
      sampleCount: 4,
      usage:
        GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    const encoder = d.createCommandEncoder();
    encoder
      .beginRenderPass({
        colorAttachments: [],
        depthStencilAttachment: {
          view: source.createView(),
          depthLoadOp: "clear",
          depthClearValue: 0.25,
          depthStoreOp: "store",
          stencilLoadOp: "clear",
          stencilStoreOp: "store",
        },
      })
      .end();
    const resolve = new DepthResolve(d);
    const resolved = resolve.resolve(encoder, source);
    expect(resolved.sampleCount).toBe(1);
    const buffer = d.createBuffer({
      size: 256 * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    encoder.copyTextureToBuffer(
      { texture: resolved, aspect: "depth-only" },
      { buffer, bytesPerRow: 256 },
      [8, 4],
    );
    d.queue.submit([encoder.finish()]);
    await buffer.mapAsync(GPUMapMode.READ);
    const rows = new Float32Array(buffer.getMappedRange());
    for (let y = 0; y < 4; y++) {
      for (let x = 0; x < 8; x++) {
        expect(rows[y * 64 + x]).toBeCloseTo(0.25, 5);
      }
    }
    buffer.unmap();
    resolve.dispose();
    source.destroy();
  });
});
