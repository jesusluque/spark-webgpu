// SrgbComposite: an empty splat layer (L = 0, T = 1) leaves a linear target
// as it was, through three's sRGB transfer functions and back.

import * as THREE from "three";
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

  // Under three's tone mapping T, a splat layer (L, t) composites over the
  // picture as shown, srgb(T(dst)), and the value written back tone-maps to
  // linear(L + t * srgb(T(dst))). Uncovered pixels keep their HDR value.
  it("composites in display space under tone mapping", async () => {
    const reinhard = (c: number) => (2 * c) / (2 * c + 1); // exposure 2
    const color = d.createTexture({
      size: [2, 1],
      format: "rgba16float",
      usage:
        GPUTextureUsage.RENDER_ATTACHMENT |
        GPUTextureUsage.COPY_SRC |
        GPUTextureUsage.COPY_DST,
    });
    const readPixels = async () => {
      const read = d.createBuffer({
        size: 256,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      const enc = d.createCommandEncoder();
      enc.copyTextureToBuffer({ texture: color }, { buffer: read }, [2, 1]);
      d.queue.submit([enc.finish()]);
      await read.mapAsync(GPUMapMode.READ);
      const px = Array.from(
        new Uint16Array(read.getMappedRange().slice(0, 16)),
      );
      read.unmap();
      read.destroy();
      return px;
    };
    const dst = 3.0;
    const half = (v: number) => {
      const f = new Float32Array([v]);
      const x = new Uint32Array(f.buffer)[0];
      return (
        ((x >> 16) & 0x8000) |
        ((((x >> 23) & 0xff) - 112) << 10) |
        ((x >> 13) & 0x3ff)
      );
    };
    d.queue.writeTexture(
      { texture: color },
      new Uint16Array([dst, dst, dst, 1, dst, dst, dst, 1].map(half)),
      { bytesPerRow: 16 },
      [2, 1],
    );
    const composite = new SrgbComposite(d);
    const toneMapping = {
      toneMapping: THREE.ReinhardToneMapping,
      exposure: 2,
    };
    // Pixels of an empty layer keep their HDR value...
    const e1 = d.createCommandEncoder();
    composite.beginLayer(e1, color, undefined).end();
    composite.composite(
      e1,
      color,
      { view: color.createView() },
      null,
      toneMapping,
    );
    // ...and under a splat of sRGB colour 0.6 at alpha 0.5 (premultiplied),
    // filling the layer, they show the blend.
    const layer = (composite as unknown as { scratch: { layer: GPUTexture } })
      .scratch.layer;
    e1.beginRenderPass({
      colorAttachments: [
        {
          view: layer.createView(),
          loadOp: "clear",
          storeOp: "store",
          clearValue: [77 / 255, 77 / 255, 77 / 255, 128 / 255],
        },
      ],
    }).end();
    d.queue.submit([e1.finish()]);
    const before = await readPixels();
    const e2 = d.createCommandEncoder();
    composite.composite(
      e2,
      color,
      { view: color.createView() },
      null,
      toneMapping,
    );
    d.queue.submit([e2.finish()]);
    const px = await readPixels();
    const fromHalf = (h: number) =>
      (h & 0x8000 ? -1 : 1) *
      2 ** (((h >> 10) & 0x1f) - 15) *
      (1 + (h & 0x3ff) / 1024);
    const t = 128 / 255; // the layer's alpha is the transmittance
    const shown = linear(77 / 255 + t * srgb(reinhard(dst)));
    expect(fromHalf(before[0])).toBe(dst);
    expect(reinhard(fromHalf(px[0]))).toBeCloseTo(shown, 3);
    composite.dispose();
  });

  // three's viewport may reach past the target (an offset or tiled view):
  // the scissor stays inside it, as a scissor past the attachment is invalid.
  it("composites within a viewport partly outside the target", async () => {
    const color = d.createTexture({
      size: [64, 64],
      format: "rgba16float",
      usage:
        GPUTextureUsage.RENDER_ATTACHMENT |
        GPUTextureUsage.COPY_SRC |
        GPUTextureUsage.COPY_DST,
    });
    const composite = new SrgbComposite(d);
    d.pushErrorScope("validation");
    const encoder = d.createCommandEncoder();
    composite.beginLayer(encoder, color, undefined).end();
    composite.composite(
      encoder,
      color,
      { view: color.createView() },
      { x: -16, y: 32, z: 64, w: 64 },
    );
    d.queue.submit([encoder.finish()]);
    expect(await d.popErrorScope()).toBeNull();
    composite.dispose();
    color.destroy();
  });
});
