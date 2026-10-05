// draw/splat_draw.slang rendered offscreen and compared with the Gaussian
// footprint the vertex shader's maths predicts.

import { describe, expect, it } from "vitest";
import { encodeExtSplat } from "../../src/utils";
import generated from "../../src/webgpu/generated/draw/splat_draw";
import {
  createBindGroups,
  createReflectedRenderPipeline,
} from "../../src/webgpu/renderPipeline";
import { UniformWriter } from "../../src/webgpu/uniforms";
import { device, storage } from "./device";

const SIZE = 64;
const DRAW_EXT = 1;
const FOV_Y = Math.PI / 3;
const NEAR = 0.1;
const FAR = 100;
const MAX_STD_DEV = Math.sqrt(8);
const BLUR = 0.3;

// Perspective projection for WebGPU depth [0, 1], as four columns.
function perspective() {
  const f = 1 / Math.tan(FOV_Y / 2);
  return [
    [f, 0, 0, 0],
    [0, f, 0, 0],
    [0, 0, FAR / (NEAR - FAR), -1],
    [0, 0, (NEAR * FAR) / (NEAR - FAR), 0],
  ];
}

describe.skipIf(!device)("splat_draw.slang", () => {
  const d = device as GPUDevice;
  const rp = createReflectedRenderPipeline(d, generated, {
    vertex: "splatVertex",
    fragment: "splatFragment",
    targets: [{ format: "rgba32float" }],
  });

  async function render(
    splatList: { center: number[]; scale: number; rgba: number[] }[],
    ordering: number[],
  ): Promise<Float32Array> {
    const a = new Uint32Array(splatList.length * 4);
    const b = new Uint32Array(splatList.length * 4);
    splatList.forEach((s, i) => {
      encodeExtSplat(
        [a, b],
        i,
        s.center[0],
        s.center[1],
        s.center[2],
        s.scale,
        s.scale,
        s.scale,
        0,
        0,
        0,
        1,
        s.rgba[3],
        s.rgba[0],
        s.rgba[1],
        s.rgba[2],
      );
    });
    const words = new Uint32Array(splatList.length * 8);
    for (let i = 0; i < splatList.length; i++) {
      words.set(a.subarray(4 * i, 4 * i + 4), 8 * i);
      words.set(b.subarray(4 * i, 4 * i + 4), 8 * i + 4);
    }
    const p = perspective();
    const params = UniformWriter.for(generated).setAll({
      proj0: p[0],
      proj1: p[1],
      proj2: p[2],
      proj3: p[3],
      renderToViewQuat: [0, 0, 0, 1],
      renderToViewPos: [0, 0, 0, 0],
      renderToViewBasis0: [1, 0, 0, 0],
      renderToViewBasis1: [0, 1, 0, 0],
      renderToViewBasis2: [0, 0, 1, 0],
      renderWidth: SIZE,
      renderHeight: SIZE,
      maxStdDev: MAX_STD_DEV,
      minPixelRadius: 0,
      maxPixelRadius: 512,
      minAlpha: 0.5 / 255,
      blurAmount: BLUR,
      preBlurAmount: 0,
      focalDistance: 0,
      apertureAngle: 0,
      clipXY: 1.4,
      focalAdjustment: 1,
      falloff: 1,
      flags: DRAW_EXT,
    });
    const uniform = d.createBuffer({
      size: params.data.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    d.queue.writeBuffer(uniform, 0, params.data);
    const [group] = createBindGroups(d, rp, {
      ordering: storage(new Uint32Array(ordering)),
      splats: storage(words),
      params: uniform,
    });
    const target = d.createTexture({
      size: [SIZE, SIZE],
      format: "rgba32float",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    const bytesPerRow = SIZE * 16;
    const readback = d.createBuffer({
      size: bytesPerRow * SIZE,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const enc = d.createCommandEncoder();
    const pass = enc.beginRenderPass({
      colorAttachments: [
        {
          view: target.createView(),
          loadOp: "clear",
          storeOp: "store",
          clearValue: [0, 0, 0, 0],
        },
      ],
    });
    pass.setPipeline(rp.pipeline);
    pass.setBindGroup(0, group);
    pass.draw(4, ordering.length);
    pass.end();
    enc.copyTextureToBuffer(
      { texture: target },
      { buffer: readback, bytesPerRow },
      [SIZE, SIZE],
    );
    d.queue.submit([enc.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const pixels = new Float32Array(readback.getMappedRange().slice(0));
    readback.unmap();
    return pixels;
  }

  const pixel = (px: Float32Array, x: number, y: number) =>
    Array.from(px.subarray((y * SIZE + x) * 4, (y * SIZE + x) * 4 + 4));

  it("draws an isotropic splat with the predicted Gaussian footprint", async () => {
    const scale = 0.1;
    const z = 5;
    const px = await render(
      [{ center: [0, 0, -z], scale, rgba: [1, 0.5, 0.25, 1] }],
      [0],
    );

    // What the vertex shader computes for this splat.
    const focal = (0.5 * SIZE) / Math.tan(FOV_Y / 2);
    const variance = ((focal * scale) / z) ** 2;
    const blurred = variance + BLUR;
    // sqrt of the ratio of 2x2 determinants: (v / (v + blur))^2 under the root.
    const a0 = variance / blurred;
    const expectedAlpha = (x: number, y: number) => {
      // Pixel centres sit at +0.5; the splat centre is at SIZE / 2.
      const dx = x + 0.5 - SIZE / 2;
      const dy = y + 0.5 - SIZE / 2;
      const z2 = (dx * dx + dy * dy) / blurred;
      return z2 > MAX_STD_DEV ** 2 ? 0 : a0 * Math.exp(-0.5 * z2);
    };

    let worst = 0;
    for (let y = 24; y < 40; y++) {
      for (let x = 24; x < 40; x++) {
        const [r, g, , alpha] = pixel(px, x, y);
        const e = expectedAlpha(x, y);
        worst = Math.max(worst, Math.abs(alpha - e));
        if (alpha > 0) {
          expect(r).toBeCloseTo(1, 3);
          expect(g).toBeCloseTo(0.5, 2);
        }
      }
    }
    expect(worst).toBeLessThan(2e-3);
    expect(pixel(px, 0, 0)[3]).toBe(0);
  });

  it("draws nothing for splats behind the camera or empty ordering slots", async () => {
    const px = await render(
      [
        { center: [0, 0, 5], scale: 0.5, rgba: [1, 1, 1, 1] },
        { center: [0, 0, -5], scale: 0.5, rgba: [1, 1, 1, 1] },
      ],
      [0, 0xffffffff],
    );
    expect(px.every((v) => v === 0)).toBe(true);
  });
});
