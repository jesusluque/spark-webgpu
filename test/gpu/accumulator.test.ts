// The accumulator formats on Dawn: packed (centers relative to the camera)
// draws what ext draws, "auto" switches to packed past the ext binding
// limit, and profiling degrades to off without timestamp-query.

import * as THREE from "three";
import { describe, expect, it, vi } from "vitest";
import { setPackedSplat } from "../../src/utils";
import { GpuProfiler } from "../../src/webgpu/GpuProfiler";
import {
  GpuSplatSource,
  WgpuSplatRenderer,
  type WgpuSplatRendererOptions,
} from "../../src/webgpu/WgpuSplatRenderer";
import { capabilitiesOf } from "../../src/webgpu/capabilities";
import { device } from "./device";

const W = 96;
const H = 64;

describe.skipIf(!device)("accumulator formats", () => {
  const d = device as GPUDevice;
  const canvas = d.createTexture({
    size: [W, H],
    format: "bgra8unorm",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  });
  const fakeRenderer = {
    backend: {
      isWebGPUBackend: true,
      device: d,
      context: {
        getCurrentTexture: () => canvas,
      } as unknown as GPUCanvasContext,
      get: () => undefined,
    },
  };

  async function readCanvas(): Promise<Uint8Array> {
    const bytesPerRow = 256 * Math.ceil((W * 4) / 256);
    const buf = d.createBuffer({
      size: bytesPerRow * H,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const enc = d.createCommandEncoder();
    enc.copyTextureToBuffer({ texture: canvas }, { buffer: buf, bytesPerRow }, [
      W,
      H,
    ]);
    d.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const src = new Uint8Array(buf.getMappedRange());
    const out = new Uint8Array(W * H * 4);
    for (let y = 0; y < H; y++) {
      out.set(
        src.subarray(y * bytesPerRow, y * bytesPerRow + W * 4),
        y * W * 4,
      );
    }
    buf.unmap();
    buf.destroy();
    return out;
  }

  function clearCanvas() {
    const enc = d.createCommandEncoder();
    enc
      .beginRenderPass({
        colorAttachments: [
          {
            view: canvas.createView(),
            loadOp: "clear",
            storeOp: "store",
            clearValue: [0, 0, 0, 1],
          },
        ],
      })
      .end();
    d.queue.submit([enc.finish()]);
  }

  // A sphere of `sphere` splats; the rest of `count` stay zero (inactive).
  function sphereSource(count: number, sphere = 4000) {
    const packed = new Uint32Array(count * 4);
    for (let i = 0; i < sphere; i++) {
      const u = 1 - (2 * (i + 0.5)) / sphere;
      const t = i * Math.PI * (3 - Math.sqrt(5));
      const r = Math.sqrt(1 - u * u);
      setPackedSplat(
        packed,
        i,
        r * Math.cos(t),
        u,
        r * Math.sin(t),
        0.06,
        0.06,
        0.06,
        0,
        0,
        0,
        1,
        0.9,
        0.2 + 0.6 * (i % 2),
        0.8,
        0.3,
      );
    }
    return GpuSplatSource.fromPacked(d, packed, count);
  }

  // Renders the sphere at `at` seen from 4 units in front of it.
  async function render(
    source: GpuSplatSource,
    options: WgpuSplatRendererOptions,
    at = new THREE.Vector3(),
  ) {
    const splats = new WgpuSplatRenderer(fakeRenderer as never, {
      depthTest: false,
      ...options,
    });
    const object = new THREE.Object3D();
    object.position.copy(at);
    splats.add(source, object);
    const camera = new THREE.PerspectiveCamera(60, W / H, 0.05, 100);
    camera.coordinateSystem = THREE.WebGPUCoordinateSystem;
    camera.updateProjectionMatrix();
    camera.position.copy(at).add(new THREE.Vector3(0.3, 0.2, 4));
    camera.lookAt(at);
    camera.updateMatrixWorld();
    clearCanvas();
    splats.render(camera);
    await d.queue.onSubmittedWorkDone();
    const px = await readCanvas();
    splats.dispose();
    return px;
  }

  function compare(a: Uint8Array, b: Uint8Array) {
    let lit = 0;
    let maxDiff = 0;
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
      const diff = Math.abs(a[i] - b[i]);
      maxDiff = Math.max(maxDiff, diff);
      sum += diff;
      if (i % 4 === 1 && a[i] > 40) lit++;
    }
    return { lit, maxDiff, mean: sum / a.length };
  }

  it("packed draws what ext draws, near and far from the origin", async () => {
    const source = sphereSource(4000);
    for (const at of [
      new THREE.Vector3(),
      new THREE.Vector3(5000, -300, 2000),
    ]) {
      const ext = await render(source, { accumulator: "ext" }, at);
      const packed = await render(source, { accumulator: "packed" }, at);
      const { lit, maxDiff, mean } = compare(ext, packed);
      expect(lit).toBeGreaterThan(W * H * 0.1);
      // 8-bit colour and half-float scales in the packed accumulator.
      expect(mean).toBeLessThan(1.5);
      expect(maxDiff).toBeLessThan(48);
    }
    source.destroy();
  });

  it("auto switches to packed past the ext accumulator's binding limit", async () => {
    const { maxSplats, maxSplatsPacked } = capabilitiesOf(d);
    expect(maxSplatsPacked).toBeGreaterThanOrEqual(maxSplats);
    if (maxSplatsPacked === maxSplats) return; // dispatch-limited device
    // Inactive splats past the sphere: ext can't hold them, packed can.
    const count = maxSplats + 65536;
    const source = sphereSource(count);
    await expect(render(source, { accumulator: "ext" })).rejects.toThrow(
      /over this device/,
    );
    const small = sphereSource(4000);
    const reference = await render(small, { accumulator: "packed" });
    const big = await render(source, {});
    const { lit, mean } = compare(reference, big);
    expect(lit).toBeGreaterThan(W * H * 0.1);
    expect(mean).toBeLessThan(0.5);
    source.destroy();
    small.destroy();
  }, 60000);

  it("profiling needs timestamp-query", async () => {
    const has = d.features.has("timestamp-query");
    expect(GpuProfiler.create(d) === null).toBe(!has);
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const source = sphereSource(1000);
    const splats = new WgpuSplatRenderer(fakeRenderer as never, {
      depthTest: false,
      profile: true,
    });
    splats.add(source);
    const camera = new THREE.PerspectiveCamera(60, W / H, 0.05, 100);
    camera.position.set(0, 0, 4);
    camera.updateMatrixWorld();
    splats.render(camera);
    await d.queue.onSubmittedWorkDone();
    expect(splats.options.profile).toBe(has);
    warn.mockRestore();
    splats.dispose();
    source.destroy();
  });
});
