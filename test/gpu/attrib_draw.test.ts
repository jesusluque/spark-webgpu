// Attributes through WgpuSplatRenderer on Dawn: gathered after generate into
// the draw pool (LOD remap, missing attributes, format conversion, direction
// rotation), then read in the draw: colour from an attribute, the composited
// slot target and the picking id target.

import * as THREE from "three";
import { describe, expect, it } from "vitest";
import { encodeExtSplat, fromHalf } from "../../src/utils";
import {
  GpuSplatSource,
  WgpuSplatRenderer,
} from "../../src/webgpu/WgpuSplatRenderer";
import { SplatAttributes } from "../../src/webgpu/attributes/SplatAttributes";
import {
  attribWords,
  columnBits,
  decodeComponent,
} from "../../src/webgpu/attributes/schema";
import { device, readBack } from "./device";

const W = 96;
const H = 64;

describe.skipIf(!device)("splat attributes in the renderer", () => {
  const d = device as GPUDevice;
  const canvas = d.createTexture({
    size: [W, H],
    format: "rgba8unorm",
    usage:
      GPUTextureUsage.RENDER_ATTACHMENT |
      GPUTextureUsage.COPY_SRC |
      GPUTextureUsage.COPY_DST,
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

  async function readTexture(tex: GPUTexture, bytesPerPixel: number) {
    const bytesPerRow = 256 * Math.ceil((W * bytesPerPixel) / 256);
    const buf = d.createBuffer({
      size: bytesPerRow * H,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const enc = d.createCommandEncoder();
    enc.copyTextureToBuffer({ texture: tex }, { buffer: buf, bytesPerRow }, [
      W,
      H,
    ]);
    d.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const src = new Uint8Array(buf.getMappedRange());
    const out = new Uint8Array(W * H * bytesPerPixel);
    for (let y = 0; y < H; y++) {
      out.set(
        src.subarray(y * bytesPerRow, y * bytesPerRow + W * bytesPerPixel),
        y * W * bytesPerPixel,
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

  // `count` splats on a row along x, opaque and round.
  function rowSource(count: number, radius: number, spacing: number) {
    const a = new Uint32Array(count * 4);
    const b = new Uint32Array(count * 4);
    for (let i = 0; i < count; i++) {
      const x = (i - (count - 1) / 2) * spacing;
      encodeExtSplat(
        [a, b],
        i,
        x,
        0,
        0,
        radius,
        radius,
        radius,
        0,
        0,
        0,
        1,
        1,
        0.5,
        0.5,
        0.5,
      );
    }
    return GpuSplatSource.fromExt(d, a, b, count);
  }

  function camera() {
    const cam = new THREE.PerspectiveCamera(60, W / H, 0.05, 100);
    cam.coordinateSystem = THREE.WebGPUCoordinateSystem;
    cam.updateProjectionMatrix();
    cam.position.set(0, 0, 4);
    cam.lookAt(0, 0, 0);
    cam.updateMatrixWorld();
    return cam;
  }

  it("gathers toDraw attributes through LOD, conversion and rotation", async () => {
    const splats = new WgpuSplatRenderer(fakeRenderer as never, {
      depthTest: false,
    });
    const attrs = new SplatAttributes(splats);

    const n0 = 50;
    const s0 = rowSource(n0, 0.01, 0.01);
    const labels = Uint8Array.from({ length: n0 }, (_, i) => (i * 7) % 256);
    s0.setAttribute("label", labels, "u8", 1, { toDraw: true });
    const normals = Float32Array.from({ length: n0 * 3 }, (_, k) =>
      k % 3 === 0 ? 1 : 0,
    );
    s0.setAttribute("normal", normals, "snorm8", 3, {
      toDraw: true,
      direction: true,
    });
    s0.setAttribute("hidden", new Float32Array(n0), "f32", 1); // not toDraw
    const feature = Float32Array.from({ length: n0 * 5 }, (_, k) => k * 0.25);
    s0.setAttribute("feature", feature, "f16", 5, { toDraw: true });
    const m0 = splats.add(s0);
    m0.object.rotation.set(0, 0, Math.PI / 2); // turns +x into +y
    const lod = Uint32Array.from({ length: 20 }, (_, i) => n0 - 1 - 2 * i);
    splats.setLodIndices(m0, lod);

    // A second mesh with only a feature, in another format: converted.
    const n1 = 30;
    const s1 = rowSource(n1, 0.01, 0.01);
    const f1 = Float32Array.from({ length: n1 * 5 }, (_, k) => -k);
    s1.setAttribute("feature", f1, "f32", 5, { toDraw: true });
    splats.add(s1);

    splats.render(camera());
    await d.queue.onSubmittedWorkDone();

    const words = new Uint32Array(await readBack(attrs.drawPool as GPUBuffer));
    const { layout } = attrs;
    expect(layout.specs.map((s) => s.name)).toEqual([
      "label",
      "normal",
      "feature",
    ]);
    expect(words[1]).toBe(3);
    const read = (slot: number, k: number, c: number) => {
      const spec = layout.specs[k];
      const at =
        layout.headerWords + slot * layout.strideWords + layout.offsets[k];
      const col = {
        spec,
        words: words.subarray(at, at + attribWords(spec)),
      };
      return decodeComponent(spec.format, columnBits(col, 0, c));
    };
    lod.forEach((src, i) => {
      expect(read(i, 0, 0)).toBe(labels[src]);
      // Rotated by the object: (1, 0, 0) -> (0, 1, 0).
      expect(read(i, 1, 0)).toBeCloseTo(0, 2);
      expect(read(i, 1, 1)).toBeCloseTo(1, 2);
      for (let c = 0; c < 5; c++) {
        expect(read(i, 2, c)).toBeCloseTo(feature[src * 5 + c], 2);
      }
    });
    for (let i = 0; i < n1; i++) {
      const slot = lod.length + i;
      expect(read(slot, 0, 0)).toBe(0); // absent: zero
      for (let c = 0; c < 5; c++) {
        expect(read(slot, 2, c)).toBeCloseTo(f1[i * 5 + c], 3); // f32 -> f16
      }
    }
    attrs.dispose();
    splats.dispose();
  });

  it("draws from attributes: colour, composited slot and pick ids", async () => {
    const splats = new WgpuSplatRenderer(fakeRenderer as never, {
      depthTest: false,
    });
    const attrs = new SplatAttributes(splats, {
      slots: ["value"],
      colorMode: "value",
      targets: { id: true, slot0: true },
    });
    const count = 3;
    const source = rowSource(count, 0.15, 1.2);
    const values = [1, 0, 0, 0, 1, 0, 0.2, 0.4, 1];
    source.setAttribute("value", values, "unorm8", 3, { toDraw: true });
    splats.add(source);

    clearCanvas();
    splats.render(camera());
    await d.queue.onSubmittedWorkDone();

    const color = await readTexture(canvas, 4);
    const slot = await readTexture(attrs.slotTextures[0] as GPUTexture, 8);
    const half = new Uint16Array(slot.buffer);
    // Pixel centres of the three splats on the row y = H / 2.
    const cam = camera();
    for (let i = 0; i < count; i++) {
      const p = new THREE.Vector3((i - 1) * 1.2, 0, 0).project(cam);
      const x = Math.floor(((p.x + 1) / 2) * W);
      const y = Math.floor(((1 - p.y) / 2) * H);
      const px = y * W + x;
      const v = values.slice(3 * i, 3 * i + 3);
      const alpha = fromHalf(half[px * 4 + 3]);
      expect(alpha).toBeGreaterThan(0.5);
      for (let c = 0; c < 3; c++) {
        // Composited and premultiplied: value * alpha over a clear target.
        expect(fromHalf(half[px * 4 + c]) / alpha).toBeCloseTo(v[c], 1);
        // Colour mode "value", premultiplied over black.
        expect(color[px * 4 + c] / 255).toBeCloseTo(v[c] * alpha, 1);
      }
      const hit = await attrs.pick(x, y);
      expect(hit?.index).toBe(i);
      expect(hit?.mesh).toBe(splats.meshes[0]);
    }
    expect(await attrs.pick(1, 1)).toBeNull();
    attrs.dispose();
    splats.dispose();
  });

  // Slots resolve on the host to a word span of the draw layout: a later
  // comp4 group of a 3-word f32 tail, and a 2-word u16 group behind it.
  it("reads slots of any format and group into the slot targets", async () => {
    const splats = new WgpuSplatRenderer(fakeRenderer as never, {
      depthTest: false,
    });
    const attrs = new SplatAttributes(splats, {
      slots: [{ name: "wide", comp4: 1 }, "count"],
      colorMode: "splat",
      targets: { slot0: true, slot1: true },
    });
    const count = 3;
    const source = rowSource(count, 0.15, 1.2);
    const wide = Array.from({ length: count * 7 }, (_, k) =>
      k % 7 < 4 ? 9 : (k % 7) * 0.1 + Math.floor(k / 7) * 0.05,
    );
    const counts = [2, 30, 300, 5, 50, 500, 7, 70, 700];
    source.setAttribute("wide", wide, "f32", 7, { toDraw: true });
    source.setAttribute("count", counts, "u16", 3, { toDraw: true });
    splats.add(source);

    clearCanvas();
    splats.render(camera());
    await d.queue.onSubmittedWorkDone();

    const slots = await Promise.all(
      attrs.slotTextures.map(
        async (t) =>
          new Uint16Array((await readTexture(t as GPUTexture, 8)).buffer),
      ),
    );
    const cam = camera();
    for (let i = 0; i < count; i++) {
      const p = new THREE.Vector3((i - 1) * 1.2, 0, 0).project(cam);
      const x = Math.floor(((p.x + 1) / 2) * W);
      const y = Math.floor(((1 - p.y) / 2) * H);
      const px = y * W + x;
      const expected = [
        wide.slice(7 * i + 4, 7 * i + 7),
        counts.slice(3 * i, 3 * i + 3),
      ];
      slots.forEach((half, k) => {
        const alpha = fromHalf(half[px * 4 + 3]);
        expect(alpha).toBeGreaterThan(0.5);
        for (let c = 0; c < 3; c++) {
          const v = fromHalf(half[px * 4 + c]) / alpha;
          expect(Math.abs(v - expected[k][c])).toBeLessThan(
            0.01 * Math.max(1, expected[k][c]),
          );
        }
      });
    }
    attrs.dispose();
    splats.dispose();
  });

  // A stage added after a frame, the camera still: the next frame gathers.
  it("gathers for a stage added after the first frame", async () => {
    const splats = new WgpuSplatRenderer(fakeRenderer as never, {
      depthTest: false,
    });
    const source = rowSource(3, 0.15, 1.2);
    source.setAttribute("label", [1, 2, 3], "u8", 1, { toDraw: true });
    splats.add(source);
    const cam = camera();
    splats.render(cam);
    const attrs = new SplatAttributes(splats);
    splats.render(cam);
    await d.queue.onSubmittedWorkDone();
    expect(attrs.drawPool).not.toBeNull();
    attrs.dispose();
    splats.dispose();
  });

  // The portal clip applies to the attribute variant as to the default draw.
  it("clips attribute draws by the portal disk", async () => {
    const splats = new WgpuSplatRenderer(fakeRenderer as never, {
      depthTest: false,
    });
    const attrs = new SplatAttributes(splats, {
      colorMode: "splat",
      targets: { id: true },
    });
    splats.add(rowSource(3, 0.15, 1.2));
    const cam = camera();
    const pixel = (i: number): [number, number] => {
      const p = new THREE.Vector3((i - 1) * 1.2, 0, 0).project(cam);
      return [Math.floor(((p.x + 1) / 2) * W), Math.floor(((1 - p.y) / 2) * H)];
    };
    // A disk in front of the middle splat only, facing the camera.
    const drawn = async (radius: number) => {
      splats.diskClip = {
        center: new THREE.Vector3(0, 0, -3.5),
        normal: new THREE.Vector3(0, 0, 1),
        radius,
        twoSided: false,
      };
      clearCanvas();
      splats.render(cam);
      await d.queue.onSubmittedWorkDone();
      const out: number[] = [];
      for (let i = 0; i < 3; i++) {
        if ((await attrs.pick(...pixel(i)))?.index === i) out.push(i);
      }
      return out;
    };
    // Behind the disk, seen through it: the middle splat only...
    expect(await drawn(0.5)).toEqual([1]);
    // ...and dropping that: the other two.
    expect(await drawn(-0.5)).toEqual([0, 2]);
    expect(await drawn(0)).toEqual([0, 1, 2]);
    attrs.dispose();
    splats.dispose();
  });
});
