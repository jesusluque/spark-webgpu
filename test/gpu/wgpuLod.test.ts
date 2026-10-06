// WgpuLod on Dawn with the LoD worker run in-process (the real WASM trees
// and traversal): tree level queries, raycast selections, hidden meshes and
// the timed release of trees no mesh draws.

import { readFileSync } from "node:fs";
import * as THREE from "three";
import { describe, expect, it, vi } from "vitest";
import { PackedSplats } from "../../src/PackedSplats";
import { SparkRenderer } from "../../src/SparkRenderer";
import { SplatMesh } from "../../src/SplatMesh";
import { SplatWorker } from "../../src/SplatWorker";
import { setPackedSplat } from "../../src/utils";
import { WgpuLod } from "../../src/webgpu/WgpuLod";
import { WgpuSplatRenderer } from "../../src/webgpu/WgpuSplatRenderer";
import { device } from "./device";

// test/unit/setup.ts stubs the wasm package; these tests need the real one.
const wasm = await vi.importActual<typeof import("spark-rs")>("spark-rs");
wasm.initSync({
  module: readFileSync(
    new URL("../../rust/spark-rs/pkg/spark_rs_bg.wasm", import.meta.url),
  ),
});

function rng(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 2 ** 32;
  };
}

// A LoD PackedSplats: a random cloud and its tiny-LoD tree.
function lodPackedSplats(n = 20000) {
  const r = rng(3);
  const packed = new Uint32Array(n * 4);
  for (let i = 0; i < n; i++) {
    const s = 0.01 + 0.02 * r();
    setPackedSplat(
      packed,
      i,
      r() * 4 - 2,
      r() * 4 - 2,
      r() * 4 - 2,
      s,
      s,
      s,
      0,
      0,
      0,
      1,
      0.3 + 0.6 * r(),
      r(),
      r(),
      r(),
    );
  }
  const encoding = {
    rgbMin: 0,
    rgbMax: 1,
    lnScaleMin: -12,
    lnScaleMax: 9,
    sh1Max: 1,
    sh2Max: 1,
    sh3Max: 1,
    lodOpacity: false,
  };
  const lod = wasm.tiny_lod_packedsplats(
    n,
    packed.slice(),
    {},
    1.5,
    false,
    undefined,
    encoding,
  ) as { numSplats: number; packed: Uint32Array; lodTree: Uint32Array };
  return new PackedSplats({
    packedArray: packed,
    numSplats: n,
    lodSplats: new PackedSplats({
      packedArray: lod.packed,
      numSplats: lod.numSplats,
      extra: { lodTree: lod.lodTree },
    }),
  });
}

type Instance = {
  lodId: number;
  rootPage?: number;
  viewToObjectCols: number[];
  lodScale: number;
  behindFoveate: number;
  coneFov0: number;
  coneFov: number;
  coneFoveate: number;
};

// SplatWorker's queue, with the RPCs WgpuLod makes answered in-process as
// src/worker.ts answers them. `calls` records each call's name and args.
function inProcessWorker() {
  const calls: { name: string; args: Record<string, unknown> }[] = [];
  const handlers: Record<string, (args: never) => unknown> = {
    initLodTree: ({ numSplats, lodTree }: Record<string, never>) => {
      const { lodId } = wasm.init_lod_tree(numSplats, lodTree) as {
        lodId: number;
      };
      return { lodId };
    },
    disposeLodTree: ({ lodId }: { lodId: number }) =>
      wasm.dispose_lod_tree(lodId),
    getLodTreeLevel: ({ lodId, level }: { lodId: number; level: number }) =>
      wasm.get_lod_tree_level(lodId, level),
    traverseLodTrees: ({
      maxSplats,
      pixelScaleLimit,
      lastPixelLimit,
      instances,
    }: {
      maxSplats: number;
      pixelScaleLimit: number;
      lastPixelLimit?: number;
      instances: Record<string, Instance>;
    }) => {
      const entries = Object.entries(instances);
      const pick = (f: (i: Instance) => number) =>
        new Float32Array(entries.map(([, i]) => f(i)));
      const result = wasm.traverse_lod_trees(
        maxSplats,
        pixelScaleLimit,
        lastPixelLimit,
        new Uint32Array(entries.map(([, i]) => i.lodId)),
        new Uint32Array(entries.map(([, i]) => i.rootPage ?? 0xffffffff)),
        new Float32Array(entries.flatMap(([, i]) => i.viewToObjectCols)),
        pick((i) => i.lodScale),
        pick((i) => i.behindFoveate),
        pick((i) => i.coneFoveate),
        pick((i) => i.coneFov0),
        pick((i) => i.coneFov),
      ) as {
        instanceIndices: { numSplats: number; indices: Uint32Array }[];
        chunks: [number, number][];
        pixelLimit?: number;
      };
      return {
        keyIndices: Object.fromEntries(
          entries.map(([key], k) => [key, result.instanceIndices[k]]),
        ),
        chunks: result.chunks,
        pixelLimit: result.pixelLimit,
      };
    },
  };
  const worker = Object.create(SplatWorker.prototype) as SplatWorker;
  worker.queue = null;
  worker.worker = { terminate() {} } as Worker;
  worker.call = (async (name: string, args: Record<string, unknown>) => {
    calls.push({ name, args });
    const handler = handlers[name];
    if (!handler) throw new Error(`no in-process handler for ${name}`);
    return handler(args as never);
  }) as SplatWorker["call"];
  // Resolves once the LoD update started before it has finished.
  const idle = () => worker.exclusive(async () => {});
  return { worker, calls, idle };
}

describe.skipIf(!device)("WgpuLod", () => {
  const d = device as GPUDevice;
  const canvas = d.createTexture({
    size: [4, 4],
    format: "rgba8unorm",
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });
  const fakeRenderer = {
    backend: {
      isWebGPUBackend: true,
      device: d,
      context: { getCurrentTexture: () => canvas },
      get: () => undefined,
    },
  };
  const size = { x: 600, y: 400 };

  function camera() {
    const cam = new THREE.PerspectiveCamera(60, 1.5, 0.05, 100);
    cam.position.set(0, 0, 3);
    cam.updateMatrixWorld();
    return cam;
  }

  function setup(options: ConstructorParameters<typeof WgpuLod>[1] = {}) {
    const splats = new WgpuSplatRenderer(fakeRenderer as never);
    const lod = new WgpuLod(splats, { lodSplatCount: 5000, ...options });
    const worker = inProcessWorker();
    (lod as unknown as { worker: SplatWorker }).worker = worker.worker;
    const frame = async (cam = camera()) => {
      lod.update(cam, size);
      await worker.idle();
    };
    const dispose = () => {
      lod.dispose();
      splats.dispose();
    };
    return { splats, lod, frame, dispose, ...worker };
  }

  it("answers tree level queries for a drawn mesh's LoD splats", async () => {
    const { lod, frame, dispose } = setup();
    const packed = lodPackedSplats();
    const lodSplats = packed.lodSplats as PackedSplats;
    expect(await lod.getLodTreeLevel(lodSplats, 0)).toBeNull();

    await lod.add(packed);
    await frame();
    const root = await lod.getLodTreeLevel(lodSplats, 0);
    expect(Array.from(root ?? [])).toEqual([0]);
    const level = (await lod.getLodTreeLevel(lodSplats, 12)) as Uint32Array;
    expect(level.length).toBeGreaterThan(1);
    expect(Math.max(...level)).toBeLessThan(lodSplats.numSplats);
    dispose();
  });

  it("keeps a coarser raycast selection, as SparkRenderer.lodRaycast", async () => {
    const { lod, frame, dispose } = setup({ lodRaycast: 200 });
    const m = await lod.add(lodPackedSplats());
    await frame();
    const drawn = m.lastIndices.length;
    expect(drawn).toBeGreaterThan(1000);
    const raycast = m.raycastIndices as { numSplats: number };
    expect(raycast.numSplats).toBeGreaterThan(0);
    expect(raycast.numSplats).toBeLessThanOrEqual(
      Math.min(200, Math.round(drawn * 0.1)),
    );
    dispose();

    const none = setup({ lodRaycast: 0 });
    const m2 = await none.lod.add(lodPackedSplats());
    await none.frame();
    expect(m2.raycastIndices).toBeUndefined();
    none.dispose();
  });

  it("hides a mesh and shows it again with its tree and selection", async () => {
    const { splats, lod, frame, calls, dispose } = setup();
    const m = await lod.add(lodPackedSplats());
    await frame();
    const selection = m.lastIndices;
    expect(selection.length).toBeGreaterThan(0);

    lod.setVisible(m, false);
    expect(m.mesh.lodIndices?.length).toBe(0);
    calls.length = 0;
    await frame();
    // Not traversed while hidden; nothing rebuilt or released.
    const traversed = calls.filter((c) => c.name === "traverseLodTrees");
    expect(traversed.every((c) => !(m.id in (c.args.instances as object))));
    expect(splats.meshes).toContain(m.mesh);

    lod.setVisible(m, true);
    expect(m.mesh.lodIndices).toBe(selection);
    await frame();
    expect(calls.map((c) => c.name)).not.toContain("initLodTree");
    expect(calls.map((c) => c.name)).not.toContain("disposeLodTree");
    expect(lod.meshes).toEqual([m]);
    dispose();
  });

  it("releases trees no mesh drew for lodCleanupTimeoutMs", async () => {
    const kept = setup();
    const packed = lodPackedSplats();
    const lodSplats = packed.lodSplats as PackedSplats;
    const first = await kept.lod.add(packed);
    await kept.frame();
    // Within the timeout: the tree stays, and a new mesh of the same splats
    // reuses it.
    kept.lod.remove(first);
    kept.calls.length = 0;
    await kept.frame();
    await kept.lod.add(packed);
    await kept.frame();
    const names = kept.calls.map((c) => c.name);
    expect(names).not.toContain("initLodTree");
    expect(names).not.toContain("disposeLodTree");
    expect(await kept.lod.getLodTreeLevel(lodSplats, 0)).not.toBeNull();
    kept.dispose();

    const released = setup({ lodCleanupTimeoutMs: 0 });
    const m = await released.lod.add(packed);
    await released.frame();
    const hidden = await released.lod.add(packed);
    released.lod.setVisible(hidden, false);
    // Still drawn by m: kept.
    await released.frame();
    expect(released.lod.active).toBe(true);
    expect(released.calls.map((c) => c.name)).not.toContain("disposeLodTree");
    // Drawn by none: released on the next update, then re-built on show.
    released.lod.remove(m);
    await released.frame();
    expect(released.calls.map((c) => c.name)).toContain("disposeLodTree");
    expect(await released.lod.getLodTreeLevel(lodSplats, 0)).toBeNull();
    released.calls.length = 0;
    released.lod.setVisible(hidden, true);
    await released.frame();
    expect(released.calls.map((c) => c.name)).toContain("initLodTree");
    expect(hidden.lastIndices.length).toBeGreaterThan(0);
    released.lod.remove(hidden);
    await released.frame();
    expect(released.lod.active).toBe(false);
    released.dispose();
  });

  // SparkRenderer on WebGPU: a LoD SplatMesh hidden and shown again reuses
  // its WgpuLod mesh, and gets raycast indices from it.
  it("reuses a hidden SplatMesh's LoD mesh through SparkRenderer", async () => {
    const W = 64;
    const target = d.createTexture({
      size: [W, W],
      format: "rgba8unorm",
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    const rc = {
      textures: null,
      depthTexture: null,
      renderTarget: null,
      width: W,
      height: W,
      viewport: false,
      viewportValue: new THREE.Vector4(0, 0, W, W),
    };
    const rcData: Record<string, unknown> = {};
    const three = {
      isWebGPURenderer: true,
      info: { frame: 0 },
      outputColorSpace: THREE.SRGBColorSpace,
      getOutputRenderTarget: () => null,
      _currentRenderContext: rc,
      _frameBufferTarget: null,
      backend: {
        isWebGPUBackend: true,
        device: d,
        context: {
          getCurrentTexture: () => target,
          canvas: { width: W, height: W },
        },
        get: (resource: object) => (resource === rc ? rcData : {}),
        utils: {
          getCurrentColorFormat: () => target.format,
          getCurrentDepthStencilFormat: () => undefined,
          getSampleCountRenderContext: () => 1,
          getCurrentColorSpace: () => THREE.SRGBColorSpace,
        },
        updateViewport: () => {},
        pipelineUtils: { _activePipelines: new WeakMap() },
      },
    };
    const spark = new SparkRenderer({ renderer: three as never });
    spark.lodSplatCount = 5000;
    spark.lodRaycast = 100;
    const scene = new THREE.Scene();
    scene.add(spark);
    const mesh = new SplatMesh({ packedSplats: lodPackedSplats() });
    scene.add(mesh);
    const cam = camera();
    const { worker, calls, idle } = inProcessWorker();
    const render = async () => {
      three.info.frame += 1;
      const encoder = d.createCommandEncoder();
      const descriptor: GPURenderPassDescriptor = {
        colorAttachments: [
          { view: target.createView(), loadOp: "clear", storeOp: "store" },
        ],
      };
      rcData.descriptor = descriptor;
      rcData.currentPass = encoder.beginRenderPass(descriptor);
      spark.onBeforeRender(three as never, scene, cam);
      (rcData.currentPass as GPURenderPassEncoder).end();
      d.queue.submit([encoder.finish()]);
      const lod = spark.webgpu?.lod as unknown as { worker: SplatWorker };
      lod.worker = worker;
      await idle();
      // WgpuLod.add's then.
      await new Promise((resolve) => setTimeout(resolve, 0));
    };
    await mesh.initialized;
    for (let i = 0; i < 3; i++) await render();
    const lod = spark.webgpu?.lod as WgpuLod;
    const [lodMesh] = lod.meshes;
    expect(lodMesh.lastIndices.length).toBeGreaterThan(0);
    expect(mesh.raycastIndices).toBe(lodMesh.raycastIndices);
    expect(mesh.raycastIndices?.numSplats).toBeLessThanOrEqual(100);

    mesh.visible = false;
    await render();
    expect(lod.meshes).toEqual([lodMesh]);
    expect(lodMesh.visible).toBe(false);
    calls.length = 0;
    mesh.visible = true;
    await render();
    expect(lod.meshes).toEqual([lodMesh]);
    expect(lodMesh.visible).toBe(true);
    expect(calls.map((c) => c.name)).not.toContain("initLodTree");

    scene.remove(mesh);
    await render();
    expect(lod.meshes).toEqual([]);
    spark.dispose();
  });
});
