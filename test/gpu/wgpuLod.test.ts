// WgpuLod on Dawn with the LoD worker run in-process (the real WASM trees
// and traversal): tree level queries, raycast selections, hidden meshes and
// the timed release of trees no mesh draws.

import { readFileSync } from "node:fs";
import * as THREE from "three";
import { describe, expect, it, vi } from "vitest";
import { PackedSplats } from "../../src/PackedSplats";
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
});
