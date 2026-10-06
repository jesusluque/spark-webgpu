// SplatPager's page pool in storage buffers instead of WebGL texture arrays.
//
// The fetch queue, page LRU and LoD tree updates are SplatPager's, unchanged.
// Only the residence changes: page p holds splats [p * 65536, (p + 1) * 65536)
// of each pool buffer, in the interleaved layout kernels/generate.slang reads
// (see GpuSplatSource). The LoD traversal already returns pool indices
// (pageBase + offset), so a paged mesh is drawn as an ordinary source over the
// whole pool, remapped through those indices.
//
// One buffer per pool (core, SH, later attributes) keeps the storage-buffer
// count of generate unchanged. Each must fit one binding, so the pool is
// capped by the device's maxStorageBufferBindingSize: with default limits
// (128 MiB) that is 8M packed splats without SH, 2.7M with SH3. Request the
// adapter's limits from WebGPURenderer (requiredLimits) for more.

import * as THREE from "three";
import type { PagedSplats } from "../PagedSplats";
import { SplatPager, type SplatPagerOptions } from "../SplatPager";
import { GpuSplatSource } from "./GpuSplatSource";
import { KernelRegistry } from "./KernelRegistry";
import restrideModule from "./generated/kernels/pool_restride";
import { createStorage } from "./gpuBuffers";

const PAGE_SPLATS = 65536;

export interface WgpuSplatPagerOptions
  extends Omit<SplatPagerOptions, "renderer"> {}

interface Pool {
  wordsPerSplat: number;
  buffer: GPUBuffer | null;
}

// SplatPager calls this from its constructor for textures we never use.
const unusedRenderer = {
  initTexture() {},
} as unknown as THREE.WebGLRenderer;

export class WgpuSplatPager extends SplatPager {
  readonly device: GPUDevice;
  // TODO(attributes): page the AttribPool (src/webgpu/attributes/ on the
  // integrated branch) as a third pool: its schema comes from RadMeta
  // attributes, each chunk's `attrib` data uploads at the page base like
  // core and SH, and source() returns the pool as GpuSplatSource.attribs.
  // Paged chunk decoding skips attributes for now.
  /** Per-splat words of each pool, by name. */
  readonly pools: Record<"core" | "sh", Pool>;
  // SH degrees the SH pool's stride holds: it is sized for the data that
  // arrives (an SH1 scene takes a third of SH3's memory) and widened when a
  // page with more degrees comes.
  private shDegree = 0;
  private registry: KernelRegistry | null = null;

  constructor(device: GPUDevice, options: WgpuSplatPagerOptions) {
    const ext = options.extSplats ?? false;
    const maxSh = options.maxSh ?? 3;
    const coreWords = ext ? 8 : 4;
    const shWords = maxSh > 0 ? (ext ? 16 : 12) : 0;
    const limit = Math.min(
      device.limits.maxStorageBufferBindingSize,
      device.limits.maxBufferSize,
    );
    const maxWords = Math.max(coreWords, shWords);
    const fitPages = Math.floor(limit / (PAGE_SPLATS * maxWords * 4));
    const wanted = options.maxSplats ?? 256 * PAGE_SPLATS;
    super({
      ...options,
      renderer: unusedRenderer,
      maxSplats: Math.min(wanted, fitPages * PAGE_SPLATS),
    });
    if (this.maxSplats < wanted) {
      console.warn(
        `WgpuSplatPager: ${this.maxPages} pages fit maxStorageBufferBindingSize (${limit}), not ${wanted / PAGE_SPLATS}`,
      );
    }
    this.device = device;
    this.pools = {
      core: { wordsPerSplat: coreWords, buffer: null },
      sh: { wordsPerSplat: 0, buffer: null },
    };
    this.ensurePool(this.pools.core, "splat pages");
  }

  private ensurePool(pool: Pool, label: string) {
    if (!pool.buffer) {
      pool.buffer = createStorage(
        this.device,
        this.maxSplats * pool.wordsPerSplat * 4,
        label,
      );
    }
    return pool.buffer;
  }

  /** The pool as a source for WgpuSplatRenderer, with `splats`' encoding. */
  source(splats: PagedSplats): GpuSplatSource {
    const sh = this.pools.sh.buffer;
    const numSh = sh ? Math.min(splats.numSh, splats.maxSh, this.curSh) : 0;
    const e = splats.splatEncoding;
    const source = new GpuSplatSource(
      this.extSplats ? "ext" : "packed",
      this.maxSplats,
      this.pools.core.buffer as GPUBuffer,
      sh,
      numSh,
      e ? [e.sh1Max, e.sh2Max, e.sh3Max] : undefined,
      e ? [e.rgbMin, e.rgbMax, e.lnScaleMin, e.lnScaleMax] : undefined,
      e?.lodOpacity ?? false,
    );
    source.shStride = this.pools.sh.wordsPerSplat / 4;
    return source;
  }

  // uint4s per splat for `degree` SH degrees: packed sh1 (padded), sh2, sh3;
  // ext sh1, sh2, sh3a + sh3b.
  private shSlots(degree: number) {
    return degree === 3 && this.extSplats ? 4 : degree;
  }

  // Makes the SH pool hold `degree` degrees, moving the pages already there.
  private ensureShDegree(degree: number) {
    if (degree <= this.shDegree) return;
    const sh = this.pools.sh;
    const oldSlots = sh.wordsPerSplat / 4;
    const newSlots = this.shSlots(degree);
    const old = sh.buffer;
    sh.wordsPerSplat = 4 * newSlots;
    sh.buffer = null;
    const buffer = this.ensurePool(sh, "SH pages");
    if (old) {
      this.registry ??= new KernelRegistry(this.device);
      this.registry.get(restrideModule, "restride").run({
        grid: [this.maxSplats],
        buffers: { src: old, dst: buffer },
        uniforms: new Uint32Array([this.maxSplats, oldSlots, newSlots, 0]),
      });
      old.destroy();
    }
    this.shDegree = degree;
  }

  // No textures: SplatPager's dyno blocks are never compiled on this path.
  protected newUint32ArrayTexture(): THREE.DataArrayTexture {
    return new THREE.DataArrayTexture(new Uint32Array(4), 1, 1, 1);
  }

  protected uploadPage(
    page: number,
    packedArray: Uint32Array,
    shArrays: Array<Uint32Array>,
    extArray?: Uint32Array,
  ) {
    const base = page * PAGE_SPLATS;
    const core = this.pools.core;
    const count = packedArray.length / 4;
    if (extArray) {
      // Ext pages arrive as two arrays; the pool interleaves a and b.
      const words = new Uint32Array(count * 8);
      for (let i = 0; i < count; i++) {
        words.set(packedArray.subarray(4 * i, 4 * i + 4), 8 * i);
        words.set(extArray.subarray(4 * i, 4 * i + 4), 8 * i + 4);
      }
      this.write(core, base, words);
    } else {
      this.write(core, base, packedArray);
    }

    const numSh = Math.min(shArrays.length, 3);
    if (numSh === 0 || this.maxSh === 0) return;
    const degree = Math.min(numSh, this.maxSh);
    this.curSh = Math.max(this.curSh, degree);
    this.ensureShDegree(degree);
    const sh = this.pools.sh;
    // Packed: sh1 (2 words, padded to 4), sh2, sh3. Ext: sh1, sh2, sh3a, sh3b.
    const stride = sh.wordsPerSplat;
    const perArray = this.extSplats ? [4, 4, 4, 4] : [2, 4, 4];
    const arrays = shArrays.slice(0, this.shSlots(degree));
    const words = new Uint32Array(count * stride);
    for (let i = 0; i < count; i++) {
      arrays.forEach((array, k) => {
        const n = perArray[k];
        words.set(array.subarray(n * i, n * i + n), stride * i + 4 * k);
      });
    }
    this.write(sh, base, words);
  }

  private write(pool: Pool, splatBase: number, words: Uint32Array) {
    this.device.queue.writeBuffer(
      pool.buffer as GPUBuffer,
      splatBase * pool.wordsPerSplat * 4,
      words.buffer,
      words.byteOffset,
      words.byteLength,
    );
  }

  dispose() {
    super.dispose();
    this.registry?.destroy();
    for (const pool of Object.values(this.pools)) {
      pool.buffer?.destroy();
      pool.buffer = null;
    }
  }
}
