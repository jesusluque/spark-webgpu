// Back-to-front splat order on the GPU with slang/kernels/sort_radix.slang:
// keys from the generate pass's sort metric, eight stable 4-bit radix passes,
// and the active count written into indirect draw arguments, all recorded
// into one compute pass with no readback.

import type { KernelRegistry } from "./KernelRegistry";
import sortModule from "./generated/kernels/sort_radix";
import { UniformWriter } from "./uniforms";

const TILE = 128 * 8;
const SCAN_CHUNK = 512;
const BINS = 16;
const PASSES = 8;

function storage(device: GPUDevice, bytes: number, label: string, extra = 0) {
  return device.createBuffer({
    label,
    size: Math.max(16, Math.ceil(bytes / 16) * 16),
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST |
      extra,
  });
}

export class GpuSorter {
  readonly device: GPUDevice;
  /** Indirect draw arguments: 4 vertices, one instance per active splat. */
  readonly drawArgs: GPUBuffer;
  private capacity = 0;
  private keys: GPUBuffer[] = [];
  private vals: GPUBuffer[] = [];
  // blockHist, then the chunk sums of each scan level.
  private scanLevels: GPUBuffer[] = [];
  private empty: GPUBuffer;

  constructor(readonly registry: KernelRegistry) {
    this.device = registry.device;
    this.drawArgs = storage(
      this.device,
      16,
      "sort draw args",
      GPUBufferUsage.INDIRECT,
    );
    this.empty = storage(this.device, 16, "sort empty");
  }

  /** The sorted splat indices; valid for the first drawArgs[1] entries. */
  get ordering(): GPUBuffer {
    return this.vals[0];
  }

  private ensure(count: number) {
    if (count <= this.capacity) return;
    this.destroyBuffers();
    this.capacity = Math.max(count, Math.ceil(this.capacity * 1.5));
    const n = this.capacity;
    for (const k of [0, 1]) {
      this.keys.push(storage(this.device, n * 4, `sort keys ${k}`));
      this.vals.push(storage(this.device, n * 4, `sort vals ${k}`));
    }
    let size = BINS * Math.ceil(n / TILE);
    this.scanLevels.push(storage(this.device, size * 4, "sort block hist"));
    do {
      size = Math.ceil(size / SCAN_CHUNK);
      this.scanLevels.push(storage(this.device, size * 4, "sort chunk sums"));
    } while (size > 1);
  }

  /** Records the sort of `metric[0..count)` into `pass`. */
  encode(pass: GPUComputePassEncoder, metric: GPUBuffer, count: number) {
    this.ensure(count);
    this.device.queue.writeBuffer(
      this.drawArgs,
      0,
      new Uint32Array([4, 0, 0, 0]),
    );
    const numBlocks = Math.ceil(count / TILE);
    const sortParams = (shift: number) =>
      UniformWriter.for(sortModule, "params").setAll({
        count,
        shift,
        numBlocks,
      }).data;
    const get = (entry: string) => this.registry.get(sortModule, entry);

    get("prepareSort").dispatch(pass, {
      grid: [count],
      buffers: {
        sortMetric: metric,
        keysIn: this.keys[0],
        valsIn: this.vals[0],
        drawArgs: this.drawArgs,
      },
      uniforms: sortParams(0),
    });

    for (let p = 0; p < PASSES; p++) {
      const src = p & 1;
      const dst = src ^ 1;
      const uniforms = sortParams(4 * p);
      get("radixHistogram").dispatch(pass, {
        grid: [numBlocks * 128],
        buffers: { keysIn: this.keys[src], blockHist: this.scanLevels[0] },
        uniforms,
      });
      this.encodeScan(pass, BINS * numBlocks);
      get("radixScatter").dispatch(pass, {
        grid: [numBlocks * 128],
        buffers: {
          keysIn: this.keys[src],
          valsIn: this.vals[src],
          keysOut: this.keys[dst],
          valsOut: this.vals[dst],
          blockHist: this.scanLevels[0],
        },
        uniforms,
      });
    }
  }

  // Exclusive scan of scanLevels[0][0..n), level by level.
  private encodeScan(pass: GPUComputePassEncoder, n: number) {
    const scan = this.registry.get(sortModule, "scanChunks");
    const add = this.registry.get(sortModule, "addChunkOffsets");
    const sizes: number[] = [];
    let size = n;
    for (let level = 0; ; level++) {
      sizes.push(size);
      const chunks = Math.ceil(size / SCAN_CHUNK);
      scan.dispatch(pass, {
        grid: [chunks * 256],
        buffers: {
          scanData: this.scanLevels[level],
          chunkSums: this.scanLevels[level + 1] ?? this.empty,
        },
        uniforms: UniformWriter.for(sortModule, "scanParams").set("count", size)
          .data,
      });
      if (chunks === 1) break;
      size = chunks;
    }
    for (let level = sizes.length - 2; level >= 0; level--) {
      add.dispatch(pass, {
        grid: [sizes[level]],
        buffers: {
          scanData: this.scanLevels[level],
          chunkSums: this.scanLevels[level + 1],
        },
        uniforms: UniformWriter.for(sortModule, "scanParams").set(
          "count",
          sizes[level],
        ).data,
      });
    }
  }

  private destroyBuffers() {
    for (const b of [...this.keys, ...this.vals, ...this.scanLevels])
      b.destroy();
    this.keys = [];
    this.vals = [];
    this.scanLevels = [];
  }

  destroy() {
    this.destroyBuffers();
    this.drawArgs.destroy();
    this.empty.destroy();
  }
}
