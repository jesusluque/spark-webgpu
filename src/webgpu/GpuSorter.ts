// Back-to-front splat order on the GPU with slang/kernels/sort_radix.slang:
// keys from the generate pass's sort metric, compacted to the active splats,
// eight stable 4-bit radix passes dispatched over those alone, and the active
// count written into indirect draw arguments, all recorded into one compute
// pass with no readback.

import type { GpuProfiler } from "./GpuProfiler";
import type { KernelRegistry } from "./KernelRegistry";
import { kernelsSortRadix } from "./generated/constants";
import sortModule from "./generated/kernels/sort_radix";
import { createStorage as storage } from "./gpuBuffers";
import { UniformWriter } from "./uniforms";

const { TILE, SCAN_CHUNK, BINS } = kernelsSortRadix;

export class GpuSorter {
  readonly device: GPUDevice;
  /** Indirect draw arguments: 4 vertices, one instance per active splat. */
  readonly drawArgs: GPUBuffer;
  // Workgroup counts of the radix passes, for the active keys only.
  private dispatchArgs: GPUBuffer;
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
    this.dispatchArgs = storage(
      this.device,
      16,
      "sort dispatch args",
      GPUBufferUsage.INDIRECT,
    );
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

  /**
   * Records the sort of `metric[0..count)` into `pass`, on the top `bits` of
   * the key (16, 24 or 32: 4, 6 or 8 passes; even, so the result lands in
   * `ordering`). Dropping low bits only merges splats whose metrics differ
   * by less than 2^-7 (16) or 2^-15 (24) relative.
   */
  encode(
    pass: GPUComputePassEncoder,
    metric: GPUBuffer,
    count: number,
    bits: 16 | 24 | 32 = 32,
  ) {
    this.encodeStages(() => pass, metric, count, bits);
  }

  /**
   * As encode, with one compute pass per stage on `encoder` so a profiler
   * can time each (labels sort.prepare, sort.histogram, sort.scan,
   * sort.scatter; summed over the radix passes).
   */
  encodeProfiled(
    encoder: GPUCommandEncoder,
    profiler: GpuProfiler,
    metric: GPUBuffer,
    count: number,
    bits: 16 | 24 | 32 = 32,
  ) {
    let open: GPUComputePassEncoder | null = null;
    this.encodeStages(
      (stage) => {
        open?.end();
        const label = `sort.${stage}`;
        open = encoder.beginComputePass({
          label,
          timestampWrites: profiler.timestampWrites(label),
        });
        return open;
      },
      metric,
      count,
      bits,
    );
    (open as GPUComputePassEncoder | null)?.end();
  }

  // `pass(stage)` gives the pass to record each stage's dispatches into.
  private encodeStages(
    stagePass: (stage: string) => GPUComputePassEncoder,
    metric: GPUBuffer,
    count: number,
    bits: 16 | 24 | 32,
  ) {
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

    // Only the active (finite) metrics become keys, in index order: count
    // them per block, scan the counts, write the keys at those offsets.
    const prepare = stagePass("prepare");
    const blocks = {
      grid: [numBlocks * 128] as const,
      uniforms: sortParams(0),
    };
    get("countActive").dispatch(prepare, {
      ...blocks,
      buffers: {
        sortMetric: metric,
        blockHist: this.scanLevels[0],
        drawArgs: this.drawArgs,
      },
    });
    this.encodeScan(prepare, numBlocks);
    get("compactKeys").dispatch(prepare, {
      ...blocks,
      buffers: {
        sortMetric: metric,
        blockHist: this.scanLevels[0],
        keysOut: this.keys[0],
        valsOut: this.vals[0],
      },
    });
    get("writeDispatch").dispatch(prepare, {
      grid: [1],
      buffers: { sortCount: this.drawArgs, dispatchArgs: this.dispatchArgs },
    });

    // The radix passes run over the active blocks only (dispatchArgs).
    const passes = bits / 4;
    for (let p = 0; p < passes; p++) {
      const src = p & 1;
      const dst = src ^ 1;
      const uniforms = sortParams(32 - bits + 4 * p);
      get("radixHistogram").dispatchIndirect(
        stagePass("histogram"),
        {
          buffers: {
            keysIn: this.keys[src],
            blockHist: this.scanLevels[0],
            sortCount: this.drawArgs,
          },
          uniforms,
        },
        this.dispatchArgs,
      );
      this.encodeScan(stagePass("scan"), BINS * numBlocks);
      get("radixScatter").dispatchIndirect(
        stagePass("scatter"),
        {
          buffers: {
            keysIn: this.keys[src],
            valsIn: this.vals[src],
            keysOut: this.keys[dst],
            valsOut: this.vals[dst],
            blockHist: this.scanLevels[0],
            sortCount: this.drawArgs,
          },
          uniforms,
        },
        this.dispatchArgs,
      );
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
      if (chunks <= 1) break;
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
    this.dispatchArgs.destroy();
    this.empty.destroy();
  }
}
