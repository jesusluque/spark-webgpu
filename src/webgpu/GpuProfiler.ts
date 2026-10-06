// GPU pass timings from timestamp queries, for WgpuSplatRenderer's stats.
//
// Core WebGPU only writes timestamps at the start and end of a pass, so a
// stage is timed by giving it its own pass (the sorter splits its dispatches
// into one pass per stage while profiling). The feature is optional: Chrome
// and Safari 26 on Mac have it, and three's WebGPURenderer requests it when
// the adapter does; create() returns null without it.
//
// Chrome rounds timestamps to 100 µs unless WebGPU developer features are on
// (chrome://flags, or --enable-webgpu-developer-features), so short stages
// read as 0 or 0.1 ms there; the frame totals stay meaningful.

const MAX_QUERIES = 128;
const READBACKS = 3;

export class GpuProfiler {
  /** Smoothed milliseconds per label (same labels in a frame are summed). */
  readonly ms: Record<string, number> = {};
  /** The latest frame read back, unsmoothed. */
  last: Record<string, number> = {};
  /** Frames whose timings have been read back. */
  resolved = 0;

  private querySet: GPUQuerySet;
  private resolveBuffer: GPUBuffer;
  private free: GPUBuffer[] = [];
  private labels: string[] = [];

  private constructor(
    readonly device: GPUDevice,
    /** Weight of the newest frame in the smoothed values. */
    readonly smoothing = 0.1,
  ) {
    this.querySet = device.createQuerySet({
      label: "GpuProfiler",
      type: "timestamp",
      count: MAX_QUERIES,
    });
    this.resolveBuffer = device.createBuffer({
      label: "GpuProfiler resolve",
      size: MAX_QUERIES * 8,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });
    for (let i = 0; i < READBACKS; i++) {
      this.free.push(
        device.createBuffer({
          label: "GpuProfiler readback",
          size: MAX_QUERIES * 8,
          usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        }),
      );
    }
  }

  /** A profiler, or null when the device lacks timestamp-query. */
  static create(device: GPUDevice, smoothing?: number): GpuProfiler | null {
    if (!device.features.has("timestamp-query")) return null;
    return new GpuProfiler(device, smoothing);
  }

  /**
   * Timestamp writes for a pass labelled `label`, or undefined when this
   * frame's queries are used up or no readback buffer is free (the frame is
   * then partly timed; resolve() drops it).
   */
  timestampWrites(
    label: string,
  ):
    | (GPUComputePassTimestampWrites & GPURenderPassTimestampWrites)
    | undefined {
    const i = this.labels.length;
    if (2 * i + 2 > MAX_QUERIES || this.free.length === 0) return undefined;
    this.labels.push(label);
    return {
      querySet: this.querySet,
      beginningOfPassWriteIndex: 2 * i,
      endOfPassWriteIndex: 2 * i + 1,
    };
  }

  /**
   * Records copying this frame's timestamps out, and returns a callback to
   * run after the encoder is submitted.
   */
  resolve(encoder: GPUCommandEncoder): (() => void) | null {
    const labels = this.labels;
    this.labels = [];
    const readback = this.free.pop();
    if (labels.length === 0 || !readback) {
      if (readback) this.free.push(readback);
      return null;
    }
    const bytes = labels.length * 16;
    encoder.resolveQuerySet(
      this.querySet,
      0,
      2 * labels.length,
      this.resolveBuffer,
      0,
    );
    encoder.copyBufferToBuffer(this.resolveBuffer, 0, readback, 0, bytes);
    return () => {
      readback.mapAsync(GPUMapMode.READ, 0, bytes).then(
        () => {
          const t = new BigInt64Array(readback.getMappedRange(0, bytes));
          const frame: Record<string, number> = {};
          let total = 0;
          let first = t[0];
          let last = t[0];
          labels.forEach((label, i) => {
            const ms = Number(t[2 * i + 1] - t[2 * i]) / 1e6;
            // Skip garbage from passes that never ran (end before start).
            if (ms < 0 || ms > 1e4) return;
            frame[label] = (frame[label] ?? 0) + ms;
            total += ms;
            if (t[2 * i] < first) first = t[2 * i];
            if (t[2 * i + 1] > last) last = t[2 * i + 1];
          });
          frame.total = total;
          // First start to last end: passes may overlap (a render pass
          // starting while the compute before it finishes), so this can be
          // under the total.
          frame.span = Number(last - first) / 1e6;
          this.last = frame;
          readback.unmap();
          this.free.push(readback);
          const k = this.resolved === 0 ? 1 : this.smoothing;
          for (const [label, ms] of Object.entries(frame)) {
            this.ms[label] = (this.ms[label] ?? ms) * (1 - k) + ms * k;
          }
          this.resolved += 1;
        },
        () => this.free.push(readback),
      );
    };
  }

  /** Forgets the smoothed values (e.g. between benchmark runs). */
  reset() {
    for (const k of Object.keys(this.ms)) delete this.ms[k];
    this.resolved = 0;
  }

  destroy() {
    this.querySet.destroy();
    this.resolveBuffer.destroy();
    for (const b of this.free) b.destroy();
    this.free = [];
  }
}
