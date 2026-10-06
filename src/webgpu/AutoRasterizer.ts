// WgpuSplatRenderer's rasterizer: "auto": which draw path to take each
// frame, from measured GPU times of the draw on each path.
//
// Neither path wins everywhere. The tile rasterizer stops a pixel once it is
// opaque, so it wins under heavy overdraw (dense or close-up scenes, 2-5x);
// the quads win on most captured scenes, where each tile still evaluates all
// 256 pixels for every listed splat. Nothing cheap computed before the draw
// predicts the order across scenes (pairs per tile, say, is no lower on
// captured scenes the quads win than on synthetic ones the tiles win), so
// the draw is timed with timestamp queries: the current path every frame it
// can be, and the other one in short probes. The two paths' images match to
// about 0.5/255, so a probe frame does not show.
//
// Probes back off while they confirm the current path (base to maxInterval
// frames), and come back to base when the current path's time moves by more
// than the gap the last probe saw (1.25x to 2x), since only then may the
// order have flipped. A probe of a much slower path is a few slow frames, so
// a still scene where one path is far ahead pays for it once every
// maxInterval frames.

import type { GpuProfiler } from "./GpuProfiler";

export type RasterPath = "hardware" | "tiles";

export interface AutoRasterizerOptions {
  /** Frames before the first probe, and between probes after a change. */
  baseInterval?: number;
  /** Longest interval the backoff reaches. */
  maxInterval?: number;
  /** Frames a probe draws on the other path (DrawTimer's batch). */
  probeFrames?: number;
  /**
   * Frames a probe waits for its timings before giving up: readbacks can
   * lag hundreds of frames in an unthrottled loop.
   */
  probeTimeout?: number;
  /** The other path must be this much faster (relative) to switch... */
  margin?: number;
  /** ...and faster by this many ms (Chrome rounds timestamps to 0.1 ms). */
  marginMs?: number;
  /** Recent timings (batches) of the current path for its estimate (a median). */
  window?: number;
}

/** What the policy knows, for WgpuSplatRenderer.stats.auto. */
export interface AutoRasterizerState {
  path: RasterPath;
  /** Median draw ms of each path, as last measured (null: never). */
  hardwareMs: number | null;
  tilesMs: number | null;
  probing: boolean;
  probes: number;
  switches: number;
  /** Frames between probes now. */
  interval: number;
}

const median = (v: readonly number[]) => {
  const s = [...v].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length & 1 ? s[m] : (s[m - 1] + s[m]) / 2;
};

const other = (p: RasterPath): RasterPath =>
  p === "hardware" ? "tiles" : "hardware";

/**
 * The frame-by-frame choice, fed timings: next() says which path to draw a
 * frame with, measured() reports a frame's draw time on a path whenever its
 * timestamps come back (frames later, possibly not at all).
 */
export class AutoRasterizer {
  readonly options: Required<AutoRasterizerOptions>;
  path: RasterPath = "hardware";
  probes = 0;
  switches = 0;
  interval: number;
  private frame = 0;
  private nextProbe: number;
  // Probe frames still to draw, and the timings it has received.
  private probeLeft = 0;
  private probeUntil = -1;
  private probeSamples: number[] = [];
  private recent: number[] = [];
  private estimate: Record<RasterPath, number | null> = {
    hardware: null,
    tiles: null,
  };
  // The current path's estimate at the last decision, and how far ahead of
  // the other path it was then (other / current, at least 1).
  private reference: number | null = null;
  private gap = 1;

  constructor(options: AutoRasterizerOptions = {}) {
    this.options = {
      baseInterval: 30,
      maxInterval: 960,
      probeFrames: 4,
      probeTimeout: 600,
      margin: 0.1,
      marginMs: 0.2,
      window: 5,
      ...options,
    };
    this.interval = this.options.baseInterval;
    this.nextProbe = this.options.baseInterval;
  }

  get probing() {
    return this.probeUntil >= 0;
  }

  /** The path to draw the next frame with. */
  next(): RasterPath {
    this.frame += 1;
    if (this.probing) {
      if (this.frame > this.probeUntil) {
        this.finishProbe();
      } else if (this.probeLeft > 0) {
        this.probeLeft -= 1;
        return other(this.path);
      }
      return this.path;
    }
    if (this.frame >= this.nextProbe && this.recent.length > 0) {
      this.probes += 1;
      this.probeSamples = [];
      this.probeLeft = this.options.probeFrames - 1;
      this.probeUntil = this.frame + this.options.probeTimeout;
      return other(this.path);
    }
    return this.path;
  }

  /**
   * One more probe frame, and as long to wait (the caller drew this one on
   * the current path, e.g. while the other cannot draw yet).
   */
  extendProbe() {
    if (!this.probing) return;
    this.probeLeft += 1;
    this.probeUntil += 1;
  }

  /** A frame drawn on `path` took `ms` on the GPU. */
  measured(path: RasterPath, ms: number) {
    if (!(ms >= 0)) return;
    if (path !== this.path) {
      // Late timings of an earlier path or probe are stale.
      if (!this.probing) return;
      this.probeSamples.push(ms);
      if (this.probeLeft === 0) this.finishProbe();
      return;
    }
    this.recent.push(ms);
    if (this.recent.length > this.options.window) this.recent.shift();
    const current = median(this.recent);
    this.estimate[this.path] = current;
    if (this.reference === null) {
      this.reference = current;
      return;
    }
    // Moved by more than the last probe's gap: the order may have flipped.
    const change = Math.max(current / this.reference, this.reference / current);
    if (
      !this.probing &&
      this.recent.length >= this.options.window &&
      change > Math.min(Math.max(this.gap, 1.25), 2)
    ) {
      this.reference = current;
      this.interval = this.options.baseInterval;
      this.nextProbe = Math.min(
        this.nextProbe,
        this.frame + this.options.baseInterval,
      );
    }
  }

  private finishProbe() {
    const samples = this.probeSamples;
    this.probeUntil = -1;
    this.probeLeft = 0;
    this.probeSamples = [];
    const current = this.estimate[this.path];
    if (samples.length === 0 || current === null) {
      // Timings lost: try again later, backing off as the probe may be slow.
      this.interval = Math.min(this.interval * 2, this.options.maxInterval);
      this.nextProbe = this.frame + this.interval;
      return;
    }
    const probe = median(samples);
    const { margin, marginMs, baseInterval, maxInterval } = this.options;
    this.estimate[other(this.path)] = probe;
    if (probe < current * (1 - margin) && probe < current - marginMs) {
      this.path = other(this.path);
      this.switches += 1;
      this.recent = samples.slice(-this.options.window);
      this.reference = probe;
      this.gap = current / probe;
      this.interval = baseInterval;
    } else {
      this.reference = current;
      this.gap = Math.max(1, probe / current);
      this.interval = Math.min(this.interval * 2, maxInterval);
    }
    this.nextProbe = this.frame + this.interval;
  }

  state(): AutoRasterizerState {
    return {
      path: this.path,
      hardwareMs: this.estimate.hardware,
      tilesMs: this.estimate.tiles,
      probing: this.probing,
      probes: this.probes,
      switches: this.switches,
      interval: this.interval,
    };
  }
}

/** A batch's frames on one path: frame number, span ms, end time (ns). */
export interface TimedFrame {
  frame: number;
  ms: number;
  end: number;
}

/**
 * A batch of frames drawn on one path as one timing: the median span, or
 * the mean period between the draws' ends of consecutive frames when less.
 * Saturated, the GPU overlaps frames, so spans stretch (3-4x on an Apple
 * GPU in an unthrottled loop) while the period is what a frame costs;
 * paced (vsync), the period is the frame interval and the span what the
 * draw costs.
 */
export function batchSample(frames: readonly TimedFrame[]): number | null {
  if (frames.length === 0) return null;
  const sorted = [...frames].sort((a, b) => a.frame - b.frame);
  const span = median(sorted.map((f) => f.ms));
  // The longest run of consecutive frames.
  let best = { first: 0, last: 0 };
  let first = 0;
  for (let i = 1; i <= sorted.length; i++) {
    if (i < sorted.length && sorted[i].frame === sorted[i - 1].frame + 1) {
      continue;
    }
    if (i - 1 - first > best.last - best.first) best = { first, last: i - 1 };
    first = i;
  }
  if (best.last === best.first) return span;
  const a = sorted[best.first];
  const b = sorted[best.last];
  const period = (b.end - a.end) / 1e6 / (b.frame - a.frame);
  return period > 0 ? Math.min(span, period) : span;
}

/**
 * Times rasterizer "auto"'s draws with a GpuProfiler: from the end of a
 * marker pass to the end of the draw's last pass. A pass's beginning can be
 * stamped before the passes ahead of it finish (render passes on Apple
 * GPUs), its end cannot; an empty pass, or one stamping only its end, gets
 * no timestamps (Dawn on Metal), so the marker dispatches a no-op and both
 * passes stamp both ends.
 * Frames are read back in batches of `batch` on one path, which keeps
 * consecutive frames together (for batchSample) and fewer readbacks in
 * flight: an unthrottled loop delays mapAsync by hundreds of frames. The
 * current path is timed one batch in `period` frames, probes every frame:
 * timestamps cost a little.
 */
export class DrawTimer {
  private frame = 0;
  private batchPath: RasterPath | null = null;
  private count = 0;
  private flushed: (() => void)[] = [];
  private mark?: GPUComputePipeline;

  constructor(
    readonly profiler: GpuProfiler,
    onSample: (path: RasterPath, ms: number) => void,
    readonly batch = 4,
    readonly period = 16,
  ) {
    profiler.onFrame = (_frame, ends) => {
      const frames: TimedFrame[] = [];
      let path: RasterPath | null = null;
      for (const [label, end] of Object.entries(ends)) {
        const [p, n, start] = label.split(":");
        const from = ends[`${label}:start`];
        if (n === undefined || start || from === undefined) continue;
        path = p as RasterPath;
        frames.push({ frame: Number(n), ms: (end - from) / 1e6, end });
      }
      const sample = batchSample(frames);
      if (path && sample !== null) onSample(path, sample);
    };
  }

  /**
   * Starts timing a frame's draw on `path` in `encoder` (if it is a probe's,
   * or its turn): records the marker and returns the timestamp writes for
   * the draw's last pass (undefined when untimed, or no readback is free).
   * skip() instead for a frame that cannot be timed.
   */
  begin(
    encoder: GPUCommandEncoder,
    path: RasterPath,
    probe: boolean,
  ): GPURenderPassTimestampWrites | undefined {
    if (this.count > 0 && path !== this.batchPath) this.flush(encoder);
    this.batchPath = path;
    const n = this.frame++;
    if (!probe && n % this.period >= this.batch) return undefined;
    const label = `${path}:${n}`;
    const start = this.profiler.timestampWrites(`${label}:start`);
    const end = start && this.profiler.timestampWrites(label);
    if (!end) return undefined;
    this.count += 1;
    const { device } = this.profiler;
    this.mark ??= device.createComputePipeline({
      label: "draw start",
      layout: "auto",
      compute: {
        module: device.createShaderModule({
          code: "@compute @workgroup_size(1) fn main() {}",
        }),
        entryPoint: "main",
      },
    });
    const pass = encoder.beginComputePass({
      label: "draw start",
      timestampWrites: start,
    });
    pass.setPipeline(this.mark);
    pass.dispatchWorkgroups(1);
    pass.end();
    return end;
  }

  skip() {
    this.frame += 1;
  }

  /** Before submitting `encoder`; returns what to call after. */
  resolve(encoder: GPUCommandEncoder): () => void {
    if (this.count >= this.batch) this.flush(encoder);
    const flushed = this.flushed;
    this.flushed = [];
    return () => {
      for (const f of flushed) f();
    };
  }

  private flush(encoder: GPUCommandEncoder) {
    const after = this.profiler.resolve(encoder);
    if (after) this.flushed.push(after);
    this.count = 0;
  }

  destroy() {
    this.profiler.destroy();
  }
}
