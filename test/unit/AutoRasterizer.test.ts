import { describe, expect, it } from "vitest";
import {
  AutoRasterizer,
  type RasterPath,
  batchSample,
} from "../../src/webgpu/AutoRasterizer";

// Runs `frames` frames: each drawn on the path the policy picks, its time
// from `cost` (plus noise) reported `delay` frames later. Returns the paths.
function simulate(
  auto: AutoRasterizer,
  frames: number,
  cost: (path: RasterPath, frame: number) => number,
  { delay = 3, noise = 0, seed = 1 } = {},
) {
  let s = seed;
  const rnd = () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 2 ** 32;
  };
  const pending: { at: number; path: RasterPath; ms: number }[] = [];
  const paths: RasterPath[] = [];
  for (let f = 0; f < frames; f++) {
    while (pending.length && pending[0].at <= f) {
      const m = pending.shift() as (typeof pending)[0];
      auto.measured(m.path, m.ms);
    }
    const path = auto.next();
    paths.push(path);
    const ms = cost(path, f) * (1 + noise * (2 * rnd() - 1));
    pending.push({ at: f + delay, path, ms });
  }
  return paths;
}

const count = (paths: RasterPath[], p: RasterPath) =>
  paths.filter((x) => x === p).length;

describe("AutoRasterizer", () => {
  it("switches to the tiles where they are clearly faster", () => {
    const auto = new AutoRasterizer();
    const paths = simulate(auto, 200, (p) => (p === "tiles" ? 8 : 40));
    expect(auto.path).toBe("tiles");
    expect(auto.switches).toBe(1);
    expect(paths.at(-1)).toBe("tiles");
    expect(auto.state()).toMatchObject({ path: "tiles", hardwareMs: 40 });
    // The first probe comes after baseInterval frames of hardware.
    expect(paths.indexOf("tiles")).toBe(auto.options.baseInterval - 1);
  });

  it("stays on hardware where it is faster, backing off its probes", () => {
    const auto = new AutoRasterizer();
    const paths = simulate(auto, 4000, (p) => (p === "tiles" ? 3 : 2));
    expect(auto.path).toBe("hardware");
    expect(auto.switches).toBe(0);
    expect(auto.interval).toBe(auto.options.maxInterval);
    // Probes cost a few frames: 30, 60, 120, ... frames apart.
    expect(count(paths, "tiles")).toBeLessThan(40);
  });

  it("keeps the current path within the margin, under noise", () => {
    const auto = new AutoRasterizer();
    const paths = simulate(auto, 5000, (p) => (p === "tiles" ? 9.7 : 10), {
      noise: 0.04,
    });
    expect(auto.switches).toBe(0);
    expect(count(paths, "tiles")).toBeLessThan(50);
  });

  it("switches back when the scene changes", () => {
    const auto = new AutoRasterizer();
    // Dense until frame 2000 (tiles 8, hardware 40), then light (tiles 3,
    // hardware 1.5): the tiles' time moving 2.7x brings a probe back soon.
    const cost = (p: RasterPath, f: number) =>
      f < 2000 ? (p === "tiles" ? 8 : 40) : p === "tiles" ? 3 : 1.5;
    const paths = simulate(auto, 2300, cost);
    expect(paths[1999]).toBe("tiles");
    expect(auto.path).toBe("hardware");
    expect(auto.switches).toBe(2);
    // Hardware probes while on the dense scene: few, backed off.
    expect(count(paths.slice(200, 2000), "hardware")).toBeLessThan(30);
  });

  it("gives up on a probe whose timings never come back", () => {
    const auto = new AutoRasterizer({ probeTimeout: 50 });
    // Only hardware frames are timed.
    const paths: RasterPath[] = [];
    for (let f = 0; f < 400; f++) {
      const path = auto.next();
      paths.push(path);
      if (path === "hardware") auto.measured("hardware", 2);
    }
    expect(auto.path).toBe("hardware");
    expect(auto.probing).toBe(false);
    expect(auto.interval).toBeGreaterThan(auto.options.baseInterval);
    expect(auto.state().tilesMs).toBeNull();
  });

  it("extends a probe by the frames the caller could not draw", () => {
    const auto = new AutoRasterizer({ baseInterval: 2, probeFrames: 2 });
    auto.next();
    auto.measured("hardware", 5);
    expect(auto.next()).toBe("tiles"); // probe starts
    auto.extendProbe(); // that frame went to hardware
    expect([auto.next(), auto.next(), auto.next()]).toEqual([
      "tiles",
      "tiles",
      "hardware",
    ]);
  });
});

describe("batchSample", () => {
  const frames = (spans: number[], periodMs: number, first = 10) =>
    spans.map((ms, i) => ({
      frame: first + i,
      ms,
      end: 1e9 + i * periodMs * 1e6,
    }));

  it("is the median span when frames are paced", () => {
    expect(batchSample(frames([2, 3, 2.5, 9], 16.7))).toBe(2.75);
  });

  it("is the period when overlapping frames stretch the spans", () => {
    expect(batchSample(frames([6, 7, 6, 8], 2))).toBeCloseTo(2);
  });

  it("uses only consecutive frames for the period", () => {
    const f = frames([5, 5, 5], 1);
    f[2] = { frame: 20, ms: 5, end: 2e9 };
    expect(batchSample(f)).toBeCloseTo(1);
    expect(batchSample([f[0]])).toBe(5);
    expect(batchSample([])).toBeNull();
  });
});
