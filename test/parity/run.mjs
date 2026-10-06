// WebGPU-vs-WebGL parity runner for the examples (see README.md).
//
//   node test/parity/run.mjs [filter ...] [options]
//
// A filter keeps the cases whose id ("name" or "name:scenario") contains it.
//   --base <url>        reuse a running server instead of starting Vite
//   --port <n>          port for the runner's Vite (default 8081)
//   --out <dir>         report directory (default test/parity/out)
//   --browser safari    WebGPU in Safari against WebGL in Chrome
//   --webgl <b>, --webgpu <b>   the browser for each side: chrome | safari
//   --perf              also measure fps on both backends (real clock)
//   --perf-only         only measure fps
//   --perf-runs <n>     fps windows per page, reported as the median (3)
//   --perf-seconds <s>  length of each window (2)
//   --retries <n>       reruns of a failing case before it fails (1)
//   --list              print the cases and exit
//   --headed            show Chrome
//   --safari-port <n>   safaridriver's port (4444)

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { ChromeDriver, SafariDriver } from "./drivers.mjs";
import { routes as allRoutes, cases, fixtures } from "./manifest.mjs";
import { decodePng, diffImages, encodePng } from "./png.mjs";
import { writeReport } from "./report.mjs";
import { startVite } from "./server.mjs";

const ROOT = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../..",
);
process.chdir(ROOT);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

export function parseArgs(argv) {
  const o = {
    filters: [],
    base: null,
    port: 8081,
    out: "test/parity/out",
    webgl: "chrome",
    webgpu: "chrome",
    perf: false,
    perfOnly: false,
    perfRuns: 3,
    perfSeconds: 2,
    retries: 1,
    list: false,
    headed: false,
    safariPort: 4444,
  };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const val = () => argv[++i];
    if (a === "--base") o.base = val().replace(/\/$/, "");
    else if (a === "--port") o.port = Number(val());
    else if (a === "--out") o.out = val();
    else if (a === "--browser") {
      const b = val();
      if (b === "safari") o.webgpu = "safari";
      else if (b !== "chrome") throw new Error(`unknown browser ${b}`);
    } else if (a === "--webgl") o.webgl = val();
    else if (a === "--webgpu") o.webgpu = val();
    else if (a === "--perf") o.perf = true;
    else if (a === "--perf-only") o.perf = o.perfOnly = true;
    else if (a === "--perf-runs") o.perfRuns = Number(val());
    else if (a === "--perf-seconds") o.perfSeconds = Number(val());
    else if (a === "--retries") o.retries = Number(val());
    else if (a === "--list") o.list = true;
    else if (a === "--headed") o.headed = true;
    else if (a === "--safari-port") o.safariPort = Number(val());
    else if (a.startsWith("--")) throw new Error(`unknown option ${a}`);
    else o.filters.push(a);
  }
  for (const side of [o.webgl, o.webgpu]) {
    if (side !== "chrome" && side !== "safari") {
      throw new Error(`unknown browser ${side}`);
    }
  }
  return o;
}

const median = (xs) => {
  const s = [...xs].sort((a, b) => a - b);
  return s.length ? s[Math.floor((s.length - 1) / 2)] : null;
};

/**
 * Waits for a page to settle: its `ready` condition if any, `minWait` ms,
 * then (unless `stable: false`) until two screenshots in a row match and no
 * resource has started or finished loading in between. A frozen clock makes
 * a loaded scene draw the same frame every time, so this catches late
 * splats, LoD refinement and paged streaming without per-page timings.
 * Gives up after `maxWait`, and says so in the report.
 */
async function settle(h, s = {}) {
  const t0 = Date.now();
  const minWait = s.minWait ?? 2000;
  const maxWait = s.maxWait ?? 60000;
  if (s.ready) await h.ctx.waitFor(s.ready, null, maxWait);
  const resources = () =>
    h
      .evaluate(() => performance.getEntriesByType("resource").length)
      .catch(() => -1);
  await sleep(Math.max(0, minWait - (Date.now() - t0)));
  let prev = null;
  let prevRes = await resources();
  let calm = 0;
  const need = s.stableShots ?? 2;
  while (Date.now() - t0 < maxWait) {
    await sleep(s.interval ?? 500);
    const res = await resources();
    const quiet = res === prevRes && h.inflight() <= 0;
    prevRes = res;
    let same = true;
    if (s.stable !== false) {
      const img = decodePng(await h.screenshot());
      same = prev !== null && diffImages(prev, img).mean < (s.epsilon ?? 0.02);
      prev = img;
    }
    calm = quiet && same ? calm + 1 : 0;
    if (calm >= need) return { settled: true, ms: Date.now() - t0 };
  }
  return { settled: false, ms: Date.now() - t0 };
}

function pageUrl(base, item, side) {
  const p = side === "webgl" ? item.webgl : item.webgpu;
  return `${base}${p}`;
}

// lil-gui panels, and stats.js's (a fixed div at z-index 10000).
const PANELS = [".lil-gui.root", 'body > div[style*="z-index: 10000"]'];

/** A fixture file's local path, downloaded into <out>/fixtures once. */
async function fixture(ctx, name) {
  const file = path.join(ctx.out, "fixtures", name);
  if (!fs.existsSync(file)) {
    const url = fixtures[name];
    if (!url) throw new Error(`no fixture ${name}`);
    const res = await fetch(url);
    if (!res.ok) throw new Error(`fixture ${name}: ${res.status} from ${url}`);
    fs.mkdirSync(path.dirname(file), { recursive: true });
    fs.writeFileSync(file, Buffer.from(await res.arrayBuffer()));
  }
  return file;
}

async function capture(ctx, item, side) {
  const driver = ctx.drivers[side];
  const sc = item;
  const h = await driver.open({
    url: pageUrl(ctx.base, item, side),
    base: ctx.base,
    viewport: sc.viewport,
    dpr: ctx.dpr,
    init: {
      freeze: sc.freeze ?? true,
      t0: sc.t0 ?? 1000,
      seed: sc.seed ?? 1234,
      manual: !!sc.manual,
    },
    routes: ctx.ownServer ? [] : (sc.routes ?? []),
    stubClient: !ctx.ownServer,
  });
  h.ctx.fixture = (name) => fixture(ctx, name);
  h.ctx.settle = (opts) => settle(h, { minWait: 0, ...opts });
  try {
    // Panels that are the same on both sides (lil-gui, stats.js) are hidden
    // so the lit check sees the scene, and so is text that differs by design
    // (frame counters, timings).
    const hide = [...(sc.showPanels ? [] : PANELS), ...(sc.hide ?? [])];
    if (hide.length) {
      await h.evaluate((sel) => {
        const style = document.createElement("style");
        style.textContent = `${sel.join(",")} { visibility: hidden !important; }`;
        document.head.appendChild(style);
      }, hide);
    }
    const settled = await settle(h, sc.settle);
    if (sc.act) await sc.act(h.ctx, side);
    await sleep(sc.after ?? 300);
    const png = await h.screenshot();
    const logs = h.logs();
    return { png, logs, settled };
  } finally {
    await h.close().catch(() => {});
  }
}

function skipReason(ctx, item) {
  for (const side of ["webgl", "webgpu"]) {
    const d = ctx.drivers[side];
    if ((item.needs ?? []).includes("files") && !d.caps.files) {
      return `${d.name} can't set file inputs`;
    }
    if (item.routes?.length && !ctx.ownServer && !d.caps.routes) {
      return `${d.name} needs the runner's Vite for asset routes`;
    }
    if (d.name === "safari" && !ctx.ownServer) {
      return "Safari needs the runner's Vite (no --base) for the frozen clock";
    }
    if (item.skip?.[d.name]) return item.skip[d.name];
  }
  return null;
}

function judge(item, d, shots) {
  const max = item.max ?? {};
  const reasons = [];
  if (max.mean != null && d.mean > max.mean) {
    reasons.push(`mean ${d.mean.toFixed(2)} > ${max.mean}`);
  }
  if (max.pct != null && d.pctOver > max.pct) {
    reasons.push(`>16: ${d.pctOver.toFixed(2)}% > ${max.pct}%`);
  }
  const minLit = item.minLit ?? 1;
  if (d.litA < minLit)
    reasons.push(`WebGL page nearly blank (${d.litA.toFixed(1)}% lit)`);
  if (d.litB < minLit)
    reasons.push(`WebGPU page nearly blank (${d.litB.toFixed(1)}% lit)`);
  // Uncaught errors that only the port throws are regressions even when the
  // picture still matches.
  const errs = (s) =>
    new Set(s.logs.filter((l) => l.startsWith("[pageerror]")));
  const glErrs = errs(shots.webgl);
  const newErrs = [...errs(shots.webgpu)].filter((l) => !glErrs.has(l));
  if (newErrs.length && !item.allowErrors) {
    reasons.push(`WebGPU page errors: ${newErrs[0].slice(0, 160)}`);
  }
  return reasons;
}

async function runCase(ctx, item) {
  const row = {
    id: item.id,
    name: item.name,
    scenario: item.scenario,
    readme: item.readme ?? null,
    max: item.max ?? null,
    webglUrl: item.webgl,
    webgpuUrl: item.webgpu,
    browsers: {
      webgl: ctx.drivers.webgl.name,
      webgpu: ctx.drivers.webgpu.name,
    },
  };
  const skip = skipReason(ctx, item);
  if (skip) return { ...row, status: "skip", reasons: [skip] };
  let attempt = 0;
  const failures = [];
  for (;;) {
    attempt++;
    const t0 = Date.now();
    try {
      const shots = {};
      for (const side of ["webgl", "webgpu"]) {
        shots[side] = await capture(ctx, item, side);
      }
      const d = diffImages(
        decodePng(shots.webgl.png),
        decodePng(shots.webgpu.png),
      );
      const file = item.id.replace(/[^a-zA-Z0-9_.-]+/g, "_");
      fs.writeFileSync(path.join(ctx.out, `${file}-webgl.png`), encodePng(d.a));
      fs.writeFileSync(
        path.join(ctx.out, `${file}-webgpu.png`),
        encodePng(d.b),
      );
      fs.writeFileSync(
        path.join(ctx.out, `${file}-diff.png`),
        encodePng(d.diff),
      );
      const reasons = judge(item, d, shots);
      const result = {
        ...row,
        status: reasons.length
          ? item.known
            ? "known"
            : "fail"
          : attempt > 1
            ? "flaky"
            : "pass",
        known: item.known,
        reasons,
        mean: d.mean,
        pctOver: d.pctOver,
        lit: { webgl: d.litA, webgpu: d.litB },
        size: [d.width, d.height],
        settled: { webgl: shots.webgl.settled, webgpu: shots.webgpu.settled },
        logs: {
          webgl: shots.webgl.logs.slice(0, 20),
          webgpu: shots.webgpu.logs.slice(0, 20),
        },
        images: {
          webgl: `${file}-webgl.png`,
          webgpu: `${file}-webgpu.png`,
          diff: `${file}-diff.png`,
        },
        seconds: (Date.now() - t0) / 1000,
        attempts: attempt,
        earlier: failures,
      };
      if (!reasons.length || item.known || attempt > ctx.retries) return result;
      failures.push({ mean: d.mean, pctOver: d.pctOver, reasons });
    } catch (e) {
      if (attempt > ctx.retries) {
        return {
          ...row,
          status: "error",
          reasons: [e.message.split("\n")[0]],
          attempts: attempt,
          earlier: failures,
        };
      }
      failures.push({ reasons: [e.message.split("\n")[0]] });
    }
  }
}

/** fps of one page: medians of `runs` windows of real-clock frames. */
async function measureFps(ctx, driver, url, item) {
  const h = await driver.open({
    url,
    base: ctx.base,
    viewport: item.viewport,
    dpr: ctx.dpr,
    init: { freeze: false, seed: item.seed ?? 1234 },
    routes: ctx.ownServer ? [] : (item.routes ?? []),
    stubClient: !ctx.ownServer,
  });
  try {
    await settle(h, {
      ...item.settle,
      stable: false,
      minWait: (item.settle?.minWait ?? 2000) + 2000,
    });
    const runs = [];
    for (let i = 0; i < ctx.perfRuns; i++) {
      const f0 = await h.evaluate(() => [window.__frames, performance.now()]);
      await sleep(ctx.perfSeconds * 1000);
      const f1 = await h.evaluate(() => [window.__frames, performance.now()]);
      runs.push(((f1[0] - f0[0]) * 1000) / (f1[1] - f0[1]));
    }
    return { fps: median(runs), runs };
  } finally {
    await h.close().catch(() => {});
  }
}

async function runPerf(ctx, items) {
  // One measurement per example page (its first scenario's URLs): input
  // scripts assume the frozen clock.
  const seen = new Set();
  const rows = [];
  for (const item of items) {
    if (seen.has(item.name)) continue;
    seen.add(item.name);
    if (skipReason(ctx, item)) continue;
    const row = { name: item.name, id: item.id };
    for (const side of ["webgl", "webgpu"]) {
      try {
        row[side] = await measureFps(
          ctx,
          ctx.perfDrivers[side],
          pageUrl(ctx.base, item, side),
          item,
        );
      } catch (e) {
        row[side] = { error: e.message.split("\n")[0] };
      }
    }
    const f = (r) =>
      r.error ? `ERR ${r.error.slice(0, 40)}` : `${r.fps.toFixed(0)} fps`;
    console.log(
      `  perf ${item.name.padEnd(28)} webgl ${f(row.webgl).padEnd(12)} webgpu ${f(row.webgpu)}`,
    );
    rows.push(row);
  }
  return rows;
}

function matches(filters, item) {
  return !filters.length || filters.some((f) => item.id.includes(f));
}

async function launch(name, opts, perf = false) {
  if (name === "chrome")
    return ChromeDriver.launch({ perf, headed: opts.headed });
  return SafariDriver.launch({ port: opts.safariPort });
}

export async function main(argv) {
  const opts = parseArgs(argv);
  const items = cases.filter((c) => matches(opts.filters, c));
  if (opts.list) {
    for (const c of items) console.log(`${c.id.padEnd(44)} ${c.webgpu}`);
    return 0;
  }
  if (!items.length) {
    console.error("no case matches", opts.filters.join(" "));
    return 1;
  }
  const out = path.resolve(opts.out);
  fs.mkdirSync(out, { recursive: true });
  const ctx = {
    out,
    retries: opts.retries,
    perfRuns: opts.perfRuns,
    perfSeconds: opts.perfSeconds,
    dpr: 1,
  };
  const closers = [];
  try {
    // Safari first: when it can't run there is nothing else to do.
    const drivers = {};
    for (const side of ["webgpu", "webgl"]) {
      const name = opts[side];
      if (name === "safari" && !drivers.safari) {
        try {
          drivers.safari = await SafariDriver.launch({ port: opts.safariPort });
          closers.push(() => drivers.safari.close());
        } catch (e) {
          if (e.code !== "SAFARI_UNAVAILABLE") throw e;
          console.log(`SKIPPED: Safari is not available.\n${e.message}`);
          writeReport(out, {
            skipped: e.message,
            results: [],
            perf: null,
            opts,
          });
          return 0;
        }
      }
    }
    if (!opts.perfOnly && [opts.webgl, opts.webgpu].includes("chrome")) {
      drivers.chrome = await launch("chrome", opts);
      closers.push(() => drivers.chrome.close());
    }
    ctx.drivers = { webgl: drivers[opts.webgl], webgpu: drivers[opts.webgpu] };
    if (drivers.safari) {
      // Compare at Safari's pixel ratio: Chrome renders at the same one.
      ctx.dpr = (await drivers.safari.wd.setViewport(800, 600))[2] ?? 1;
    }
    let server = null;
    if (opts.base) {
      ctx.base = opts.base;
      ctx.ownServer = false;
    } else {
      server = await startVite({ port: opts.port, routes: allRoutes });
      closers.push(() => server.close());
      ctx.base = server.base;
      ctx.ownServer = true;
    }
    const versions = Object.values(drivers).map((d) => d.version());
    console.log(
      `parity: ${items.length} cases, ${[...versions, ctx.base].join(", ")}${opts.perf ? ", perf" : ""}`,
    );
    const results = [];
    const t0 = Date.now();
    if (!opts.perfOnly) {
      for (const item of items) {
        const r = await runCase(ctx, item);
        results.push(r);
        const num =
          r.mean != null
            ? `${r.mean.toFixed(2)} (max ${r.max?.mean ?? "-"})  >16 ${r.pctOver.toFixed(2)}%`
            : "";
        console.log(
          `${r.status.toUpperCase().padEnd(5)} ${r.id.padEnd(44)} ${num}  ${r.reasons?.join("; ") ?? ""}`,
        );
        writeReport(out, {
          results,
          perf: null,
          opts,
          versions,
          base: ctx.base,
        });
      }
    }
    let perf = null;
    if (opts.perf) {
      ctx.perfDrivers = {};
      for (const side of ["webgl", "webgpu"]) {
        const name = opts[side];
        if (name === "chrome") {
          if (!drivers.chromePerf) {
            drivers.chromePerf = await ChromeDriver.launch({
              perf: true,
              headed: opts.headed,
            });
            closers.push(() => drivers.chromePerf.close());
          }
          ctx.perfDrivers[side] = drivers.chromePerf;
        } else {
          ctx.perfDrivers[side] = drivers.safari;
        }
      }
      ctx.drivers = ctx.perfDrivers;
      perf = await runPerf(ctx, items);
    }
    writeReport(out, {
      results,
      perf,
      opts,
      versions: [...new Set(Object.values(drivers).map((d) => d.version()))],
      base: ctx.base,
      seconds: (Date.now() - t0) / 1000,
    });
    const bad = results.filter(
      (r) => r.status === "fail" || r.status === "error",
    );
    const count = (s) => results.filter((r) => r.status === s).length;
    console.log(
      `\n${count("pass")} pass, ${count("flaky")} flaky, ${count("known")} known, ${bad.length} fail/error, ${count("skip")} skipped in ${((Date.now() - t0) / 1000).toFixed(0)} s; report: ${path.relative(ROOT, path.join(out, "index.html"))}`,
    );
    return bad.length ? 1 : 0;
  } finally {
    for (const c of closers.reverse()) await c().catch(() => {});
  }
}

if (
  process.argv[1] &&
  path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)
) {
  main(process.argv.slice(2)).then(
    (code) => process.exit(code),
    (e) => {
      console.error(e);
      process.exit(1);
    },
  );
}
