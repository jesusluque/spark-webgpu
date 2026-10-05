// Runs example pages in Safari through safaridriver (plain WebDriver over
// HTTP, no dependencies): a screenshot of each, the console errors and
// WebGPU errors seen after it loaded, and its #info line.
//
//   node tools/safari/run.mjs [--base http://localhost:8083] [--out dir]
//     (default out: $TMPDIR/spark-safari-run)
//                             [--port 4444] [--wait 30] [page.html ...]
//
// Pages default to every examples/webgpu/*.html. A page counts as ready when
// it sets window.__ready, window.__caps.done or window.__sparkFrames; the
// result of caps.html (window.__caps) is saved beside the screenshots.
//
// Needs Safari's "Allow remote automation" (Safari > Settings > Advanced >
// "Show features for web developers", then Developer > "Allow remote
// automation"), or once from a terminal: `safaridriver --enable`. Without it
// the run stops at the first step and says so.
//
// WebDriver has no console log for Safari, so errors are collected by hooks
// installed once the page has loaded: console.error/warn, window errors,
// unhandled rejections and `uncapturederror` on the page's GPUDevice
// (window.__splats.device or window.__fx.gpu.device). Errors thrown while the
// page loads show as a page that never became ready, with its #info text.

import { spawn } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../..",
);

function parseArgs(argv) {
  const opts = {
    base: "http://localhost:8083",
    out: path.join(os.tmpdir(), "spark-safari-run"),
    port: 4444,
    wait: 30,
    pages: [],
  };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--base") opts.base = argv[++i];
    else if (a === "--out") opts.out = argv[++i];
    else if (a === "--port") opts.port = Number(argv[++i]);
    else if (a === "--wait") opts.wait = Number(argv[++i]);
    else opts.pages.push(a);
  }
  if (opts.pages.length === 0) {
    const dir = path.join(ROOT, "examples/webgpu");
    opts.pages = fs
      .readdirSync(dir)
      .filter((f) => f.endsWith(".html"))
      .sort()
      .map((f) => `examples/webgpu/${f}`);
  }
  return opts;
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

class WebDriver {
  constructor(port) {
    this.url = `http://localhost:${port}`;
    this.session = null;
  }

  async call(method, route, body) {
    const res = await fetch(`${this.url}${route}`, {
      method,
      headers: { "content-type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    const json = await res.json().catch(() => ({}));
    if (!res.ok || json.value?.error) {
      const v = json.value ?? {};
      const err = new Error(
        `${method} ${route}: ${v.error ?? res.status} ${v.message ?? ""}`.trim(),
      );
      err.webdriver = v.error;
      err.detail = v.message ?? "";
      throw err;
    }
    return json.value;
  }

  async start() {
    const value = await this.call("POST", "/session", {
      capabilities: { alwaysMatch: { browserName: "safari" } },
    });
    this.session = value.sessionId;
    return value.capabilities;
  }

  s(route) {
    return `/session/${this.session}${route}`;
  }

  navigate(url) {
    return this.call("POST", this.s("/url"), { url });
  }

  exec(script, args = []) {
    return this.call("POST", this.s("/execute/sync"), { script, args });
  }

  async screenshot(file) {
    const b64 = await this.call("GET", this.s("/screenshot"));
    fs.writeFileSync(file, Buffer.from(b64, "base64"));
  }

  setWindow(width, height) {
    return this.call("POST", this.s("/window/rect"), { width, height });
  }

  async stop() {
    if (this.session) await this.call("DELETE", this.s("")).catch(() => {});
    this.session = null;
  }
}

const HOOKS = `
  if (!window.__safariRun) {
    const log = (window.__safariRun = []);
    const text = (a) => a.map((x) => (x instanceof Error ? x.message : typeof x === "object" ? JSON.stringify(x) : String(x))).join(" ");
    for (const level of ["error", "warn"]) {
      const orig = console[level].bind(console);
      console[level] = (...a) => { log.push("[" + level + "] " + text(a)); orig(...a); };
    }
    window.addEventListener("error", (e) => log.push("[pageerror] " + e.message));
    window.addEventListener("unhandledrejection", (e) => log.push("[rejection] " + (e.reason?.message ?? e.reason)));
    const device = window.__splats?.device ?? window.__fx?.gpu?.device;
    device?.addEventListener("uncapturederror", (e) => log.push("[gpu] " + e.error.message));
    device?.lost?.then((i) => log.push("[gpu lost] " + i.message));
  }
  return true;`;

const READY =
  "return !!(window.__ready || window.__caps?.done || (window.__sparkFrames ?? 0) >= 2);";

const STATE = `return {
  info: document.querySelector("#info, #out")?.textContent?.slice(0, 400) ?? null,
  log: window.__safariRun ?? [],
  caps: window.__caps ?? null,
  stats: window.__splats?.stats ?? null,
  webgpu: !!navigator.gpu,
};`;

async function startDriver(port) {
  const driver = spawn("safaridriver", ["-p", String(port)], {
    stdio: ["ignore", "pipe", "pipe"],
  });
  let output = "";
  driver.stdout.on("data", (d) => {
    output += d;
  });
  driver.stderr.on("data", (d) => {
    output += d;
  });
  for (let i = 0; i < 50; i++) {
    if (driver.exitCode !== null) {
      throw new Error(
        `safaridriver exited (${driver.exitCode}): ${output.trim()}`,
      );
    }
    try {
      const res = await fetch(`http://localhost:${port}/status`);
      if (res.ok) return driver;
    } catch {}
    await sleep(100);
  }
  driver.kill();
  throw new Error(
    `safaridriver did not answer on port ${port}: ${output.trim()}`,
  );
}

export async function runSafari(opts) {
  fs.mkdirSync(opts.out, { recursive: true });
  const driver = await startDriver(opts.port);
  const wd = new WebDriver(opts.port);
  const results = [];
  try {
    try {
      const caps = await wd.start();
      console.log(`Safari ${caps.browserVersion} (${caps.platformName})`);
    } catch (e) {
      if (
        /remote automation|Allow Remote Automation|session not created/i.test(
          `${e.message}`,
        )
      ) {
        const err = new Error(
          `Safari refused the WebDriver session: remote automation is not enabled.\nTurn on Safari > Settings > Advanced > "Show features for web developers", then Developer > "Allow remote automation" (or run \`safaridriver --enable\` once).\nsafaridriver said: ${e.detail || e.message}`,
        );
        err.code = "AUTOMATION_DISABLED";
        throw err;
      }
      throw e;
    }
    await wd.setWindow(1024, 720).catch(() => {});
    for (const page of opts.pages) {
      const name = path.basename(page, ".html");
      const url = `${opts.base.replace(/\/$/, "")}/${page.replace(/^\//, "")}`;
      const result = { page, url, ready: false };
      try {
        await wd.navigate(url);
        await wd.exec(HOOKS);
        const t0 = Date.now();
        while (Date.now() - t0 < opts.wait * 1000) {
          if (await wd.exec(READY).catch(() => false)) {
            result.ready = true;
            break;
          }
          await sleep(250);
        }
        await wd.exec(HOOKS); // the device exists by now
        await sleep(1500); // a few more frames, for per-frame errors
        Object.assign(result, await wd.exec(STATE));
        if (result.caps) {
          fs.writeFileSync(
            path.join(opts.out, `${name}.json`),
            JSON.stringify(result.caps, null, 2),
          );
          result.caps = `${name}.json`;
        }
        result.screenshot = `${name}.png`;
        await wd.screenshot(path.join(opts.out, result.screenshot));
      } catch (e) {
        result.error = e.message;
      }
      results.push(result);
      const status = result.error ? "ERROR" : result.ready ? "ok" : "NOT READY";
      console.log(
        `${status.padEnd(9)} ${page}  ${(result.info ?? "").replace(/\s+/g, " ").slice(0, 100)}`,
      );
      for (const line of result.log ?? [])
        console.log(`          ${line.slice(0, 300)}`);
      if (result.error) console.log(`          ${result.error}`);
    }
  } finally {
    await wd.stop();
    driver.kill();
  }
  fs.writeFileSync(
    path.join(opts.out, "results.json"),
    JSON.stringify(results, null, 2),
  );
  return results;
}

if (
  process.argv[1] &&
  path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)
) {
  const opts = parseArgs(process.argv.slice(2));
  runSafari(opts).then(
    (results) => {
      const bad = results.filter(
        (r) => r.error || !r.ready || (r.log ?? []).length,
      );
      console.log(
        `\n${results.length - bad.length}/${results.length} clean; screenshots in ${opts.out}`,
      );
      process.exit(bad.length ? 1 : 0);
    },
    (e) => {
      console.error(e.message);
      process.exit(e.code === "AUTOMATION_DISABLED" ? 2 : 1);
    },
  );
}
