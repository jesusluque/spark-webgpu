// Browser drivers behind one small interface, so the manifest's input scripts
// run unchanged on Chrome (Playwright) and Safari (WebDriver).
//
// driver.open({ url, viewport, init, routes }) returns a page handle:
//   handle.ctx            what input scripts get (see makeCtx)
//   handle.evaluate(fn, arg), handle.screenshot() -> PNG Buffer
//   handle.logs()         console errors/warnings and page errors so far
//   handle.inflight()     requests not finished yet (Chrome only; else 0)
//   handle.close()

import { parityInit } from "./clock.mjs";
import { VITE_CLIENT_STUB } from "./clock.mjs";
import {
  WebDriver,
  isAutomationDisabled,
  startSafariDriver,
} from "./webdriver.mjs";

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// Noise both backends print, or that says nothing about rendering.
const IGNORED = [
  /404/,
  /\[vite\]/,
  /GPU stall due to ReadPixels/,
  /favicon/,
  /DevTools/,
  /Download the React DevTools/,
];
const keep = (line) => !IGNORED.some((r) => r.test(line));

/**
 * The input-script API. Coordinates are viewport CSS pixels; `step(n, dt)`
 * runs n frames of the frozen clock, dt ms apart (see clock.mjs).
 */
function makeCtx(h, extra) {
  return {
    step: (n = 1, dt = 1000 / 60) =>
      h.evaluate(([n, dt]) => window.__step(n, dt), [n, dt]),
    setTime: (t) => h.evaluate((t) => window.__setTime(t), t),
    wait: (ms) => sleep(ms),
    eval: (fn, arg) => h.evaluate(fn, arg),
    async waitFor(fn, arg, timeout = 60000) {
      const t0 = Date.now();
      while (!(await h.evaluate(fn, arg).catch(() => false))) {
        if (Date.now() - t0 > timeout) {
          throw new Error(
            `timed out waiting for ${fn.toString().slice(0, 80)}`,
          );
        }
        await sleep(200);
      }
    },
    // DOM helpers that work the same on any driver: they act through the
    // page's own event handlers, as lil-gui and the examples' controls do.
    clickText: (text) =>
      h.evaluate((text) => {
        const xp = `//*[normalize-space(text())=${JSON.stringify(text)}]`;
        const el = document.evaluate(
          xp,
          document,
          null,
          9,
          null,
        ).singleNodeValue;
        if (!el) throw new Error(`no element with text ${text}`);
        el.click();
      }, text),
    select: (selector, label) =>
      h.evaluate(
        ([selector, label]) => {
          const sel = [...document.querySelectorAll(selector)].find((s) =>
            [...s.options].some((o) => o.text === label),
          );
          if (!sel) throw new Error(`no ${selector} with option ${label}`);
          sel.value = [...sel.options].find((o) => o.text === label).value;
          sel.dispatchEvent(new Event("input", { bubbles: true }));
          sel.dispatchEvent(new Event("change", { bubbles: true }));
        },
        [selector, label],
      ),
    click: (selector) =>
      h.evaluate((selector) => {
        const el = document.querySelector(selector);
        if (!el) throw new Error(`no element ${selector}`);
        el.click();
      }, selector),
    ...extra,
  };
}

export class ChromeDriver {
  constructor(browser, opts) {
    this.browser = browser;
    this.opts = opts;
    this.name = "chrome";
    this.caps = { files: true, routes: true };
  }

  static async launch({ perf = false, headed = false } = {}) {
    const { chromium } = await import("playwright");
    const args = ["--enable-unsafe-webgpu"];
    if (perf) args.push("--disable-gpu-vsync", "--disable-frame-rate-limit");
    const browser = await chromium.launch({
      channel: "chrome",
      headless: !headed,
      args,
    });
    return new ChromeDriver(browser, { perf });
  }

  version() {
    return `Chrome ${this.browser.version()}`;
  }

  async open({ url, viewport, dpr = 1, init, routes = [], stubClient }) {
    const page = await this.browser.newPage({
      viewport,
      deviceScaleFactor: dpr,
    });
    const logs = [];
    let inflight = 0;
    page.on("console", (m) => {
      if (m.type() === "error" || m.type() === "warning") {
        const line = `[${m.type() === "warning" ? "warn" : "error"}] ${m.text()}`;
        if (keep(line)) logs.push(line);
      }
    });
    page.on("pageerror", (e) => logs.push(`[pageerror] ${e.message}`));
    page.on("request", () => inflight++);
    page.on("requestfinished", () => inflight--);
    page.on("requestfailed", () => inflight--);
    if (stubClient) {
      await page.route("**/@vite/client", (r) =>
        r.fulfill({
          contentType: "application/javascript",
          body: VITE_CLIENT_STUB,
        }),
      );
    }
    for (const route of routes) {
      await page.route(
        (u) => route.path.test(u.pathname),
        async (r) => {
          const u = new URL(r.request().url());
          if (route.redirect) {
            await r.fulfill({
              status: 302,
              headers: {
                location: route.redirect(u.pathname),
                "access-control-allow-origin": "*",
              },
            });
          } else {
            const res = await r.fetch();
            await r.fulfill({
              response: res,
              body: route.rewrite(await res.text()),
            });
          }
        },
      );
    }
    if (init) await page.addInitScript(parityInit, init);
    await page.goto(url, { timeout: 120000 });
    const h = {
      page,
      evaluate: (fn, arg) => page.evaluate(fn, arg),
      // Long: a page streaming LoD under a busy GPU can take a while to
      // present a frame.
      screenshot: () => page.screenshot({ timeout: 120000 }),
      logs: () => logs.filter(keep),
      inflight: () => inflight,
      close: () => page.close(),
    };
    h.ctx = makeCtx(h, {
      browser: "chrome",
      mouse: {
        move: (x, y, steps = 1) => page.mouse.move(x, y, { steps }),
        down: () => page.mouse.down(),
        up: () => page.mouse.up(),
        click: (x, y) => page.mouse.click(x, y),
      },
      key: {
        press: (k) => page.keyboard.press(k),
        down: (k) => page.keyboard.down(k),
        up: (k) => page.keyboard.up(k),
      },
      setFiles: (selector, file) => page.setInputFiles(selector, file),
    });
    return h;
  }

  close() {
    return this.browser.close();
  }
}

/**
 * Safari through safaridriver. One session (Safari allows one at a time),
 * reused for every page. Init scripts don't exist in WebDriver, so the
 * frozen clock comes from the runner's Vite plugin, which runs it at the top
 * of every page when sessionStorage.__parity is set (set here from a blank
 * same-origin page before each navigation): Safari needs the runner's own
 * Vite, not --base.
 */
export class SafariDriver {
  constructor(wd, proc, caps) {
    this.wd = wd;
    this.proc = proc;
    this.name = "safari";
    this.caps = { files: false, routes: false };
    this.browserVersion = caps?.browserVersion;
    this.viewport = null;
  }

  /** Throws an error with code SAFARI_UNAVAILABLE when there's no session. */
  static async launch({ port = 4444, bin = "safaridriver" } = {}) {
    let proc;
    try {
      proc = await startSafariDriver(port, bin);
    } catch (e) {
      const err = new Error(`safaridriver could not start: ${e.message}`);
      err.code = "SAFARI_UNAVAILABLE";
      throw err;
    }
    const wd = new WebDriver(`http://localhost:${port}`);
    try {
      const caps = await wd.start({ browserName: "safari" });
      return new SafariDriver(wd, proc, caps);
    } catch (e) {
      proc.kill();
      const err = new Error(
        isAutomationDisabled(e)
          ? `Safari refused the WebDriver session (remote automation is off).\nEnable Safari > Settings > Advanced > "Show features for web developers", then Develop > "Allow Remote Automation" (or run \`safaridriver --enable\` once).\nsafaridriver said: ${e.detail || e.message}`
          : `safaridriver could not create a session: ${e.message}`,
      );
      err.code = "SAFARI_UNAVAILABLE";
      throw err;
    }
  }

  version() {
    return `Safari ${this.browserVersion ?? "?"}`;
  }

  async open({ url, viewport, init, base }) {
    const wd = this.wd;
    if (
      !this.viewport ||
      this.viewport[0] !== viewport.width ||
      this.viewport[1] !== viewport.height
    ) {
      this.viewport = await wd.setViewport(viewport.width, viewport.height);
    }
    await wd.navigate(`${base}/__parity/blank`);
    await wd.exec(
      init
        ? "sessionStorage.setItem('__parity', arguments[0]); return true;"
        : "sessionStorage.removeItem('__parity'); return true;",
      [JSON.stringify({ ...init, log: true })],
    );
    await wd.navigate(url);
    const h = {
      evaluate: (fn, arg) => wd.evaluate(fn, arg),
      screenshot: () => wd.screenshot(),
      logs: () => [],
      inflight: () => 0,
      close: async () => {
        await wd.releaseActions().catch(() => {});
        await wd.navigate("about:blank").catch(() => {});
      },
    };
    let lastLog = [];
    h.logs = () => lastLog.filter(keep);
    const refreshLogs = async () => {
      lastLog =
        (await wd
          .exec("return window.__parityLog || [];")
          .catch(() => lastLog)) ?? [];
    };
    const close = h.close;
    h.close = async () => {
      await refreshLogs();
      await close();
    };
    const shot = h.screenshot;
    h.screenshot = async () => {
      await refreshLogs();
      return shot();
    };
    h.ctx = makeCtx(h, {
      browser: "safari",
      mouse: {
        move: (x, y, steps = 1) => wd.mouseMove(x, y, steps),
        down: () => wd.mouseDown(),
        up: () => wd.mouseUp(),
        click: (x, y) => wd.click(x, y),
      },
      key: {
        press: (k) => wd.press(k),
        down: (k) => wd.keyDown(k),
        up: (k) => wd.keyUp(k),
      },
      setFiles: () => {
        throw new Error("file inputs are not supported on Safari");
      },
    });
    return h;
  }

  async close() {
    await this.wd.stop();
    this.proc.kill();
  }
}
