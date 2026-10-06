// A small W3C WebDriver client over fetch, for safaridriver (Safari has no
// Playwright or CDP). Same protocol as tools/safari/run.mjs, plus input
// actions, async scripts and viewport sizing for the parity runner.

import { spawn } from "node:child_process";

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

export class WebDriverError extends Error {
  constructor(message, { code, detail, status } = {}) {
    super(message);
    this.webdriver = code;
    this.detail = detail ?? "";
    this.status = status;
  }
}

// WebDriver key codes for the non-character keys input scripts use.
export const KEYS = {
  Enter: "",
  Escape: "",
  Space: " ",
  Tab: "",
  Backspace: "",
  Delete: "",
  Shift: "",
  Control: "",
  Alt: "",
  Meta: "",
  ArrowLeft: "",
  ArrowUp: "",
  ArrowRight: "",
  ArrowDown: "",
};

export function keyValue(key) {
  if (KEYS[key]) return KEYS[key];
  if ([...key].length === 1) return key;
  throw new Error(`no WebDriver key for "${key}"`);
}

export class WebDriver {
  constructor(url) {
    this.url = url.replace(/\/$/, "");
    this.session = null;
    this.mousePos = { x: 0, y: 0 };
  }

  async call(method, route, body) {
    let res;
    try {
      res = await fetch(`${this.url}${route}`, {
        method,
        headers: { "content-type": "application/json" },
        body: body === undefined ? undefined : JSON.stringify(body),
      });
    } catch (e) {
      throw new WebDriverError(`${method} ${route}: ${e.message}`, {
        code: "connection refused",
      });
    }
    const json = await res.json().catch(() => ({}));
    if (!res.ok || json.value?.error) {
      const v = json.value ?? {};
      throw new WebDriverError(
        `${method} ${route}: ${v.error ?? res.status} ${v.message ?? ""}`.trim(),
        { code: v.error, detail: v.message, status: res.status },
      );
    }
    return json.value;
  }

  async status() {
    return this.call("GET", "/status");
  }

  /** Creates a session; returns its capabilities. */
  async start(capabilities = { browserName: "safari" }) {
    const value = await this.call("POST", "/session", {
      capabilities: { alwaysMatch: capabilities },
    });
    this.session = value.sessionId;
    await this.call("POST", this.s("/timeouts"), {
      script: 300000,
      pageLoad: 120000,
    }).catch(() => {});
    return value.capabilities;
  }

  s(route) {
    if (!this.session) throw new Error("no WebDriver session");
    return `/session/${this.session}${route}`;
  }

  navigate(url) {
    return this.call("POST", this.s("/url"), { url });
  }

  exec(script, args = []) {
    return this.call("POST", this.s("/execute/sync"), { script, args });
  }

  /**
   * Runs `fn(arg)` in the page and returns its (awaited) result. Uses
   * execute/async so a returned promise is awaited on every driver; errors
   * come back as values and are rethrown here.
   */
  async evaluate(fn, arg) {
    const script = `const done = arguments[arguments.length - 1];
      Promise.resolve().then(() => (${fn.toString()})(arguments[0])).then(
        (v) => done({ ok: v === undefined ? null : v }),
        (e) => done({ err: String((e && e.stack) || e) }));`;
    const r = await this.call("POST", this.s("/execute/async"), {
      script,
      args: [arg === undefined ? null : arg],
    });
    if (r && "err" in r) throw new Error(`page script failed: ${r.err}`);
    return r?.ok;
  }

  /** A PNG of the viewport, as a Buffer. */
  async screenshot() {
    const b64 = await this.call("GET", this.s("/screenshot"));
    return Buffer.from(b64, "base64");
  }

  setWindow(width, height) {
    return this.call("POST", this.s("/window/rect"), { width, height });
  }

  /**
   * Sizes the window so the viewport is width x height CSS pixels: the
   * window rect includes the toolbar, so measure and correct once.
   */
  async setViewport(width, height) {
    await this.setWindow(width, height);
    const inner = await this.exec(
      "return [window.innerWidth, window.innerHeight];",
    );
    if (inner[0] !== width || inner[1] !== height) {
      await this.setWindow(
        width + (width - inner[0]),
        height + (height - inner[1]),
      );
    }
    return this.exec(
      "return [window.innerWidth, window.innerHeight, window.devicePixelRatio];",
    );
  }

  actions(actions) {
    return this.call("POST", this.s("/actions"), { actions });
  }

  releaseActions() {
    return this.call("DELETE", this.s("/actions"));
  }

  pointer(steps) {
    return this.actions([
      {
        type: "pointer",
        id: "mouse",
        parameters: { pointerType: "mouse" },
        actions: steps,
      },
    ]);
  }

  /** Moves the mouse to (x, y) in viewport CSS pixels, in `steps` moves. */
  async mouseMove(x, y, steps = 1) {
    const from = this.mousePos;
    const moves = [];
    for (let i = 1; i <= steps; i++) {
      moves.push({
        type: "pointerMove",
        origin: "viewport",
        x: Math.round(from.x + ((x - from.x) * i) / steps),
        y: Math.round(from.y + ((y - from.y) * i) / steps),
        duration: 0,
      });
    }
    await this.pointer(moves);
    this.mousePos = { x, y };
  }

  mouseDown(button = 0) {
    return this.pointer([{ type: "pointerDown", button }]);
  }

  mouseUp(button = 0) {
    return this.pointer([{ type: "pointerUp", button }]);
  }

  async click(x, y) {
    await this.mouseMove(x, y);
    await this.pointer([
      { type: "pointerDown", button: 0 },
      { type: "pointerUp", button: 0 },
    ]);
  }

  keyDown(key) {
    return this.actions([
      {
        type: "key",
        id: "keyboard",
        actions: [{ type: "keyDown", value: keyValue(key) }],
      },
    ]);
  }

  keyUp(key) {
    return this.actions([
      {
        type: "key",
        id: "keyboard",
        actions: [{ type: "keyUp", value: keyValue(key) }],
      },
    ]);
  }

  press(key) {
    const value = keyValue(key);
    return this.actions([
      {
        type: "key",
        id: "keyboard",
        actions: [
          { type: "keyDown", value },
          { type: "keyUp", value },
        ],
      },
    ]);
  }

  async stop() {
    if (this.session) await this.call("DELETE", this.s("")).catch(() => {});
    this.session = null;
  }
}

/** Is this a refusal to automate (Safari's "Allow remote automation" off)? */
export function isAutomationDisabled(e) {
  return /remote automation|Allow Remote Automation|session not created|connection refused/i.test(
    `${e?.webdriver ?? ""} ${e?.message ?? ""} ${e?.detail ?? ""}`,
  );
}

/** Is Safari already driven by another session (only one at a time)? */
export function isSafariBusy(e) {
  return /already paired|already (has|in) (a|an active) session|session already/i.test(
    `${e?.message ?? ""} ${e?.detail ?? ""}`,
  );
}

/** Starts `safaridriver -p port` and waits for it to answer /status. */
export async function startSafariDriver(port, bin = "safaridriver") {
  const driver = spawn(bin, ["-p", String(port)], {
    stdio: ["ignore", "pipe", "pipe"],
  });
  let output = "";
  driver.stdout.on("data", (d) => {
    output += d;
  });
  driver.stderr.on("data", (d) => {
    output += d;
  });
  let spawnError = null;
  driver.on("error", (e) => {
    spawnError = e;
  });
  for (let i = 0; i < 50; i++) {
    if (spawnError) throw new Error(`can't run ${bin}: ${spawnError.message}`);
    if (driver.exitCode !== null) {
      throw new Error(`${bin} exited (${driver.exitCode}): ${output.trim()}`);
    }
    try {
      const res = await fetch(`http://localhost:${port}/status`);
      if (res.ok) return driver;
    } catch {}
    await sleep(100);
  }
  driver.kill();
  throw new Error(`${bin} did not answer on port ${port}: ${output.trim()}`);
}
