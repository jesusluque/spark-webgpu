// The WebDriver client and Safari launch path against a mock safaridriver,
// the Vite plugin that stands in for init scripts on Safari,
// and the PNG codec. Run: npm run test:parity:unit
//
// safaridriver itself needs "Allow remote automation", so the client is
// checked here against the wire protocol it should speak.

import assert from "node:assert/strict";
import http from "node:http";
import { after, before, describe, test } from "node:test";
import { SafariDriver } from "./drivers.mjs";
import { decodePng, diffImages, downscale, encodePng } from "./png.mjs";
import { parityPlugin } from "./server.mjs";
import {
  KEYS,
  WebDriver,
  isAutomationDisabled,
  isSafariBusy,
  keyValue,
} from "./webdriver.mjs";

const PNG_1x1 = encodePng({
  width: 1,
  height: 1,
  data: new Uint8Array([10, 20, 30, 255]),
});

/** A mock WebDriver server: records requests, answers per route. */
function mockDriver({ refuseSession = false } = {}) {
  const requests = [];
  const state = { inner: [700, 520], window: null, script: null };
  const server = http.createServer(async (req, res) => {
    let body = "";
    for await (const c of req) body += c;
    const json = body ? JSON.parse(body) : undefined;
    requests.push({ method: req.method, url: req.url, body: json });
    const send = (status, value) => {
      res.writeHead(status, { "content-type": "application/json" });
      res.end(JSON.stringify({ value }));
    };
    const route = `${req.method} ${req.url}`;
    if (route === "GET /status") return send(200, { ready: true });
    if (route === "POST /session") {
      if (refuseSession) {
        return send(500, {
          error: "session not created",
          message:
            "Could not create a session: You must enable 'Allow remote automation' in the Develop menu of Safari to control Safari via WebDriver.",
        });
      }
      return send(200, {
        sessionId: "S1",
        capabilities: {
          browserName: "safari",
          browserVersion: "26.0",
          platformName: "macOS",
        },
      });
    }
    const m = req.url.match(/^\/session\/S1(\/.*)?$/);
    if (!m) return send(404, { error: "invalid session id", message: req.url });
    const r = `${req.method} ${m[1] ?? ""}`;
    if (r === "POST /timeouts") return send(200, null);
    if (r === "POST /url") return send(200, null);
    if (r === "GET /screenshot") return send(200, PNG_1x1.toString("base64"));
    if (r === "POST /window/rect") {
      // The toolbar takes 80 px: the viewport is the window minus that.
      state.window = [json.width, json.height];
      state.inner = [json.width, json.height - 80];
      return send(200, { width: json.width, height: json.height });
    }
    if (r === "POST /execute/sync") {
      if (json.script.includes("devicePixelRatio"))
        return send(200, [...state.inner, 2]);
      if (json.script.includes("innerWidth")) return send(200, state.inner);
      return send(200, json.args?.[0] ?? true);
    }
    if (r === "POST /execute/async") {
      state.script = json.script;
      const arg = json.args[0];
      if (arg === "throw") return send(200, { err: "Error: boom" });
      return send(200, { ok: { echoed: arg } });
    }
    if (r === "POST /actions" || r === "DELETE /actions")
      return send(200, null);
    if (r === "DELETE ") return send(200, null);
    return send(404, { error: "unknown command", message: r });
  });
  return new Promise((resolve) =>
    server.listen(0, () =>
      resolve({
        server,
        requests,
        state,
        url: `http://localhost:${server.address().port}`,
      }),
    ),
  );
}

describe("WebDriver client", () => {
  let mock;
  let wd;
  before(async () => {
    mock = await mockDriver();
    wd = new WebDriver(mock.url);
  });
  after(() => mock.server.close());

  test("creates a session with Safari capabilities and long script timeouts", async () => {
    const caps = await wd.start();
    assert.equal(caps.browserVersion, "26.0");
    assert.equal(wd.session, "S1");
    const create = mock.requests.find((r) => r.url === "/session");
    assert.deepEqual(create.body, {
      capabilities: { alwaysMatch: { browserName: "safari" } },
    });
    const timeouts = mock.requests.find(
      (r) => r.url === "/session/S1/timeouts",
    );
    assert.ok(timeouts.body.script >= 60000);
  });

  test("navigates and runs sync scripts", async () => {
    await wd.navigate("http://localhost:8081/x.html");
    assert.deepEqual(mock.requests.at(-1), {
      method: "POST",
      url: "/session/S1/url",
      body: { url: "http://localhost:8081/x.html" },
    });
    assert.equal(await wd.exec("return arguments[0];", [5]), 5);
  });

  test("evaluate wraps a function as an async script and unwraps its result", async () => {
    const v = await wd.evaluate((a) => a * 2, 21);
    assert.deepEqual(v, { echoed: 21 });
    assert.match(mock.state.script, /arguments\[arguments.length - 1\]/);
    assert.match(mock.state.script, /\(a\) => a \* 2/);
    await assert.rejects(
      wd.evaluate(() => 0, "throw"),
      /page script failed: Error: boom/,
    );
  });

  test("sizes the window so the viewport is exact", async () => {
    const [w, h, dpr] = await wd.setViewport(800, 600);
    assert.deepEqual([w, h, dpr], [800, 600, 2]);
    assert.deepEqual(mock.state.window, [800, 680]);
  });

  test("screenshot decodes base64 PNG", async () => {
    const png = await wd.screenshot();
    const img = decodePng(png);
    assert.deepEqual(
      [img.width, img.height, ...img.data],
      [1, 1, 10, 20, 30, 255],
    );
  });

  test("pointer actions: moves interpolate from the last position", async () => {
    await wd.mouseMove(100, 50);
    await wd.mouseMove(200, 150, 2);
    const moves = mock.requests.at(-1).body.actions[0];
    assert.equal(moves.type, "pointer");
    assert.equal(moves.parameters.pointerType, "mouse");
    assert.deepEqual(
      moves.actions.map((a) => [a.type, a.origin, a.x, a.y]),
      [
        ["pointerMove", "viewport", 150, 100],
        ["pointerMove", "viewport", 200, 150],
      ],
    );
    await wd.click(10, 20);
    const click = mock.requests.at(-1).body.actions[0].actions;
    assert.deepEqual(
      click.map((a) => a.type),
      ["pointerDown", "pointerUp"],
    );
    assert.deepEqual(mock.requests.at(-2).body.actions[0].actions[0].x, 10);
  });

  test("key actions map named keys to WebDriver codes", async () => {
    await wd.press("Escape");
    const keys = mock.requests.at(-1).body.actions[0];
    assert.equal(keys.type, "key");
    assert.deepEqual(keys.actions, [
      { type: "keyDown", value: KEYS.Escape },
      { type: "keyUp", value: KEYS.Escape },
    ]);
    await wd.keyDown("w");
    assert.deepEqual(mock.requests.at(-1).body.actions[0].actions, [
      { type: "keyDown", value: "w" },
    ]);
    assert.equal(keyValue("]"), "]");
    assert.throws(() => keyValue("F13"), /no WebDriver key/);
  });

  test("stop deletes the session", async () => {
    await wd.stop();
    assert.deepEqual(mock.requests.at(-1), {
      method: "DELETE",
      url: "/session/S1",
      body: undefined,
    });
    assert.equal(wd.session, null);
  });
});

describe("Safari unavailable", () => {
  test("a refused session is reported as automation disabled", async () => {
    const mock = await mockDriver({ refuseSession: true });
    try {
      const wd = new WebDriver(mock.url);
      const err = await wd.start().catch((e) => e);
      assert.equal(err.webdriver, "session not created");
      assert.ok(isAutomationDisabled(err));
    } finally {
      mock.server.close();
    }
  });

  test("a Safari driven by another session is busy, not disabled", () => {
    const e = {
      webdriver: "session not created",
      message: "POST /session: session not created",
      detail:
        "Could not create a session: The Safari instance is already paired with a different session.",
    };
    assert.ok(isSafariBusy(e));
    assert.ok(
      !isSafariBusy({ detail: "You must enable 'Allow remote automation'" }),
    );
  });

  test("no server is reported as unavailable", async () => {
    const err = await new WebDriver("http://localhost:1")
      .status()
      .catch((e) => e);
    assert.ok(isAutomationDisabled(err));
  });

  test("SafariDriver.launch gives SAFARI_UNAVAILABLE when the driver can't run", async () => {
    const err = await SafariDriver.launch({
      port: 4999,
      bin: "/nonexistent/safaridriver",
    }).catch((e) => e);
    assert.equal(err.code, "SAFARI_UNAVAILABLE");
    assert.match(err.message, /safaridriver could not start/);
  });
});

describe("PNG and diff", () => {
  const img = (w, h, f) => {
    const data = new Uint8Array(w * h * 4);
    for (let i = 0; i < w * h; i++) data.set([...f(i), 255], i * 4);
    return { width: w, height: h, data };
  };

  test("encode/decode round trip", () => {
    const a = img(5, 3, (i) => [i * 10, 255 - i, (i * 37) % 256]);
    const b = decodePng(encodePng(a));
    assert.equal(b.width, 5);
    assert.deepEqual([...b.data], [...a.data]);
  });

  test("mean, % over 16 and lit fractions", () => {
    const a = img(2, 2, () => [0, 0, 0]);
    const b = img(2, 2, (i) => (i === 0 ? [30, 0, 0] : [0, 0, 0]));
    const d = diffImages(a, b);
    assert.equal(d.mean, 30 / 12);
    assert.equal(d.pctOver, 25);
    assert.equal(d.litA, 0);
    assert.equal(d.litB, 25);
  });

  test("a 2x image is box-filtered to the other's size", () => {
    const big = img(4, 4, () => [100, 100, 100]);
    const small = img(2, 2, () => [100, 100, 100]);
    assert.equal(diffImages(big, small).mean, 0);
    assert.equal(downscale(big, 2).width, 2);
    assert.throws(
      () =>
        diffImages(
          img(3, 3, () => [0, 0, 0]),
          small,
        ),
      /sizes differ/,
    );
  });
});

describe("Vite plugin (Safari's clock, routes)", () => {
  const plugin = parityPlugin([
    { path: /^\/a\/b\.spz$/, redirect: () => "https://example.com/x.spz" },
    { path: /^\/p\.html$/, rewrite: (t) => t.replace("OLD", "NEW") },
  ]);
  let handler;
  plugin.configureServer({
    middlewares: {
      use: (fn) => {
        handler = fn;
      },
    },
    transformRequest: async () => ({ code: "const v = 'OLD';" }),
  });
  const request = (url) =>
    new Promise((resolve) => {
      const res = {
        statusCode: 200,
        headers: {},
        setHeader(k, v) {
          this.headers[k.toLowerCase()] = v;
        },
        end(body) {
          resolve({ status: this.statusCode, headers: this.headers, body });
        },
      };
      handler({ url }, res, () => resolve({ next: true }));
    });

  test("pages get the clock, run from sessionStorage", () => {
    const out = plugin.transformIndexHtml("<p>OLD</p>", { path: "/p.html" });
    assert.equal(out.html, "<p>NEW</p>");
    assert.equal(out.tags[0].injectTo, "head-prepend");
    assert.match(out.tags[0].children, /sessionStorage\.getItem\("__parity"\)/);
    assert.match(out.tags[0].children, /window\.__step/);
  });

  test("serves the HMR stub and the blank page", async () => {
    const client = await request("/@vite/client");
    assert.match(client.body, /export function createHotContext/);
    assert.doesNotMatch(client.body, /WebSocket/);
    assert.equal(
      (await request("/__parity/blank")).headers["content-type"],
      "text/html",
    );
  });

  test("redirects and rewrites routed paths, passes the rest on", async () => {
    const r = await request("/a/b.spz");
    assert.equal(r.status, 302);
    assert.equal(r.headers.location, "https://example.com/x.spz");
    const m = await request("/p.html?html-proxy&index=0.js");
    assert.equal(m.body, "const v = 'NEW';");
    assert.deepEqual(await request("/p.html"), { next: true });
    assert.deepEqual(await request("/other.js"), { next: true });
  });
});
