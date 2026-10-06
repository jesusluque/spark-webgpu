// Page-side setup, installed before any page script runs (Playwright's
// addInitScript in Chrome; on Safari, the runner's Vite plugin inlines it at
// the top of every page and reads the config from sessionStorage).
//
// With `freeze`, performance.now, Date.now and rAF timestamps all return a
// time T that moves only when the runner steps frames: animations and physics
// stop where they are, so two pages that load at different speeds end up
// drawing the same instant. rAF callbacks still run every real frame
// ("auto") until the first __step(n, dt), which runs n frames, advancing T by
// dt before each, and leaves the page paused. Math.random is a seeded
// mulberry32 in both modes; __reseed(s) restarts it (lofi reseeds where it
// shuffles, after the backends have drawn different numbers of randoms).
//
// This function is serialized into the page: it must not close over anything.
export function parityInit(cfg) {
  if (window.__parity) return;
  window.__parity = cfg;
  let s = cfg.seed >>> 0;
  window.__reseed = (v = cfg.seed) => {
    s = v >>> 0;
  };
  Math.random = () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let r = Math.imul(s ^ (s >>> 15), 1 | s);
    r = (r + Math.imul(r ^ (r >>> 7), 61 | r)) ^ r;
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
  window.__frames = 0;
  if (cfg.log) {
    // WebDriver has no console log: keep errors where the runner can read
    // them, from the first script on.
    const log = [];
    window.__parityLog = log;
    const text = (a) =>
      a
        .map((x) =>
          x instanceof Error
            ? x.message
            : typeof x === "object"
              ? (() => {
                  try {
                    return JSON.stringify(x);
                  } catch {
                    return String(x);
                  }
                })()
              : String(x),
        )
        .join(" ");
    for (const level of ["error", "warn", "log", "info"]) {
      const orig = console[level].bind(console);
      console[level] = (...a) => {
        log.push(`[${level}] ${text(a)}`);
        orig(...a);
      };
    }
    window.addEventListener("error", (e) =>
      log.push(`[pageerror] ${e.message}`),
    );
    window.addEventListener("unhandledrejection", (e) =>
      log.push(`[pageerror] ${e.reason?.message ?? e.reason}`),
    );
  }
  if (!cfg.freeze) {
    // Perf mode: the real clock, and a frame counter for fps.
    const realRaf = window.requestAnimationFrame.bind(window);
    const count = () => {
      window.__frames++;
      realRaf(count);
    };
    realRaf(count);
    return;
  }
  let t = cfg.t0;
  const realRaf = window.requestAnimationFrame.bind(window);
  const start = cfg.date ?? 1.7e12;
  performance.now = () => t;
  Date.now = () => start + t;
  let queue = [];
  // `manual` pages draw nothing until stepped: for animations that advance
  // per frame rather than with time, so load speed doesn't change the frame.
  let auto = !cfg.manual;
  let id = 1;
  window.requestAnimationFrame = (cb) => {
    queue.push({ cb, id });
    return id++;
  };
  window.cancelAnimationFrame = (cid) => {
    queue = queue.filter((q) => q.id !== cid);
  };
  const run = () => {
    const q = queue;
    queue = [];
    window.__frames++;
    for (const { cb } of q) {
      try {
        cb(t);
      } catch (e) {
        console.error(e);
      }
    }
  };
  const pump = () => {
    if (auto) run();
    realRaf(pump);
  };
  realRaf(pump);
  // Runs n frames, advancing the clock by dt ms before each, and waits a real
  // frame after each so the GPU work is presented.
  window.__step = async (n = 1, dt = 1000 / 60) => {
    auto = false;
    for (let i = 0; i < n; i++) {
      t += dt;
      run();
      await new Promise((r) => realRaf(r));
    }
  };
  window.__setTime = (v) => {
    t = v;
  };
  window.__auto = (v) => {
    auto = v;
  };
}

// Vite's HMR client, stubbed: the runner's pages must not reload or restyle
// when another agent edits files mid-run. Keeps the exports that transformed
// modules import.
export const VITE_CLIENT_STUB = `
const hot = { data: {}, accept() {}, acceptExports() {}, dispose() {}, prune() {}, decline() {}, invalidate() {}, on() {}, off() {}, send() {} };
export function createHotContext() { return hot; }
export function updateStyle(id, content) {
  let s = document.querySelector('style[data-vite-dev-id="' + id + '"]');
  if (!s) {
    s = document.createElement("style");
    s.setAttribute("type", "text/css");
    s.setAttribute("data-vite-dev-id", id);
    document.head.appendChild(s);
  }
  s.textContent = content;
}
export function removeStyle(id) {
  document.querySelector('style[data-vite-dev-id="' + id + '"]')?.remove();
}
export function injectQuery(url) { return url; }
export class ErrorOverlay extends HTMLElement {}
`;
