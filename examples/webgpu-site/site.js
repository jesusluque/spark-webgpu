// Shared helpers for the public Spark WebGPU demo pages.

export const NO_WEBGPU =
  "This demo needs WebGPU: current Chrome, Edge or Safari 26+.";

/** The WebGPU adapter, or null when the browser has no usable WebGPU. */
export async function webgpuAdapter() {
  if (!navigator.gpu) return null;
  try {
    return await navigator.gpu.requestAdapter();
  } catch {
    return null;
  }
}

/** Covers the page with the "needs WebGPU" message. */
export function showNoWebGPU(detail = "") {
  const box = document.createElement("div");
  box.className = "nogpu";
  box.id = "nogpu";
  box.innerHTML = `<div><h2>${NO_WEBGPU}</h2>
    <p>WebGPU is not available in this browser${detail ? ` (${detail})` : ""}.</p>
    <p><a href="./">Back to the demos</a></p></div>`;
  document.body.appendChild(box);
}

/** Stops the page with the message when there is no WebGPU adapter. */
export async function requireWebGPU() {
  const adapter = await webgpuAdapter();
  if (!adapter) {
    showNoWebGPU(navigator.gpu ? "no adapter" : "navigator.gpu is missing");
    throw new Error("no WebGPU");
  }
  return adapter;
}

/** Shows a short message at the top of the page. */
export function banner(text) {
  const el = document.createElement("div");
  el.className = "banner";
  el.id = "banner";
  el.textContent = text;
  document.body.appendChild(el);
  return el;
}

/** Counts frames and calls report(fps) twice a second. */
export function fpsMeter(report) {
  let frames = 0;
  let last = performance.now();
  return () => {
    frames++;
    const now = performance.now();
    if (now - last > 500) {
      report((frames * 1000) / (now - last));
      frames = 0;
      last = now;
    }
  };
}

const MB = 1024 * 1024;

function formatSeconds(s) {
  if (!Number.isFinite(s)) return "…";
  if (s < 60) return `${Math.max(1, Math.round(s))} s`;
  const m = Math.floor(s / 60);
  return `${m} min ${String(Math.round(s % 60)).padStart(2, "0")} s`;
}

/**
 * Shows a download bar with megabytes, speed and time left for every fetch
 * whose URL matches `match`. It wraps window.fetch and counts the bytes as
 * the response bodies are read, so it covers the loaders' own fetches.
 * `expectedBytes` (optional) is the total to show before the responses'
 * Content-Length headers arrive. Returns { done() } to remove the bar.
 */
export function downloadProgress({
  match,
  expectedBytes = 0,
  label = "Downloading",
}) {
  const box = document.createElement("div");
  box.className = "download";
  box.innerHTML = `<div class="download-text"></div><div class="download-bar"><div></div></div>`;
  document.body.appendChild(box);
  const text = box.querySelector(".download-text");
  const fill = box.querySelector(".download-bar > div");

  let loaded = 0;
  let announced = 0; // sum of Content-Length of the matched responses
  const started = performance.now();
  let finished = false;

  function render() {
    if (finished) return;
    const total = Math.max(announced, expectedBytes, loaded);
    const elapsed = (performance.now() - started) / 1000;
    const rate = elapsed > 0.5 ? loaded / elapsed : 0;
    const left =
      rate > 0 && total > loaded ? (total - loaded) / rate : Number.NaN;
    const parts = [
      `${label} ${(loaded / MB).toFixed(0)} of ${(total / MB).toFixed(0)} MB`,
    ];
    if (rate > 0) parts.push(`${(rate / MB).toFixed(1)} MB/s`);
    if (total > loaded) parts.push(`${formatSeconds(left)} left`);
    text.textContent = parts.join(" · ");
    fill.style.width = `${total ? Math.min(100, (100 * loaded) / total) : 0}%`;
  }
  const timer = setInterval(render, 250);
  render();

  const originalFetch = window.fetch.bind(window);
  window.fetch = async (input, init) => {
    const response = await originalFetch(input, init);
    const url = typeof input === "string" ? input : input.url;
    if (finished || !match(url) || !response.body) return response;
    const length = Number(response.headers.get("Content-Length") ?? 0);
    announced += length;
    const reader = response.body.getReader();
    const counted = new ReadableStream({
      async pull(controller) {
        const { done, value } = await reader.read();
        if (done) {
          controller.close();
          return;
        }
        loaded += value.byteLength;
        controller.enqueue(value);
      },
      cancel(reason) {
        return reader.cancel(reason);
      },
    });
    return new Response(counted, {
      status: response.status,
      statusText: response.statusText,
      headers: response.headers,
    });
  };

  return {
    done() {
      finished = true;
      clearInterval(timer);
      window.fetch = originalFetch;
      box.remove();
    },
  };
}

/**
 * aofx Grade as the scene's colour corrector, after the display transform
 * (display-referred, as a colourist grades): a step in `chain` and a folder
 * in `gui`. Lift, gain, gamma, multiply and offset each get a tint and an
 * amount; alpha is left alone. Call keepLast() after anything else (a
 * PluginHost's applyFx) has added its steps, so the grade stays the last one.
 */
export function addColourCorrector(gui, chain, fx) {
  const grade = new fx.Grade();
  const step = chain.add(grade, fx.FxChain.defaults(grade), {
    enabled: true,
    channels: [true, true, true, false],
    instance: "colour-corrector",
  });
  const folder = gui.addFolder("colour corrector (aofx Grade)").close();
  folder.add(step, "enabled");
  const knobs = [
    ["black", "lift", 0, -0.5, 0.5],
    ["white", "gain", 1, 0, 4],
    ["gamma", "gamma", 1, 0.2, 3],
    ["multiply", "multiply", 1, 0, 4],
    ["offset", "offset", 0, -0.5, 0.5],
  ];
  const ui = {};
  const apply = (name) => {
    const { tint, amount } = ui[name];
    const rgb = tint.map((c) => c * amount);
    // A neutral tint keeps lift and offset adding the same to every channel.
    step.params[name] = [...rgb, step.params[name]?.[3] ?? 1];
  };
  for (const [name, label, value, min, max] of knobs) {
    ui[name] = { tint: [1, 1, 1], amount: value };
    folder
      .add(ui[name], "amount", min, max, 0.01)
      .name(label)
      .onChange(() => apply(name));
    folder
      .addColor(ui[name], "tint")
      .name(`${label} tint`)
      .onChange(() => apply(name));
    apply(name);
  }
  folder.add(step.params, "clampWhite").name("clamp white");
  folder
    .add(
      {
        reset() {
          for (const [name, , value] of knobs) {
            ui[name].tint = [1, 1, 1];
            ui[name].amount = value;
            apply(name);
          }
          for (const c of folder.controllersRecursive()) c.updateDisplay();
        },
      },
      "reset",
    )
    .name("reset grade");
  return {
    step,
    keepLast() {
      chain.steps = [...chain.steps.filter((s) => s !== step), step];
    },
  };
}

/** A desktop with a mouse (not a phone or tablet). */
export function isWorkstation() {
  return (
    matchMedia("(pointer: fine)").matches &&
    !/Android|iPhone|iPad|Mobile/i.test(navigator.userAgent)
  );
}

/**
 * Autodesk Maya's camera on an OrbitControls: Alt + left drag tumbles,
 * Alt + middle drag tracks, Alt + right drag dollies, the wheel dollies, F
 * frames the subject again. Without Alt the mouse does nothing to the
 * camera, so the panels and the scene can be clicked freely.
 */
export function mayaControls(controls, frame) {
  const ROTATE = 0;
  const DOLLY = 1;
  const PAN = 2;
  controls.mouseButtons = { LEFT: ROTATE, MIDDLE: PAN, RIGHT: DOLLY };
  controls.enableDamping = false;
  controls.screenSpacePanning = true;
  const dom = controls.domElement;
  // OrbitControls listens on pointerdown; decide first, in the capture phase.
  dom.addEventListener(
    "pointerdown",
    (event) => {
      if (event.pointerType === "mouse") controls.enabled = event.altKey;
    },
    { capture: true },
  );
  const restore = () => {
    controls.enabled = true;
  };
  dom.addEventListener("pointerup", restore);
  dom.addEventListener("pointercancel", restore);
  // Alt + right drag must not open the context menu.
  dom.addEventListener("contextmenu", (event) => event.preventDefault());
  addEventListener("keydown", (event) => {
    if (event.target instanceof HTMLInputElement) return;
    if (event.key === "f" || event.key === "F") frame();
  });
}

/**
 * The relit pages' domes: Poly Haven's full-resolution 4k lat-longs (4096 x
 * 2048, CC0), Radiance RGBE as published -- never downsized web copies. One
 * folder on R2 for every page (?hdriBase=<url/> points elsewhere).
 */
export const HDRI_BASE =
  "https://athenea-assets.lucab.co.uk/sparkwebgpu/hdri-4k/";
export const HDRIS_4K = [
  "autoshop_01_4k.hdr",
  "brown_photostudio_02_4k.hdr",
  "dikhololo_night_4k.hdr",
  "golden_gate_hills_4k.hdr",
  "kloppenheim_06_puresky_4k.hdr",
  "lythwood_room_4k.hdr",
  "moonless_golf_4k.hdr",
  "snowy_park_01_4k.hdr",
  "studio_small_09_4k.hdr",
  "sunset_fairway_4k.hdr",
  "thatch_chapel_4k.hdr",
  "urban_alley_01_4k.hdr",
  "venice_sunset_4k.hdr",
];
/** About what one 4k HDRI downloads (for the download bar). */
export const HDRI_4K_BYTES = 25 * 1024 * 1024;

/** The full-resolution file for a name a scene JSON may give at 1k or 2k. */
export function fullResHdri(name) {
  return name.replace(/_(1k|2k|8k)\.hdr$/, "_4k.hdr");
}

/** A dome's name for the HUD: autoshop_01_4k.hdr -> autoshop_01. */
export function hdriLabel(name) {
  return name.replace(/_\dk\.hdr$/, "");
}

/**
 * The largest finite half float. A dome drawn into a HalfFloat target
 * overflows to infinity above it (golden_gate_hills' and moonless_golf's
 * suns reach 1e5), and infinity under a splat's transmittance stays
 * infinite: the backdrop is held here, still far above any display white.
 */
export const HALF_MAX = 65504;
