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
