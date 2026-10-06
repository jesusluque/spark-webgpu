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
