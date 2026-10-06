// Writes results.json and index.html (a table, then WebGL | WebGPU | diff
// for each case) into the output directory.

import fs from "node:fs";
import path from "node:path";

const esc = (s) =>
  String(s ?? "").replace(
    /[&<>"]/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c],
  );
const num = (v, d = 2) => (v == null ? "" : v.toFixed(d));

export function writeReport(
  out,
  { results, perf, opts, versions, base, seconds, skipped },
) {
  const json = {
    date: new Date().toISOString(),
    browsers: { webgl: opts.webgl, webgpu: opts.webgpu },
    versions,
    base,
    seconds,
    skipped,
    results,
    perf,
  };
  fs.writeFileSync(
    path.join(out, "results.json"),
    JSON.stringify(json, null, 2),
  );

  const rows = results
    .map(
      (r) => `<tr class="${r.status}">
  <td><a href="#${esc(r.id)}">${esc(r.id)}</a></td>
  <td class="st">${r.status}</td>
  <td>${num(r.mean)}</td><td>${r.max?.mean ?? ""}</td><td>${r.readme ?? ""}</td>
  <td>${num(r.pctOver)}</td><td>${r.max?.pct ?? ""}</td>
  <td>${r.settled ? `${r.settled.webgl.settled ? "" : "⏱"}${(r.settled.webgl.ms / 1000).toFixed(0)} / ${r.settled.webgpu.settled ? "" : "⏱"}${(r.settled.webgpu.ms / 1000).toFixed(0)}` : ""}</td>
  <td class="why">${esc((r.reasons ?? []).join("; "))}</td>
</tr>`,
    )
    .join("\n");
  const details = results
    .filter((r) => r.images)
    .map(
      (r) => `<section id="${esc(r.id)}">
  <h2>${esc(r.id)} <span class="st ${r.status}">${r.status}</span></h2>
  <p>mean ${num(r.mean)}/255 (max ${r.max?.mean ?? "–"}, README ${r.readme ?? "–"}) · ${num(r.pctOver)}% of pixels over 16 · lit ${num(r.lit.webgl, 1)}% / ${num(r.lit.webgpu, 1)}%
  · <a href="${esc(base + r.webglUrl)}">WebGL page</a> · <a href="${esc(base + r.webgpuUrl)}">WebGPU page</a></p>
  <div class="imgs">
    <figure><img loading="lazy" src="${r.images.webgl}"><figcaption>WebGL (${r.browsers.webgl})</figcaption></figure>
    <figure><img loading="lazy" src="${r.images.webgpu}"><figcaption>WebGPU (${r.browsers.webgpu})</figcaption></figure>
    <figure><img loading="lazy" src="${r.images.diff}"><figcaption>|diff| × 4</figcaption></figure>
  </div>
  ${["webgl", "webgpu"]
    .filter((s) => r.logs?.[s]?.length)
    .map(
      (s) =>
        `<details><summary>${s} console (${r.logs[s].length})</summary><pre>${esc(r.logs[s].join("\n"))}</pre></details>`,
    )
    .join("")}
</section>`,
    )
    .join("\n");
  const perfTable = perf?.length
    ? `<h2>Performance</h2>
<p>fps, median of ${opts.perfRuns} × ${opts.perfSeconds} s, real clock, Chrome with vsync and the frame-rate limit off. The GPU may be shared: treat as rough.</p>
<table><tr><th>example</th><th>WebGL</th><th>WebGPU</th><th>ratio</th></tr>
${perf
  .map((p) => {
    const f = (x) =>
      x?.error ? `<span title="${esc(x.error)}">error</span>` : num(x?.fps, 0);
    const ratio =
      p.webgl?.fps && p.webgpu?.fps ? num(p.webgpu.fps / p.webgl.fps) : "";
    return `<tr><td>${esc(p.name)}</td><td>${f(p.webgl)}</td><td>${f(p.webgpu)}</td><td>${ratio}</td></tr>`;
  })
  .join("\n")}
</table>`
    : "";
  const html = `<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Spark parity</title>
<style>
  :root { color-scheme: light dark; --fg: #222; --bg: #fafafa; --mute: #777; --line: #ddd; --pass: #2a7a3a; --fail: #b3261e; --warn: #a06200; }
  @media (prefers-color-scheme: dark) { :root { --fg: #ddd; --bg: #161616; --mute: #999; --line: #333; --pass: #6c6; --fail: #f77; --warn: #fb5; } }
  body { font: 14px/1.4 system-ui, sans-serif; color: var(--fg); background: var(--bg); margin: 0; padding: 16px 24px; }
  table { border-collapse: collapse; font-variant-numeric: tabular-nums; }
  td, th { border-bottom: 1px solid var(--line); padding: 3px 10px 3px 0; text-align: left; vertical-align: top; }
  .wrap { overflow-x: auto; }
  .pass .st, .st.pass { color: var(--pass); }
  .fail .st, .error .st, .st.fail, .st.error { color: var(--fail); font-weight: 600; }
  .flaky .st, .skip .st, .st.flaky { color: var(--warn); }
  .why { color: var(--mute); max-width: 40em; }
  .imgs { display: flex; flex-wrap: wrap; gap: 8px; }
  figure { margin: 0; flex: 1 1 260px; max-width: 400px; }
  img { width: 100%; display: block; image-rendering: pixelated; }
  figcaption { color: var(--mute); font-size: 12px; }
  section { margin-top: 28px; }
  h2 { font-size: 16px; margin: 0 0 4px; }
  pre { white-space: pre-wrap; font-size: 12px; }
</style>
<h1>Spark WebGPU vs WebGL parity</h1>
<p>${esc(json.date)} · ${esc((versions ?? []).join(", "))} · WebGL in ${opts.webgl}, WebGPU in ${opts.webgpu}${seconds ? ` · ${seconds.toFixed(0)} s` : ""}</p>
${skipped ? `<p class="why">Skipped: ${esc(skipped)}</p>` : ""}
<p>Mean absolute difference per channel, in /255; "&gt;16" is the % of pixels with a channel off by more than 16. Settle: seconds to a stable frame (⏱: gave up waiting).</p>
<div class="wrap"><table>
<tr><th>case</th><th>status</th><th>mean</th><th>max</th><th>README</th><th>&gt;16 %</th><th>max</th><th>settle s</th><th></th></tr>
${rows}
</table></div>
${perfTable}
${details}
`;
  fs.writeFileSync(path.join(out, "index.html"), html);
}
