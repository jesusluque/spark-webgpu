// node compare.mjs <json spec>
// spec: { ref: dir|file-pattern, cands: {label: dir}, views: [..], car: {bgDir}|{alpha: file},
//         parts: {name: maskDir pattern "<dir>/<view>.no-<g>.pfm"} , lamps: {onDir, offDir}, out: dir, tag }
// Metrics over masks, linear float, values clamped to [0, CLAMP]:
//   relMSE = mean over pixels and channels of (x - r)^2 / (r^2 + 0.01)
//   ratio  = sum x / sum r (brightness bias)
import { execFileSync } from "node:child_process";
import fs from "node:fs";

const CLAMP = 16;
export function readPfm(file) {
  const b = fs.readFileSync(file);
  let p = 0;
  const tok = () => {
    while (b[p] === 10 || b[p] === 32) p++;
    const s = p;
    while (b[p] !== 10 && b[p] !== 32) p++;
    return b.toString("latin1", s, p);
  };
  const kind = tok();
  const w = Number(tok());
  const h = Number(tok());
  const scale = Number(tok());
  p++;
  const ch = kind === "PF" ? 3 : 1;
  const f = new Float32Array(w * h * ch);
  for (let i = 0; i < f.length; i++)
    f[i] = scale < 0 ? b.readFloatLE(p + 4 * i) : b.readFloatBE(p + 4 * i);
  return { w, h, ch, f };
}
function writePfm(file, w, h, f) {
  const buf = Buffer.alloc(w * h * 12);
  for (let i = 0; i < f.length; i++) buf.writeFloatLE(f[i], 4 * i);
  fs.writeFileSync(
    file,
    Buffer.concat([Buffer.from(`PF\n${w} ${h}\n-1.0\n`), buf]),
  );
}
const lum = (f, i) =>
  0.2126 * f[3 * i] + 0.7152 * f[3 * i + 1] + 0.0722 * f[3 * i + 2];
const cl = (v) => Math.min(Math.max(v, 0), CLAMP);

export function diffMask(a, b, rel = 0.02, abs = 0.01) {
  const n = a.w * a.h;
  const m = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    let d = 0;
    let s = 0;
    for (let c = 0; c < 3; c++) {
      d = Math.max(d, Math.abs(cl(a.f[3 * i + c]) - cl(b.f[3 * i + c])));
      s = Math.max(s, cl(b.f[3 * i + c]));
    }
    m[i] = d > abs + rel * s ? 1 : 0;
  }
  return m;
}
export function metrics(x, r, mask) {
  let e = 0;
  let n = 0;
  let sx = 0;
  let sr = 0;
  let flies = 0;
  const per = [];
  for (let i = 0; i < x.w * x.h; i++) {
    if (mask && !mask[i]) continue;
    let pe = 0;
    for (let c = 0; c < 3; c++) {
      const a = cl(x.f[3 * i + c]);
      const b = cl(r.f[3 * i + c]);
      pe += ((a - b) * (a - b)) / (b * b + 0.01);
      sx += a;
      sr += b;
    }
    e += pe;
    per.push(pe / 3);
    // a firefly: a pixel far brighter than the reference (and bright)
    const lx = lum(x.f, i);
    const lr = lum(r.f, i);
    if (lx > 1 && lx > 4 * lr + 0.25) flies++;
    n++;
  }
  // robust: the mean without the worst 0.1% of the pixels
  per.sort((a, b) => a - b);
  const keep = Math.floor(per.length * 0.999);
  let t = 0;
  for (let i = 0; i < keep; i++) t += per[i];
  return {
    relMSE: n ? e / (3 * n) : null,
    trim: keep ? t / keep : null,
    ratio: sr ? sx / sr : null,
    flies,
    pixels: n,
  };
}
export function preview(pfm, png, exposure = 0) {
  execFileSync("oiiotool", [
    pfm,
    "--mulc",
    String(2 ** exposure),
    "--clamp:min=0:max=1",
    "--colorconvert",
    "linear",
    "sRGB",
    "-d",
    "uint8",
    "-o",
    png,
  ]);
}
export function diffImage(x, r, file, gain = 4) {
  const d = new Float32Array(x.f.length);
  for (let i = 0; i < x.w * x.h; i++) {
    let e = 0;
    for (let c = 0; c < 3; c++) {
      const a = cl(x.f[3 * i + c]);
      const b = cl(r.f[3 * i + c]);
      e += ((a - b) * (a - b)) / (b * b + 0.01);
    }
    e = Math.min(1, (Math.sqrt(e / 3) * gain) / 4);
    // black -> red -> yellow -> white
    d[3 * i] = Math.min(1, 3 * e);
    d[3 * i + 1] = Math.min(1, Math.max(0, 3 * e - 1));
    d[3 * i + 2] = Math.max(0, 3 * e - 2);
  }
  writePfm(file, x.w, x.h, d);
}

if (process.argv[2]) {
  const spec = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
  fs.mkdirSync(spec.out, { recursive: true });
  const rows = [];
  for (const view of spec.views) {
    const refFile = spec.refFile ?? `${spec.ref}/${view}.pfm`;
    const r = readPfm(refFile);
    const masks = {};
    if (spec.car?.bgDir)
      masks.car = diffMask(r, readPfm(`${spec.car.bgDir}/${view}.bg.pfm`));
    if (spec.car?.alpha) {
      const a = readPfm(spec.car.alpha);
      masks.car = Uint8Array.from({ length: a.w * a.h }, (_, i) =>
        a.f[3 * i] > 0.99 ? 1 : 0,
      );
    }
    if (spec.car?.maskFrom) {
      const ref2 = readPfm(`${spec.car.maskFrom}/${view}.pfm`);
      masks.car = diffMask(
        ref2,
        readPfm(`${spec.car.maskFrom}/${view}.bg.pfm`),
      );
    }
    for (const [g, pat] of Object.entries(spec.parts ?? {})) {
      const full = readPfm(
        `${pat.dir}/${view}.pfm`.replace(pat.dir, pat.fullDir ?? pat.dir),
      );
      const no = readPfm(`${pat.dir}/${view}.no-${g}.pfm`);
      const m = diffMask(full, no, 0.05, 0.01);
      if (masks.car) for (let i = 0; i < m.length; i++) m[i] &= masks.car[i];
      masks[g] = m;
    }
    if (spec.car?.unionParts) {
      const u = new Uint8Array(r.w * r.h);
      for (const g of Object.keys(spec.parts))
        for (let i = 0; i < u.length; i++) u[i] |= masks[g][i];
      masks.car = u;
    }
    if (spec.lamps) {
      const on = readPfm(`${spec.lamps.onDir}/${view}.pfm`);
      const off = readPfm(`${spec.lamps.offDir}/${view}.pfm`);
      const m = new Uint8Array(on.w * on.h);
      for (let i = 0; i < m.length; i++) {
        const d = lum(on.f, i) - lum(off.f, i);
        m[i] = d > Math.max(0.02, lum(off.f, i)) ? 1 : 0;
      }
      masks.lamps = m;
    }
    masks.frame = null;
    if (spec.previews !== false)
      preview(
        refFile,
        `${spec.out}/${view}-${spec.refLabel ?? "ref"}.png`,
        spec.exposure ?? 0,
      );
    for (const [label, dir] of Object.entries(spec.cands)) {
      const x = readPfm(`${dir}/${view}.pfm`);
      const row = { view, cand: label };
      for (const [mn, m] of Object.entries(masks)) row[mn] = metrics(x, r, m);
      rows.push(row);
      if (spec.previews !== false) {
        preview(
          `${dir}/${view}.pfm`,
          `${spec.out}/${view}-${label}.png`,
          spec.exposure ?? 0,
        );
        diffImage(x, r, `${spec.out}/${view}-${label}-diff.pfm`);
        execFileSync("oiiotool", [
          `${spec.out}/${view}-${label}-diff.pfm`,
          "-d",
          "uint8",
          "-o",
          `${spec.out}/${view}-${label}-diff.png`,
        ]);
        fs.rmSync(`${spec.out}/${view}-${label}-diff.pfm`);
      }
    }
  }
  fs.writeFileSync(
    `${spec.out}/metrics-${spec.tag}.json`,
    JSON.stringify(rows, null, 1),
  );
  const fmt = (v) => (v == null ? "-" : v.toExponential(2));
  const keys = Object.keys(rows[0]).filter((k) => k !== "view" && k !== "cand");
  console.log(
    [
      "view",
      "cand",
      ...keys.map((k) => `${k} relMSE/trim999 (ratio, flies)`),
    ].join(" | "),
  );
  for (const r of rows)
    console.log(
      [
        r.view,
        r.cand,
        ...keys.map(
          (k) =>
            `${fmt(r[k].relMSE)}/${fmt(r[k].trim)} (${r[k].ratio?.toFixed(3)}, ${r[k].flies})`,
        ),
      ].join(" | "),
    );
}
