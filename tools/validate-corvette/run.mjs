// node run.mjs <set> <outdir> <query> <views comma list|all> [--time]
// Loads /r2/<set>/ in validate-corvette.html at 1920x1080, captures each view
// as linear float (PFM, RGB) and optionally times frames.
import fs from "node:fs";
import { chromium } from "playwright";

const [set, outdir, query = "", viewArg = "all", ...flags] =
  process.argv.slice(2);
const T = [0, 0.553, 0.252];
const AWAY = [4.378, 1.123, 5.602];
const len = Math.hypot(...AWAY);
const along = (k) => T.map((t, i) => t + AWAY[i] * k);
const orbit = (azDeg, elDeg, d, t = T) => {
  const az = (azDeg * Math.PI) / 180;
  const el = (elDeg * Math.PI) / 180;
  return [
    t[0] + d * Math.cos(el) * Math.cos(az),
    t[1] + d * Math.sin(el),
    t[2] + d * Math.cos(el) * Math.sin(az),
  ];
};
export const VIEWS = {
  close: { position: along(1.0), target: T }, // athenea's /World/Camera
  default: { position: along(2.2), target: T }, // the light page's default
  side: { position: orbit(0, 6, 6.0), target: T },
  glass: { position: orbit(140, 25, 6.0), target: T },
  rear: { position: orbit(-130, 12, 6.5), target: T },
  far: { position: along(4.0), target: T },
  // athenea's corvette_scene_night.usda /World/CameraRear (45 mm over 36 mm),
  // Z-up turned to Y-up: (x, y, z) -> (x, z, -y).
  athrear: {
    position: [3.2, 1.45, -6.4],
    target: [-0.8245, 0.5697, -0.7406],
    vfov: (2 * Math.atan(0.10125 / 0.45) * 180) / Math.PI,
  },
};
// farS0..farS7 (thread BO): the far view moved sideways 3 mm a step (about
// a quarter of a pixel at the car), camera and target together: temporal
// stability over a small camera move.
const SIDE = [
  -AWAY[2] / Math.hypot(AWAY[0], AWAY[2]),
  0,
  AWAY[0] / Math.hypot(AWAY[0], AWAY[2]),
];
for (let k = 0; k < 8; k++) {
  const t = T.map((v, i) => v + SIDE[i] * 0.003 * k);
  VIEWS[`farS${k}`] = {
    position: along(4.0).map((v, i) => v + SIDE[i] * 0.003 * k),
    target: t,
  };
}
const names =
  viewArg === "all"
    ? Object.keys(VIEWS).filter((v) => v !== "athrear" && !v.startsWith("farS"))
    : viewArg.split(",");
// VALIDATE_SIZE=1280x720: athenea's EXRs' size (default 1920x1080).
const [W, H] = (process.env.VALIDATE_SIZE ?? "1920x1080")
  .split("x")
  .map(Number);
fs.mkdirSync(outdir, { recursive: true });

const browser = await chromium.launch({
  channel: "chrome",
  headless: true,
  args: [
    "--enable-unsafe-webgpu",
    "--enable-dawn-features=allow_unsafe_apis",
    "--disable-gpu-vsync",
  ],
});
const page = await browser.newPage({
  viewport: { width: W, height: H },
  deviceScaleFactor: 1,
});
const log = [];
page.on("console", (m) => {
  if (m.type() === "error" || m.type() === "warning")
    log.push(`${m.type()}: ${m.text()}`);
});
page.on("pageerror", (e) => log.push(`pageerror: ${e.message}`));
const url = `http://127.0.0.1:${process.env.VALIDATE_PORT ?? 8121}/examples/webgpu/validate-corvette.html?base=/r2/${set}/&hdriBase=/r2/hdri-4k/&gui=0&bloom=0&${query}`;
const t0 = Date.now();
await page.goto(url);
await page.waitForFunction(
  () => window.__bd?.ready || window.__athenea?.error,
  null,
  { timeout: 400000 },
);
const err = await page.evaluate(() => window.__athenea.error);
if (err) {
  console.log("ERROR", err, log.join("\n"));
  await browser.close();
  process.exit(1);
}
const loadS = (Date.now() - t0) / 1000;
const result = { set, query, loadS, views: {}, log };
await page.evaluate((v) => window.__bd.setView(v, 90), VIEWS[names[0]]);
for (const name of names) {
  const { width, height, data } = await page.evaluate(
    (v) => window.__bd.capture(v, 40),
    VIEWS[name],
  );
  const f = new Float32Array(Buffer.from(data, "base64").buffer.slice(0));
  // PFM: bottom row first, RGB, little endian (scale -1).
  const rgb = Buffer.alloc(width * height * 12);
  let nonFinite = 0;
  for (let y = 0; y < height; y++) {
    const sy = height - 1 - y;
    for (let x = 0; x < width; x++) {
      for (let c = 0; c < 3; c++) {
        let v = f[(sy * width + x) * 4 + c];
        if (!Number.isFinite(v)) {
          nonFinite++;
          v = 0;
        }
        rgb.writeFloatLE(v, (y * width + x) * 12 + c * 4);
      }
    }
  }
  fs.writeFileSync(
    `${outdir}/${name}.pfm`,
    Buffer.concat([Buffer.from(`PF\n${width} ${height}\n-1.0\n`), rgb]),
  );
  result.views[name] = { width, height, nonFinite };
  // The splats generated and drawn for the view (LoD switching, thread BS).
  result.views[name].stats = await page.evaluate(() => {
    const st = window.__athenea.spark?.webgpu?.splats?.stats;
    return st ? { drawn: st.drawn } : null;
  });
  if (
    flags.includes("--time") &&
    (name === "close" || name === "far" || name === "default")
  ) {
    result.views[name].time = await page.evaluate(() => window.__bd.time(40));
  }
  console.log(name, JSON.stringify(result.views[name]).slice(0, 400));
}
if (flags.includes("--bg")) {
  await page.evaluate(() => window.__bd.hide(true));
  for (const name of names) {
    const { width, height, data } = await page.evaluate(
      (v) => window.__bd.capture(v, 15),
      VIEWS[name],
    );
    const f = new Float32Array(Buffer.from(data, "base64").buffer.slice(0));
    const rgb = Buffer.alloc(width * height * 12);
    for (let y = 0; y < height; y++)
      for (let x = 0; x < width; x++)
        for (let c = 0; c < 3; c++) {
          const v = f[((height - 1 - y) * width + x) * 4 + c];
          rgb.writeFloatLE(
            Number.isFinite(v) ? v : 0,
            (y * width + x) * 12 + c * 4,
          );
        }
    fs.writeFileSync(
      `${outdir}/${name}.bg.pfm`,
      Buffer.concat([Buffer.from(`PF\n${width} ${height}\n-1.0\n`), rgb]),
    );
  }
}
if (flags.includes("--masks")) {
  const GROUPS = {
    paint: ["paint"],
    body: ["body"],
    glass: ["windshield", "tinted", "headlights"],
    trim: ["trim"],
  };
  for (const [g, parts] of Object.entries(GROUPS)) {
    await page.evaluate((ps) => {
      for (const [n, m] of Object.entries(window.__athenea.meshes))
        m.visible = !ps.includes(n);
    }, parts);
    for (const name of names) {
      const { width, height, data } = await page.evaluate(
        (v) => window.__bd.capture(v, 15),
        VIEWS[name],
      );
      const f = new Float32Array(Buffer.from(data, "base64").buffer.slice(0));
      const rgb = Buffer.alloc(width * height * 12);
      for (let y = 0; y < height; y++)
        for (let x = 0; x < width; x++)
          for (let c = 0; c < 3; c++) {
            const v = f[((height - 1 - y) * width + x) * 4 + c];
            rgb.writeFloatLE(
              Number.isFinite(v) ? v : 0,
              (y * width + x) * 12 + c * 4,
            );
          }
      fs.writeFileSync(
        `${outdir}/${name}.no-${g}.pfm`,
        Buffer.concat([Buffer.from(`PF\n${width} ${height}\n-1.0\n`), rgb]),
      );
    }
  }
  await page.evaluate(() => {
    for (const m of Object.values(window.__athenea.meshes)) m.visible = true;
  });
}
const mem = await page.evaluate(() => ({
  splats: window.__athenea.spark?.webgpu?.splats?.stats?.numSplats,
}));
result.mem = mem;
fs.writeFileSync(`${outdir}/run.json`, JSON.stringify(result, null, 1));
console.log("load", loadS, "s; log lines", log.length);
await browser.close();
