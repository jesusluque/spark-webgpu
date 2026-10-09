// node run-pawn.mjs <outdir> <query> <views comma list|all> [--notop]
// Thread BL: the relit pawn (examples/webgpu/validate-pawn.html, the public
// page's scene) at 1920x1080, each view read back as linear float (PFM, RGB)
// before the fx chain; --notop also renders every view with the glass head
// hidden (<view>.notop.pfm: a mask of the head). Served by serve.mjs, the
// pawn's files under /r2/pawn/ (VALIDATE_PORT, default 8121).
import fs from "node:fs";
import { chromium } from "playwright";

const [outdir, query = "", viewArg = "all", ...flags] = process.argv.slice(2);
const T = [0, 0.048, 0];
const at = (z, y = 0.048) => ({ position: [0, y, z], target: T });
export const VIEWS = {
  default: at(0.18), // pawn.html's /World/Cam
  half: at(0.5),
  m1: at(1.0),
  far: at(3.0),
  below: { position: [0, 0.012, 0.11], target: [0, 0.07, 0] },
};
const names = viewArg === "all" ? Object.keys(VIEWS) : viewArg.split(",");
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
  viewport: { width: 1920, height: 1080 },
  deviceScaleFactor: 1,
});
const log = [];
page.on("console", (m) => {
  if (m.type() === "error" || m.type() === "warning")
    log.push(`${m.type()}: ${m.text()}`);
});
page.on("pageerror", (e) => log.push(`pageerror: ${e.message}`));
const url = `http://127.0.0.1:${process.env.VALIDATE_PORT ?? 8121}/examples/webgpu/validate-pawn.html?${query}`;
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
const result = { query, loadS: (Date.now() - t0) / 1000, views: {}, log };
async function grab(name, file, settle) {
  const { width, height, data, stats } = await page.evaluate(
    ([v, s]) => window.__bd.capture(v, s),
    [VIEWS[name], settle],
  );
  const f = new Float32Array(Buffer.from(data, "base64").buffer.slice(0));
  const rgb = Buffer.alloc(width * height * 12);
  let nonFinite = 0;
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++)
      for (let c = 0; c < 3; c++) {
        let v = f[((height - 1 - y) * width + x) * 4 + c];
        if (!Number.isFinite(v)) {
          nonFinite++;
          v = 0;
        }
        rgb.writeFloatLE(v, (y * width + x) * 12 + c * 4);
      }
  fs.writeFileSync(
    `${outdir}/${file}`,
    Buffer.concat([Buffer.from(`PF\n${width} ${height}\n-1.0\n`), rgb]),
  );
  return { width, height, nonFinite, stats };
}
for (const name of names) {
  result.views[name] = await grab(name, `${name}.pfm`, 40);
  console.log(name, JSON.stringify(result.views[name]));
}
if (flags.includes("--notop")) {
  await page.evaluate(() => window.__bd.hideTop(true));
  for (const name of names) await grab(name, `${name}.notop.pfm`, 15);
}
fs.writeFileSync(`${outdir}/run.json`, JSON.stringify(result, null, 1));
console.log("load", result.loadS, "s; log lines", log.length);
await browser.close();
