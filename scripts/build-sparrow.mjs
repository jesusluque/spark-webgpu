#!/usr/bin/env node
// The sparrow demo's data in one command (examples/webgpu-site/sparrow.html):
// athenea's skinned, relightable sparrow cloud (a .usdc with a zonal
// transfer, `athenea mesh2splat --skinned --transfer`) to a .athc v3, and a
// set of athenea's animation clips (`<clip>_rig.usda`, each the cloud's
// `skinningXforms` over time) to one gzipped ATCL file a clip, fetched by
// the page when it plays that clip; then sparrow.json with the camera, the
// dome and the clip list.
//
//   node scripts/build-sparrow.mjs [--cloud sparrow.usdc] [--clips-dir DIR]
//        [--scene scene.json] [--skeleton SparrowAir.usdc] [--out DIR]
//        [--clips a,b,c] [--usd-athc path]
//
// Defaults: the stand-in cloud (athenea-skin-data/sparrow160tx.usdc), the
// rigs athenea delivered (~/luc/athenea-renders/sparrow-web/clips, its
// scene.json), output in publish-r2/sparkwebgpu/sparrow/. When athenea's
// new cloud lands: --cloud ~/luc/athenea-renders/sparrow-web/sparrow_tx.usdc.
// File names carry a hash of their bytes (R2 caches them); sparrow.json is
// the one name the page knows.

import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const project = path.resolve(repo, "..");
const home = os.homedir();
const args = process.argv.slice(2);
const arg = (name, fallback) => {
  const k = args.indexOf(name);
  return k >= 0 ? args[k + 1] : fallback;
};

// Loopable clips (first pose = last pose; usd-athc reports the seam), in
// the order the page lists them, with their labels.
const CLIPS = [
  ["air_fly_A0", "Flying"],
  ["air_gliding_A0", "Gliding"],
  ["air_fly_turnL_A0", "Flying, turning left"],
  ["air_fly_turnR_A0", "Flying, turning right"],
  ["air_fly_revolve_A0", "Flying, rolling"],
  ["land_idle_B1", "On the ground, idle"],
  ["land_eat_B1", "Pecking"],
  ["land_hop_1", "Hopping"],
  ["land_jump_A", "Jumping"],
  ["treex_idle_A1", "On a branch, idle"],
  ["treex_side_stepL_A0", "On a branch, side-stepping"],
  ["treey_hop_forward_B1", "On a branch, hopping forward"],
];

const cloud = arg(
  "--cloud",
  path.join(project, "athenea-skin-data/sparrow160tx.usdc"),
);
const delivery = path.join(home, "luc/athenea-renders/sparrow-web");
const clipsDir = arg("--clips-dir", path.join(delivery, "clips"));
const scenePath = arg("--scene", path.join(delivery, "scene.json"));
const skeleton = arg(
  "--skeleton",
  path.join(home, "tools/assets/Sparrow/SparrowAir.usdc"),
);
const out = arg("--out", path.join(project, "publish-r2/sparkwebgpu/sparrow"));
const wanted = arg("--clips")?.split(",") ?? CLIPS.map((c) => c[0]);
const labels = Object.fromEntries(CLIPS);

let usdAthc = arg("--usd-athc");
if (!usdAthc) {
  const env = {
    ...process.env,
    PATH: `/opt/homebrew/opt/rustup/bin:${process.env.PATH}`,
  };
  execFileSync(
    "cargo",
    ["build", "--release", "-p", "build-lod", "--bin", "usd-athc"],
    { cwd: path.join(repo, "rust"), stdio: "inherit", env },
  );
  usdAthc = path.join(repo, "rust/target/release/usd-athc");
}

const work = fs.mkdtempSync(path.join(os.tmpdir(), "sparrow-"));
const report = path.join(work, "report.json");
execFileSync(
  usdAthc,
  [
    cloud,
    path.join(work, "sparrow.athc"),
    "--clip",
    "own",
    "--clip-dir",
    clipsDir,
    "--clips",
    wanted.join(","),
    "--clip-skeleton",
    skeleton,
    "--drop-own-clip",
    "--clip-files",
    path.join(work, "clips"),
    "--gzip",
    "--json",
    report,
  ],
  { stdio: ["ignore", "ignore", "inherit"] },
);
const r = JSON.parse(fs.readFileSync(report, "utf8"));

const hashed = (file, name, ext) => {
  const bytes = fs.readFileSync(file);
  const h = createHash("sha256").update(bytes).digest("hex").slice(0, 10);
  return { name: `${name}-${h}${ext}`, bytes };
};
// Earlier builds' hashed files go (hdri/ and anything else stay).
fs.mkdirSync(path.join(out, "clips"), { recursive: true });
for (const f of fs.readdirSync(out)) {
  if (/^sparrow-[0-9a-f]+\.athc$/.test(f)) fs.rmSync(path.join(out, f));
}
for (const f of fs.readdirSync(path.join(out, "clips"))) {
  if (f.endsWith(".atcl.gz")) fs.rmSync(path.join(out, "clips", f));
}
const athc = hashed(path.join(work, "sparrow.athc"), "sparrow", ".athc");
fs.writeFileSync(path.join(out, athc.name), athc.bytes);

const clips = r.skin.clips.map((c) => {
  const f = hashed(path.join(work, "clips", c.file), c.name, ".atcl.gz");
  fs.writeFileSync(path.join(out, "clips", f.name), f.bytes);
  return {
    name: c.name,
    label: labels[c.name] ?? c.name,
    file: `clips/${f.name}`,
    bytes: f.bytes.length,
    from: c.from,
    to: c.to,
    timeCodesPerSecond: c.timeCodesPerSecond,
    seconds: c.seconds,
    // A seam this small is a loop (the clips athenea marks _A0 / _B1 ...).
    loops: c.seamRotation < 0.01 && c.seamTranslation < 0.001,
    rootTravel: c.rootTravel,
    rootExcursion: c.rootExcursion,
  };
});

// athenea's camera (Z-up stage, metres) and dome.
const scene = JSON.parse(fs.readFileSync(scenePath, "utf8"));
const cam = scene.camera;
const vfov =
  (2 * Math.atan(cam.aperture_mm[1] / 2 / cam.focalLength_mm) * 180) / Math.PI;
const hdri = `${path.basename(scene.bake?.hdri ?? "san_giuseppe_bridge.hdr", ".hdr")}_4k.hdr`;
const info = {
  title: "sparrow",
  upAxis: r.upAxis ?? scene.upAxis ?? "Z",
  cloud: athc.name,
  cloudBytes: athc.bytes.length,
  splats: r.splats,
  sourceSplats: r.sourceSplats,
  boundsMin: r.boundsMin,
  boundsMax: r.boundsMax,
  joints: r.skin.joints,
  // Stage coordinates (upAxis), as athenea's /Ref/Camera.
  camera: {
    eye: cam.eye,
    target: cam.target,
    up: cam.up,
    verticalFov: vfov,
    near: 0.005,
    far: 60,
  },
  hdri: [hdri],
  hdriDefault: hdri,
  // Phones: the 1k domes next to the cloud (hdri/, made by hand: Poly
  // Haven's 1k files, and the bake's dome resized), the bake's first.
  hdriMobile: fs.existsSync(path.join(out, "hdri"))
    ? fs
        .readdirSync(path.join(out, "hdri"))
        .filter((f) => f.endsWith("_1k.hdr"))
        .sort((a, b) =>
          a.startsWith(hdri.replace("_4k.hdr", ""))
            ? -1
            : b.startsWith(hdri.replace("_4k.hdr", ""))
              ? 1
              : a.localeCompare(b),
        )
    : [],
  domeRotation: 0,
  defaultClip: clips[0].name,
  clips,
  built: {
    date: new Date().toISOString(),
    cloud: path.basename(cloud),
    clipsDir: path.relative(home, clipsDir),
  },
};
fs.writeFileSync(
  path.join(out, "sparrow.json"),
  `${JSON.stringify(info, null, 1)}\n`,
);
fs.rmSync(work, { recursive: true, force: true });

const mb = (b) => (b / 1e6).toFixed(2);
const clipBytes = clips.reduce((s, c) => s + c.bytes, 0);
console.log(
  `${out}: ${athc.name} ${mb(athc.bytes.length)} MB (${r.splats} splats), ` +
    `${clips.length} clips ${mb(clipBytes)} MB`,
);
for (const c of clips) {
  console.log(
    `  ${c.name.padEnd(24)} ${c.seconds.toFixed(2)} s  ${mb(c.bytes)} MB  ${c.loops ? "loops" : "NO LOOP"}`,
  );
}
