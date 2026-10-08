#!/usr/bin/env node
// The Corvette with switchable lamps in one command (corvette.html?set=lights):
// athenea's base cloud (TX with lobes and curvature, every lamp off, dome
// autoshop_01) and its per-group layer clouds (the same gaussians in the same
// order, each baked with only that group lit at radiance 1 and a black dome:
// their linear colours are the group's light per unit of its radiance) to
//
//   corvette-lights-hd/      the detailed set: every splat of the base
//                            (usd-athc --drop-backs), .athc + .lights.athl a part
//   corvette-lights-light/   about --light-splats splats: the detailed LoD tree
//                            cut (athc-convert --keep-splats --creases), the
//                            .athl carried through the same cut
//
// each with corvette.json (camera, domes, floor and the parts, from the live
// light set's), the sidecar (corvette.lights.usda) and the 1k domes for phones.
//
//   node scripts/build-corvette-lights.mjs [--src DIR] [--out DIR]
//        [--transfer 16] [--drop-backs 0.008] [--light-splats 1000000]
//        [--creases auto|D] [--threshold 1e-4] [--floor 1e-4]
//        [--part NAME=TEXT,TEXT[@IOR][!catcher]]... [--only hd|light]
//        [--sidecar file.lights.usda] [--template corvette.json] [--list]
//        [--no-catcher] [--layer GROUP=other.usdc]... [--ground-exclude a,b]
//        [--ground-floor 1e-4]
//
// Defaults: athenea's delivery in ~/luc/athenea-renders/corvette-lights
// (clouds/base_tx.usdc, clouds/layer_<group>.usdc; the sidecar there or
// ~/tools/assets/CorvetteC7/usd/corvette.lights.usda), output in
// publish-r2/sparkwebgpu/ (outside git). One part, "car", by default; --part
// splits the base by the prims its cryptoManifest names (a part keeps the
// splats of prims whose path contains one of the TEXTs; the last part with
// no TEXT, `--part rest=`, takes what no other part took), e.g. to give glass
// its index (@1.16) or to mark the floor catcher (!catcher). --list prints
// the base's prims and stops. Every part's .athl goes through the same
// selection as its cloud (usd-athc --light-layer), so the layers line up.
// --floor zeroes the splats whose layer is fainter than that (radiance
// units, after the group's luminance): athenea's grainy faint indirect
// (the second bake, detailed: 21 MB at 1e-4; at 1e-2 it was 4.75 MB but lost
// the faint lamp light on the body that shows under athenea's night dome of 0.005). --creases auto (the
// default) finds the smallest crease threshold whose cut stays within 10 %
// of --light-splats (this car's thin two-sided parts spread their normals
// everywhere: 0.03 kept 1.7M of 2.8M).
//
// The lamps' light on the floor: an additive cloud, athenea's
// clouds/ground_<group>.usdc (or catcher_layer_<group>.usdc) over the
// geometry clouds/ground_base.usdc (black), built as a part of its own with
// its .athl and `additive: true` in corvette.json (no cut: the same in both
// sets); --ground-exclude leaves groups out of it, --ground-floor its floor.
// --layer GROUP=PATH takes a group's car layer from elsewhere (an earlier
// bake: clouds/glow/layer_<group>.usdc).
// File names carry a hash of their bytes (R2 caches them); corvette.json
// is the one name the page knows.

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
const all = (name) =>
  args.flatMap((a, k) => (a === name && args[k + 1] ? [args[k + 1]] : []));

const GROUPS = ["cruce", "largas", "drl", "pilotos", "tubo_trasero"];
const src = arg(
  "--src",
  path.join(home, "luc/athenea-renders/corvette-lights"),
);
const out = arg("--out", path.join(project, "publish-r2/sparkwebgpu"));
const transfer = arg("--transfer", "16");
const dropBacks = arg("--drop-backs", "0.008");
const lightSplats = Number(arg("--light-splats", "1000000"));
const creasesArg = arg("--creases", "auto");
const threshold = arg("--threshold", "1e-4");
const floor = arg("--floor", "1e-4");
const only = arg("--only", null);
const template = arg(
  "--template",
  path.join(project, "publish-r2/sparkwebgpu/corvette-v5-light/corvette.json"),
);

const base = path.join(src, "clouds/base_tx.usdc");
// --layer NAME=PATH: that group's layer from elsewhere (an earlier bake).
const layerOverride = Object.fromEntries(
  all("--layer").map((v) => v.split("=")),
);
const layers = GROUPS.map((g) => [
  g,
  layerOverride[g] ?? path.join(src, `clouds/layer_${g}.usdc`),
]);
const groundExclude = (arg("--ground-exclude", "") ?? "")
  .split(",")
  .filter(Boolean);
const groundFloor = arg("--ground-floor", "1e-4");
const sidecar =
  arg("--sidecar", null) ??
  [
    ...fs
      .readdirSync(src)
      .filter((f) => f.endsWith(".lights.usda"))
      .map((f) => path.join(src, f)),
    path.join(home, "tools/assets/CorvetteC7/usd/corvette.lights.usda"),
  ].find((f) => fs.existsSync(f));
const missing = [base, ...layers.map(([, f]) => f), sidecar, template].filter(
  (f) => !f || !fs.existsSync(f),
);
if (missing.length) {
  console.error(`missing:\n  ${missing.join("\n  ")}`);
  process.exit(1);
}

// The tools, built once (release).
const rust = path.join(repo, "rust");
execFileSync(
  "cargo",
  [
    "build",
    "--release",
    "-p",
    "build-lod",
    "--bin",
    "usd-athc",
    "--bin",
    "athc-convert",
  ],
  { cwd: rust, stdio: "inherit" },
);
const bin = (name) => path.join(rust, "target/release", name);
const run = (tool, argv) => {
  console.log(`$ ${tool} ${argv.join(" ")}`);
  return execFileSync(bin(tool), argv, {
    stdio: ["ignore", "pipe", "inherit"],
    maxBuffer: 1 << 28,
  }).toString();
};

if (args.includes("--list")) {
  const listing = run("usd-athc", [base, "--list"]);
  const manifest = listing.match(/^cryptoManifest: (.*)$/m);
  console.log(listing.replace(/^cryptoManifest: .*$/m, ""));
  if (manifest)
    for (const p of Object.keys(JSON.parse(manifest[1]))) console.log(`  ${p}`);
  process.exit(0);
}

// Parts: NAME=TEXT,TEXT[@IOR][!catcher]; the default one part, "car".
const parts = (all("--part").length ? all("--part") : ["car="]).map((spec) => {
  const [name, rest = ""] = spec.split("=");
  const catcher = rest.includes("!catcher");
  const [texts, ior] = rest.replace("!catcher", "").split("@");
  return {
    name,
    texts: texts.split(",").filter(Boolean),
    ior: ior ? Number(ior) : undefined,
    catcher,
  };
});
const claimed = parts.flatMap((p) => p.texts);
const selection = (p) =>
  p.texts.length
    ? p.texts.flatMap((t) => ["--only-prim", t])
    : claimed.flatMap((t) => ["--exclude-prim", t]);

const short = (file) =>
  createHash("sha256").update(fs.readFileSync(file)).digest("hex").slice(0, 10);
const hashed = (dir, file, stem, ext) => {
  const name = `${stem}-${short(file)}${ext}`;
  fs.copyFileSync(file, path.join(dir, name));
  return name;
};
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "corvette-lights-"));
const layerArgs = layers.flatMap(([g, f]) => ["--light-layer", `${g}=${f}`]);
const t = `t${transfer}`;

// The detailed set: usd-athc (selection, layers through it), then the
// smallest encoding by section, the .athl re-stamped for that file.
const hdDir = path.join(out, "corvette-lights-hd");
const lightDir = path.join(out, "corvette-lights-light");
const built = [];
for (const p of parts) {
  const raw = path.join(tmp, `${p.name}.athc`);
  const rawAthl = path.join(tmp, `${p.name}.lights.athl`);
  const json = path.join(tmp, `${p.name}.json`);
  run("usd-athc", [
    base,
    raw,
    "--transfer",
    transfer,
    "--drop-backs",
    dropBacks,
    ...selection(p),
    ...layerArgs,
    "--lights-usda",
    sidecar,
    "--athl",
    rawAthl,
    "--light-threshold",
    threshold,
    "--light-floor",
    floor,
    "--json",
    json,
  ]);
  built.push({
    ...p,
    raw,
    rawAthl,
    report: JSON.parse(fs.readFileSync(json, "utf8")),
  });
}
// The lamps' light on the floor (athenea's ground_base.usdc, black, and
// ground_<group>.usdc; or catcher_layer_<group>.usdc): an additive cloud.
const catcherLayers = GROUPS.filter((g) => !groundExclude.includes(g))
  .map((g) => [
    g,
    [`ground_${g}.usdc`, `catcher_layer_${g}.usdc`]
      .map((f) => path.join(src, "clouds", f))
      .find((f) => fs.existsSync(f)),
  ])
  .filter(([, f]) => f);
if (catcherLayers.length) {
  const geometry =
    ["ground_base.usdc", "catcher.usdc", "catcher_base.usdc"]
      .map((f) => path.join(src, "clouds", f))
      .find((f) => fs.existsSync(f)) ?? catcherLayers[0][1];
  const raw = path.join(tmp, "lamp-floor.athc");
  const rawAthl = path.join(tmp, "lamp-floor.lights.athl");
  const json = path.join(tmp, "lamp-floor.json");
  run("usd-athc", [
    geometry,
    raw,
    "--transfer",
    "none",
    "--no-material",
    "--no-curvature",
    ...catcherLayers.flatMap(([g, f]) => ["--light-layer", `${g}=${f}`]),
    "--lights-usda",
    sidecar,
    "--athl",
    rawAthl,
    "--light-threshold",
    threshold,
    "--light-floor",
    groundFloor,
    "--json",
    json,
  ]);
  built.push({
    name: "lamp-floor",
    texts: [],
    additive: true,
    raw,
    rawAthl,
    report: JSON.parse(fs.readFileSync(json, "utf8")),
  });
}
const totalHd = built
  .filter((p) => !p.additive)
  .reduce((s, p) => s + p.report.splats, 0);

const info = JSON.parse(fs.readFileSync(template, "utf8"));
function writeSet(dir, quality, entries) {
  fs.mkdirSync(dir, { recursive: true });
  for (const f of fs.readdirSync(dir)) {
    if (/\.(athc|athl)$/.test(f)) fs.rmSync(path.join(dir, f));
  }
  const parts = entries.map((e) => {
    const file = hashed(dir, e.athc, `${e.name}-${t}-gz`, ".athc");
    const athl = hashed(dir, e.athl, `${e.name}`, ".lights.athl");
    const bytes = fs.statSync(path.join(dir, file)).size;
    const athlBytes = fs.statSync(path.join(dir, athl)).size;
    return {
      name: e.name,
      label: e.additive ? "lamp light on the floor" : e.name,
      file,
      bytes,
      splats: e.splats,
      athl,
      athlBytes,
      transferCount: Number(transfer),
      lobesWords: e.report.lobesWords,
      curvature: e.report.curvature,
      droppedBacks: e.report.droppedBacks,
      ...(e.ior ? { ior: e.ior } : {}),
      ...(e.catcher ? { catcher: true } : {}),
      ...(e.additive ? { additive: true } : {}),
      // The detailed build's (usd-athc) per-group stats.
      lights: e.report.lights.groups.map((g) => ({
        group: g.group,
        radiance: g.radiance,
        litSplats: g.litSplats,
        flooredSplats: g.flooredSplats,
        blocks: g.blocks,
        maxPositionError: g.maxPositionError,
      })),
      origin: e.origin,
    };
  });
  // athenea's lights base has no floor: the live set's shadow catcher (a
  // bake of the same car under the same dome), unlit by the lamps.
  const liveSet = path.join(
    out,
    quality === "light" ? "corvette-v5-light" : "corvette-v2-hd",
  );
  const liveInfo = path.join(liveSet, "corvette.json");
  if (
    !args.includes("--no-catcher") &&
    !parts.some((p) => p.catcher) &&
    fs.existsSync(liveInfo)
  ) {
    const c = JSON.parse(fs.readFileSync(liveInfo, "utf8")).parts.find(
      (p) => p.catcher,
    );
    if (c) {
      const file = hashed(
        dir,
        path.join(liveSet, c.file),
        "catcher-t16-gz",
        ".athc",
      );
      parts.push({
        ...c,
        file,
        origin: `${c.origin ?? ""} (from ${path.basename(liveSet)}; not lit by the lamps)`,
      });
    }
  }
  const sidecarName = "corvette.lights.usda";
  // The published copy names no local folders.
  fs.writeFileSync(
    path.join(dir, sidecarName),
    fs
      .readFileSync(sidecar, "utf8")
      .replaceAll(`${os.homedir()}/luc/`, "")
      .replaceAll("~/luc/", ""),
  );
  // The phones' 1k domes and the credits, as the live sets have them.
  const live = path.dirname(template);
  if (fs.existsSync(path.join(live, "hdri"))) {
    fs.cpSync(path.join(live, "hdri"), path.join(dir, "hdri"), {
      recursive: true,
    });
  }
  if (fs.existsSync(path.join(live, "CREDITS.txt"))) {
    fs.copyFileSync(
      path.join(live, "CREDITS.txt"),
      path.join(dir, "CREDITS.txt"),
    );
  }
  const scene = {
    ...info,
    title: `Corvette C7 Stingray, relightable, with switchable lamps (${quality})`,
    bakeSky: info.bakeSky,
    lights: sidecarName,
    lightsState: "noche_ciudad",
    splats: parts.reduce((s, p) => s + p.splats, 0),
    bytes: parts.reduce((s, p) => s + p.bytes + (p.athlBytes ?? 0), 0),
    parts,
    lightRecipe: `athenea's lights bake (${path.basename(src)}): base_tx.usdc + layer_<group>.usdc (${GROUPS.join(", ")}), scripts/build-corvette-lights.mjs --transfer ${transfer} --drop-backs ${dropBacks} --threshold ${threshold} --floor ${floor}${Object.keys(
      layerOverride,
    )
      .map((g) => ` --layer ${g}=${path.relative(src, layerOverride[g])}`)
      .join(
        "",
      )}${groundExclude.length ? ` --ground-exclude ${groundExclude}` : ""}${quality === "light" ? ` --light-splats ${lightSplats} --creases ${creasesArg}` : ""}`,
  };
  fs.writeFileSync(
    path.join(dir, "corvette.json"),
    `${JSON.stringify(scene, null, 2)}\n`,
  );
  return scene;
}

const summary = {};
const hdEntries = [];
for (const p of built) {
  // The smallest encoding a section, the layers re-stamped for the bytes.
  const athc = path.join(tmp, `${p.name}-hd.athc`);
  const athl = path.join(tmp, `${p.name}-hd.lights.athl`);
  run("athc-convert", [
    p.raw,
    athc,
    "--gzip",
    "--planes",
    "--athl",
    p.rawAthl,
    "--athl-out",
    athl,
  ]);
  hdEntries.push({
    ...p,
    athc,
    athl,
    splats: p.report.splats,
    origin: p.additive
      ? "athenea's lamps' light on the floor, an additive cloud (its own .athl)"
      : `athenea's lights base bake, every splat (usd-athc --drop-backs ${dropBacks}${p.texts.length ? `, prims ${p.texts.join(", ")}` : ""})`,
  });
}
if (only !== "light") {
  const scene = writeSet(hdDir, "detailed", hdEntries);
  summary.hd = {
    dir: hdDir,
    splats: scene.splats,
    bytes: scene.bytes,
    parts: scene.parts,
  };
}
if (only !== "hd") {
  // The smallest crease threshold (finer cut along creases) whose cut has
  // at most 1.1 x keep splats; none (a plain cut) if even 1 is over.
  const cutSplats = (athc, keep, creases) => {
    const probe = path.join(tmp, "probe.athc");
    const log = run("athc-convert", [
      athc,
      probe,
      "--keep-splats",
      String(keep),
      "--creases",
      String(creases),
    ]);
    fs.rmSync(probe, { force: true });
    return Number(log.match(/: (\d+) splats \(of/)?.[1] ?? 0);
  };
  const findCreases = (athc, keep) => {
    let [lo, hi] = [0.03, 1];
    if (cutSplats(athc, keep, lo) <= 1.1 * keep) return String(lo);
    if (cutSplats(athc, keep, hi) > 1.1 * keep) return "2";
    for (let k = 0; k < 6; k++) {
      const mid = Math.sqrt(lo * hi);
      if (cutSplats(athc, keep, mid) <= 1.1 * keep) hi = mid;
      else lo = mid;
    }
    return hi.toFixed(3);
  };
  const lightEntries = [];
  for (const e of hdEntries) {
    if (e.additive) {
      // The floor light is light already: the same cloud in both sets.
      lightEntries.push(e);
      continue;
    }
    // Each part's share of the light set's splats.
    const keep = Math.max(1000, Math.round((lightSplats * e.splats) / totalHd));
    const creases =
      creasesArg === "auto" ? findCreases(e.athc, keep) : creasesArg;
    const athc = path.join(tmp, `${e.name}-light.athc`);
    const athl = path.join(tmp, `${e.name}-light.lights.athl`);
    const log = run("athc-convert", [
      e.athc,
      athc,
      "--gzip",
      "--planes",
      "--keep-splats",
      String(keep),
      "--creases",
      creases,
      "--athl",
      e.athl,
      "--athl-out",
      athl,
    ]);
    const splats = Number(log.match(/: (\d+) splats \(of/)?.[1] ?? 0);
    lightEntries.push({
      ...e,
      athc,
      athl,
      splats,
      origin: `the detailed set's LoD tree cut (athc-convert --keep-splats ${keep} --creases ${creases}), the .athl carried through the cut`,
    });
  }
  const scene = writeSet(lightDir, "light", lightEntries);
  summary.light = {
    dir: lightDir,
    splats: scene.splats,
    bytes: scene.bytes,
    parts: scene.parts,
  };
}
fs.rmSync(tmp, { recursive: true, force: true });
for (const [k, s] of Object.entries(summary)) {
  console.log(
    `${k}: ${s.dir} -- ${(s.splats / 1e6).toFixed(2)}M splats, ${(s.bytes / 1e6).toFixed(1)} MB`,
  );
  for (const p of s.parts) {
    console.log(
      p.athl
        ? `  ${p.name}: ${p.file} ${(p.bytes / 1e6).toFixed(1)} MB, ${p.athl} ${(p.athlBytes / 1e6).toFixed(2)} MB; ${p.lights.map((g) => `${g.group} ${g.blocks} blocks`).join(", ")}`
        : `  ${p.name}: ${p.file} ${(p.bytes / 1e6).toFixed(1)} MB (no lamps)`,
    );
  }
}
console.log(
  "Not uploaded. To publish (when the user says so): the two folders to R2 athenea-assets/sparkwebgpu/, then corvette.html?set=lights",
);
