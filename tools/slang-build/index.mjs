// Compiles every .slang file under slang/ that has entry points to WGSL, and
// writes src/webgpu/generated/<path>.ts exporting the code and its reflection.
//
// The reflection walk follows aopenfx/sdk/cmake/AofxKernelTrailer.cmake: entry
// names and workgroup sizes, then each parameter's binding, its storage-buffer
// element size or its uniform-block size. Here it is emitted as typed JSON
// instead of a binary trailer, and the host builds bind-group layouts from it.
//
//   node tools/slang-build/index.mjs           build all
//   node tools/slang-build/index.mjs --check   fail if a generated file is stale
//
// As a Vite plugin: `import { slangBuild } from "./tools/slang-build/index.mjs"`.

import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../..",
);
const SLANG_DIR = path.join(ROOT, "slang");
const OUT_DIR = path.join(ROOT, "src/webgpu/generated");

export function findSlangc() {
  const candidates = [
    process.env.SLANGC,
    process.env.SLANG_ROOT && path.join(process.env.SLANG_ROOT, "bin/slangc"),
    path.join(os.homedir(), "tools/slang/bin/slangc"),
  ].filter(Boolean);
  for (const c of candidates) {
    if (fs.existsSync(c)) return c;
  }
  try {
    return execFileSync("which", ["slangc"], { encoding: "utf8" }).trim();
  } catch {
    return null;
  }
}

function listSlang(dir) {
  if (!fs.existsSync(dir)) return [];
  return fs.readdirSync(dir, { withFileTypes: true }).flatMap((e) => {
    const p = path.join(dir, e.name);
    if (e.isDirectory()) return listSlang(p);
    return e.name.endsWith(".slang") ? [p] : [];
  });
}

// An entry is a function under a run of attributes that includes either
// [shader("stage")] or, as in the aofx kernels (where CMake names the entries),
// [numthreads(...)], which means compute. Modules without one are import-only.
const ENTRY_RE = /((?:\[[^\]]*\]\s*)+)[\w<>,]+\s+(\w+)\s*\(/g;

function stripComments(source) {
  return source.replace(/\/\*[\s\S]*?\*\//g, "").replace(/\/\/.*$/gm, "");
}

export function findEntries(source) {
  const entries = [];
  for (const m of stripComments(source).matchAll(ENTRY_RE)) {
    const stage =
      /\[shader\("(\w+)"\)\]/.exec(m[1])?.[1] ??
      (/\[numthreads\s*\(/.test(m[1]) ? "compute" : null);
    if (stage) entries.push({ stage, name: m[2] });
  }
  return entries;
}

function scalarBytes(t) {
  if (!t) return 0;
  for (const s of t.sizes ?? []) {
    if (s.kind === "uniform") return s.value;
  }
  return 0;
}

export function reflect(json) {
  const bindings = [];
  for (const p of json.parameters ?? []) {
    const b = p.binding ?? {};
    const t = p.type ?? {};
    const entry = {
      name: p.name,
      group: b.space ?? 0,
      binding: b.index ?? 0,
    };
    if (t.kind === "resource" && t.baseShape === "structuredBuffer") {
      entry.kind = t.access === "readWrite" ? "storage" : "read-only-storage";
      entry.elementBytes = scalarBytes(t.resultType);
    } else if (t.kind === "constantBuffer" || t.kind === "parameterBlock") {
      entry.kind = "uniform";
      entry.bytes = t.elementVarLayout?.binding?.size ?? 0;
      entry.fields = (t.elementType?.fields ?? []).map((f) => ({
        name: f.name,
        offset: f.binding?.offset ?? 0,
        size: f.binding?.size ?? 0,
      }));
    } else if (t.kind === "resource" && t.baseShape === "byteAddressBuffer") {
      entry.kind = t.access === "readWrite" ? "storage" : "read-only-storage";
      entry.elementBytes = 4;
    } else {
      entry.kind = "unsupported";
      entry.type = t.kind;
    }
    bindings.push(entry);
  }
  const entries = (json.entryPoints ?? []).map((e) => ({
    name: e.name,
    stage: e.stage,
    workgroupSize: e.threadGroupSize ?? null,
    // Bindings this entry actually uses, so a pipeline can skip the rest.
    uses: (e.bindings ?? []).filter((x) => x.binding?.used).map((x) => x.name),
  }));
  return { entries, bindings };
}

// Relative import of src/webgpu/KernelModule.ts from generated/<rel>.ts.
function kernelModuleImport(rel) {
  const from = path.dirname(path.join(OUT_DIR, rel));
  const to = path.join(OUT_DIR, "..", "KernelModule");
  return path.relative(from, to).split(path.sep).join("/");
}

function compileOne(slangc, file) {
  const source = fs.readFileSync(file, "utf8");
  const entries = findEntries(source);
  if (entries.length === 0) return null;

  const rel = path.relative(SLANG_DIR, file).replace(/\.slang$/, "");
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "slang-build-"));
  const wgslPath = path.join(tmp, "out.wgsl");
  const jsonPath = path.join(tmp, "out.json");
  const args = [file, "-target", "wgsl", "-I", SLANG_DIR];
  for (const e of entries) args.push("-entry", e.name, "-stage", e.stage);
  args.push("-o", wgslPath, "-reflection-json", jsonPath);
  try {
    execFileSync(slangc, args, { stdio: ["ignore", "pipe", "pipe"] });
  } catch (err) {
    const msg = `${err.stdout ?? ""}${err.stderr ?? ""}`;
    throw new Error(`slangc failed on ${path.relative(ROOT, file)}:\n${msg}`);
  }
  const wgsl = fs.readFileSync(wgslPath, "utf8");
  const reflection = reflect(JSON.parse(fs.readFileSync(jsonPath, "utf8")));
  fs.rmSync(tmp, { recursive: true, force: true });

  const ts = [
    "// Generated by tools/slang-build from",
    `// slang/${rel}.slang — do not edit.`,
    `import type { KernelModule } from "${kernelModuleImport(rel)}";`,
    "",
    `export const wgsl = ${JSON.stringify(wgsl)};`,
    "",
    `export const reflection = ${JSON.stringify(reflection, null, 2)} as const;`,
    "",
    `const module: KernelModule = { name: ${JSON.stringify(rel)}, wgsl, reflection };`,
    "export default module;",
    "",
  ].join("\n");
  return { rel, ts };
}

export function buildAll({ check = false, log = console.log } = {}) {
  const slangc = findSlangc();
  if (!slangc) {
    throw new Error(
      "slangc not found: set SLANGC or SLANG_ROOT, or install to ~/tools/slang/bin",
    );
  }
  const stale = [];
  let count = 0;
  for (const file of listSlang(SLANG_DIR)) {
    const out = compileOne(slangc, file);
    if (!out) continue;
    count += 1;
    const dest = path.join(OUT_DIR, `${out.rel}.ts`);
    const prev = fs.existsSync(dest) ? fs.readFileSync(dest, "utf8") : null;
    if (prev === out.ts) continue;
    if (check) {
      stale.push(out.rel);
      continue;
    }
    fs.mkdirSync(path.dirname(dest), { recursive: true });
    fs.writeFileSync(dest, out.ts);
    log(`slang-build: ${out.rel}`);
  }
  if (stale.length) {
    throw new Error(`slang-build: stale generated files: ${stale.join(", ")}`);
  }
  return count;
}

// Vite plugin: rebuild on .slang changes in dev. A missing slangc is only a
// warning, since the generated files are committed.
export function slangBuild() {
  const run = () => {
    try {
      buildAll({ log: () => {} });
    } catch (err) {
      console.warn(String(err.message ?? err));
    }
  };
  return {
    name: "slang-build",
    buildStart: run,
    handleHotUpdate({ file }) {
      if (file.endsWith(".slang")) run();
    },
  };
}

if (
  process.argv[1] &&
  fileURLToPath(import.meta.url) === path.resolve(process.argv[1])
) {
  const check = process.argv.includes("--check");
  try {
    const n = buildAll({ check });
    console.log(
      `slang-build: ${n} module(s) ${check ? "up to date" : "built"}`,
    );
  } catch (err) {
    console.error(err.message);
    process.exit(1);
  }
}
