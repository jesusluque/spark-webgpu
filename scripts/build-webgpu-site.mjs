// Builds the public Spark WebGPU demo site: a static, self-contained folder
// with relative paths only.
//   node scripts/build-webgpu-site.mjs [outDir] [--cache=<dir>]
// Needs `npm run build:production` first (dist/*.min.js). Sample splats
// (Spark's MIT example assets) are downloaded once into the cache folder.
// Files over 25 MB never go in the site: they are served from R2 (see
// lod.html's R2_BASE).
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";

const root = path.resolve(import.meta.dirname, "..");
const args = process.argv.slice(2);
const out = path.resolve(
  args.find((a) => !a.startsWith("--")) ?? path.join(root, "..", "publish"),
);
const cache = path.resolve(
  args.find((a) => a.startsWith("--cache="))?.slice(8) ??
    path.join(root, "..", "publish-cache"),
);
const src = path.join(root, "examples", "webgpu-site");
const nm = path.join(root, "node_modules");
const LIMIT = 25 * 1024 * 1024;

const copy = (from, to) => {
  fs.mkdirSync(path.dirname(path.join(out, to)), { recursive: true });
  fs.copyFileSync(from, path.join(out, to));
};

// Sample splats from Spark's examples (examples/assets.json), mirrored.
const ASSETS = {
  "butterfly.spz": "https://sparkjs.dev/assets/splats/butterfly.spz",
  "valley.spz": "https://sparkjs.dev/assets/splats/valley.spz",
  "cat.spz": "https://sparkjs.dev/assets/splats/cat.spz",
};

fs.rmSync(out, { recursive: true, force: true });
fs.mkdirSync(out, { recursive: true });

for (const name of fs.readdirSync(src)) {
  if (name === "thumbs") continue;
  copy(path.join(src, name), name);
}
// Shared with examples/webgpu/dyno-webgpu.html.
copy(path.join(root, "examples/webgpu/dyno-scenes.js"), "dyno-scenes.js");
const thumbs = path.join(src, "thumbs");
if (fs.existsSync(thumbs)) {
  for (const name of fs.readdirSync(thumbs)) {
    copy(path.join(thumbs, name), `thumbs/${name}`);
  }
}

for (const name of ["spark.module.min.js", "spark.webgpu.module.min.js"]) {
  const file = path.join(root, "dist", name);
  if (!fs.existsSync(file)) {
    throw new Error(`${file} missing: run npm run build:production`);
  }
}
// The entry points and their lazy chunks (plugin kernel variants), in a
// folder named by their content hash: the entry points keep fixed names and
// the browsers and Cloudflare cache them for hours, so a new build must not
// share their URLs with an old one (an old spark.module.min.js under new
// chunks fails to link, an old WASM fails to read new files).
const sparkFiles = fs
  .readdirSync(path.join(root, "dist"))
  .filter((name) => name.endsWith(".module.min.js"))
  .sort();
const hash = crypto.createHash("sha256");
for (const name of sparkFiles) {
  hash.update(name);
  hash.update(fs.readFileSync(path.join(root, "dist", name)));
}
const sparkDir = `vendor/spark/${hash.digest("hex").slice(0, 12)}`;
for (const name of sparkFiles) {
  copy(path.join(root, "dist", name), `${sparkDir}/${name}`);
}
for (const name of fs.readdirSync(out)) {
  if (!name.endsWith(".html")) continue;
  const file = path.join(out, name);
  const html = fs.readFileSync(file, "utf8");
  fs.writeFileSync(
    file,
    html.replaceAll('"./vendor/spark/spark.', `"./${sparkDir}/spark.`),
  );
}
// Pages and the shared helpers revalidate on every load; the hashed Spark
// folder never changes. (Rules must not overlap: Pages joins the headers of
// every rule that matches.)
const revalidate = ["/"];
for (const name of fs.readdirSync(out)) {
  if (/\.(html|js|css)$/.test(name)) {
    revalidate.push(`/${name}`);
    if (name.endsWith(".html")) revalidate.push(`/${name.slice(0, -5)}`);
  }
}
fs.writeFileSync(
  path.join(out, "_headers"),
  [
    ...revalidate.map((p) => `${p}\n  Cache-Control: no-cache`),
    `/${sparkDir}/*\n  Cache-Control: public, max-age=31536000, immutable`,
    "",
  ].join("\n"),
);
for (const name of [
  "three.core.min.js",
  "three.module.min.js",
  "three.webgpu.min.js",
]) {
  copy(path.join(nm, "three/build", name), `vendor/three/${name}`);
}
for (const name of [
  "controls/OrbitControls.js",
  "postprocessing/Pass.js",
  "loaders/HDRLoader.js",
]) {
  copy(
    path.join(nm, "three/examples/jsm", name),
    `vendor/three/addons/${name}`,
  );
}
copy(path.join(nm, "three/LICENSE"), "vendor/three/LICENSE");
copy(
  path.join(nm, "lil-gui/dist/lil-gui.esm.min.js"),
  "vendor/lil-gui/lil-gui.esm.min.js",
);
copy(path.join(nm, "lil-gui/LICENSE.md"), "vendor/lil-gui/LICENSE.md");
copy(path.join(root, "LICENSE"), "vendor/spark/LICENSE");

fs.mkdirSync(cache, { recursive: true });
for (const [name, url] of Object.entries(ASSETS)) {
  const cached = path.join(cache, name);
  if (!fs.existsSync(cached)) {
    console.log(`downloading ${url}`);
    const response = await fetch(url);
    if (!response.ok) throw new Error(`${url}: ${response.status}`);
    fs.writeFileSync(cached, Buffer.from(await response.arrayBuffer()));
  }
  copy(cached, `assets/${name}`);
}

// Check: sizes, and no absolute paths in the pages.
let total = 0;
const walk = (dir) => {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const file = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      walk(file);
      continue;
    }
    const { size } = fs.statSync(file);
    total += size;
    if (size >= LIMIT) throw new Error(`${file} is over 25 MB`);
    if (file.endsWith(".html")) {
      const html = fs.readFileSync(file, "utf8");
      const bad = html.match(/(?:src|href)="\/[^/]|"\.\.\//);
      if (bad) throw new Error(`${file}: non-relative path ${bad[0]}`);
    }
    console.log(
      `${(size / 1024).toFixed(0).padStart(8)} KB  ${path.relative(out, file)}`,
    );
  }
};
walk(out);
console.log(`total ${(total / 1024 / 1024).toFixed(1)} MB in ${out}`);
