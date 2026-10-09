// Static server with Range for the BD validation: / -> wt-validate,
// /examples/js/vendor/ -> node_modules, /r2/ -> publish-r2/sparkwebgpu.
import fs from "node:fs";
import http from "node:http";
import path from "node:path";

const ROOT = new URL("../..", import.meta.url).pathname.replace(/\/$/, "");
const PORT = Number(process.env.VALIDATE_PORT ?? 8121);
const MOUNTS = [
  ["/examples/js/vendor/", `${ROOT}/node_modules/`],
  ["/r2/", process.env.R2_DIR ?? `${ROOT}/../publish-r2/sparkwebgpu/`],
  ["/r2x/", process.env.SETS_DIR ?? `${ROOT}/../publish-r2/sets/`],
  ["/", `${ROOT}/`],
];
const TYPES = {
  ".js": "application/javascript",
  ".mjs": "application/javascript",
  ".html": "text/html",
  ".json": "application/json",
  ".css": "text/css",
  ".wasm": "application/wasm",
};
http
  .createServer((req, res) => {
    const url = decodeURIComponent(req.url.split("?")[0]);
    const [prefix, dir] = MOUNTS.find(([p]) => url.startsWith(p));
    const file = path.join(dir, url.slice(prefix.length));
    let st;
    try {
      st = fs.statSync(file);
    } catch {
      res.writeHead(404);
      res.end();
      return;
    }
    if (!st.isFile()) {
      res.writeHead(404);
      res.end();
      return;
    }
    const type = TYPES[path.extname(file)] ?? "application/octet-stream";
    const range = req.headers.range?.match(/bytes=(\d+)-(\d*)/);
    const headers = {
      "Content-Type": type,
      "Accept-Ranges": "bytes",
      "Cache-Control": "no-store",
    };
    if (range) {
      const start = Number(range[1]);
      const end = range[2]
        ? Math.min(Number(range[2]), st.size - 1)
        : st.size - 1;
      res.writeHead(206, {
        ...headers,
        "Content-Range": `bytes ${start}-${end}/${st.size}`,
        "Content-Length": end - start + 1,
      });
      fs.createReadStream(file, { start, end }).pipe(res);
    } else {
      res.writeHead(200, { ...headers, "Content-Length": st.size });
      fs.createReadStream(file).pipe(res);
    }
  })
  .listen(PORT, "127.0.0.1", () => console.log(`serving on ${PORT}`));
