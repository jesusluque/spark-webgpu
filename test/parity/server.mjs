// The runner's Vite dev server: the repo's vite.config.ts without file
// watching, plus a plugin that
//   - serves a stub for /@vite/client, so nothing reloads mid-run;
//   - inlines parityInit at the top of every page, active when
//     sessionStorage.__parity holds its config (how Safari, which has no
//     init scripts over WebDriver, gets the frozen clock);
//   - applies the manifest's routes (asset substitutions, source rewrites)
//     for browsers that can't intercept requests;
//   - serves /__parity/blank, a same-origin page to set sessionStorage on.

import { VITE_CLIENT_STUB, parityInit } from "./clock.mjs";

const INLINE_INIT = `(function(){try{var c=sessionStorage.getItem("__parity");if(c)(${parityInit.toString()})(JSON.parse(c));}catch(e){console.error("parity init",e)}})();`;

export function parityPlugin(routes = []) {
  let server;
  return {
    name: "spark-parity",
    configureServer(s) {
      server = s;
      s.middlewares.use(async (req, res, next) => {
        const url = new URL(req.url ?? "/", "http://x");
        if (url.pathname === "/@vite/client") {
          res.setHeader("content-type", "application/javascript");
          res.end(VITE_CLIENT_STUB);
          return;
        }
        if (url.pathname === "/__parity/blank") {
          res.setHeader("content-type", "text/html");
          res.end("<!doctype html><title>parity</title>");
          return;
        }
        const route = routes.find((r) => r.path.test(url.pathname));
        if (!route) return next();
        try {
          if (route.redirect) {
            res.statusCode = 302;
            res.setHeader("location", route.redirect(url.pathname));
            res.setHeader("access-control-allow-origin", "*");
            res.end();
          } else if (
            route.rewrite &&
            // A page's inline module comes back as ?html-proxy JS.
            (!url.pathname.endsWith(".html") ||
              url.searchParams.has("html-proxy"))
          ) {
            const out = await server.transformRequest(req.url);
            res.setHeader("content-type", "application/javascript");
            res.end(route.rewrite(out.code));
          } else {
            next();
          }
        } catch (e) {
          next(e);
        }
      });
    },
    transformIndexHtml(page, { path }) {
      let html = page;
      for (const r of routes) {
        if (r.rewrite && r.path.test(path.split("?")[0]))
          html = r.rewrite(html);
      }
      return {
        html,
        tags: [
          { tag: "script", children: INLINE_INIT, injectTo: "head-prepend" },
        ],
      };
    },
  };
}

/**
 * Starts Vite on `port` from the repo root (the cwd). The repo's config
 * polls every file for changes; the runner's pages never reload, so it
 * doesn't watch at all.
 */
export async function startVite({ port, routes }) {
  const { createServer, loadConfigFromFile } = await import("vite");
  const loaded = await loadConfigFromFile(
    { command: "serve", mode: "development" },
    "vite.config.ts",
    undefined,
    "silent",
  );
  const config = loaded.config;
  config.server = {
    ...config.server,
    port,
    strictPort: true,
    hmr: false,
    watch: null,
  };
  config.plugins = [...(config.plugins ?? []), parityPlugin(routes)];
  const server = await createServer({
    ...config,
    configFile: false,
    logLevel: "warn",
    clearScreen: false,
  });
  await server.listen();
  return {
    base: `http://localhost:${port}`,
    close: () => server.close(),
  };
}
