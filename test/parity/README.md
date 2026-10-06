# WebGPU vs WebGL parity suite

Screenshots each WebGPU port in `examples/webgpu/` and its WebGL original
under the same frozen clock and scripted input, diffs them, and checks the
diff against per-case thresholds.

```sh
npm run build:dev                # the examples import dist/
npm run test:parity              # every case, Chrome; report in test/parity/out/
npm run test:parity -- lofi dyno # cases whose id contains "lofi" or "dyno"
npm run test:parity -- --list
npm run test:parity -- --perf    # also fps per example on both backends
npm run test:parity:unit         # WebDriver client (mock server), Vite plugin, PNG codec
```

Options: `--base http://localhost:8080` reuses a running server (else the
runner starts Vite on `--port`, 8081 by default), `--out <dir>`,
`--retries <n>` (1: a failing case reruns once and reports `flaky` if it then
passes), `--headed`, `--perf-only`, `--perf-runs <n>`, `--perf-seconds <s>`.
It exits non-zero when a case fails. The report is `results.json` plus
`index.html`, with WebGL | WebGPU | diff images per case.

## How a case runs

- **Frozen clock.** `clock.mjs` replaces `performance.now`, `Date.now` and
  rAF timestamps with a time `t0` that moves only when the input script
  calls `ctx.step(n, dt)`. `Math.random` is seeded. Until the first step,
  rAF callbacks still run each real frame at `t0`, so a page that loads
  slower draws the same instant. `manual` cases draw nothing until stepped,
  for animations that advance per frame.
- **No reloads.** The runner's Vite has no file watcher and serves a stub
  `/@vite/client`. With `--base`, Chrome stubs it through a route.
- **Settling.** The runner waits for the case's `ready` condition and
  `minWait`, then until two screenshots in a row match and no resource
  loaded in between. This covers late splats, LoD refinement and paging.
  It gives up after `maxWait` (60 s) and marks the case ⏱ in the report.
- **Diff.** The diff is the mean absolute difference per RGB channel in /255
  units (the README's numbers) and the % of pixels off by more than 16 in
  any channel. A case also fails when either page is nearly blank (under 1%
  of pixels lit, set by `minLit`), or when the port throws a page error the
  original doesn't.

## The manifest

`manifest.mjs` lists the cases: each example's scenarios, with query
params, input scripts (`act`), settle conditions, asset routes and
thresholds. The mean threshold defaults to the README number × 1.5 + 0.15.
Where the README has no number, the threshold was set from measured runs,
with a margin. Input scripts use the driver-neutral `ctx`
(`drivers.mjs`): `step`, `wait`, `mouse.*`, `key.*`, `click(selector)`,
`clickText`, `select`, `eval`, `waitFor`, `waitForLog`, `settle`,
`setFiles`/`fixture`. A new example
is one `ex(name, [scenarios])` line.

## Safari

```sh
npm run test:parity -- --browser safari                  # WebGPU vs WebGL, both in Safari
npm run test:parity -- --webgl chrome --webgpu safari    # WebGPU in Safari vs WebGL in Chrome
```

Safari runs through `safaridriver` (W3C WebDriver over HTTP, `webdriver.mjs`).
It needs Safari's Develop > "Allow Remote Automation" (or
`safaridriver --enable`). Without it, the runner prints how to enable it and
skips with exit code 0. Safari drives one automation session at a time, so
the runner waits up to `--safari-wait` seconds (600) for another session to
end. Use `--safari-port` to keep away from other safaridrivers.

WebDriver has no init scripts, so the runner's Vite injects the clock into
every page and reads its config from `sessionStorage`, which the runner sets
from a blank same-origin page. For that reason, Safari needs the runner's
own Vite, not `--base`. Cases with file inputs are skipped on Safari.

Across browsers, the screenshots differ in colour management as well as in
rendering: Safari's come in the display's colour space. Chrome runs with
`--force-color-profile=display-p3-d65` against Safari (`--color-profile`),
which brings hello-world from 1.00 to 0.79/255, but the thresholds are
calibrated on same-browser pairs. Safari WebGL vs Safari WebGPU is the
comparison to gate on, and hello-world is 0.03 there, as on Chrome.

## Performance mode

`--perf` relaunches Chrome with `--disable-gpu-vsync
--disable-frame-rate-limit` and the real clock. For each example's first
scenario, it counts frames in `--perf-runs` windows of `--perf-seconds` and
reports the median fps per backend. Other processes on the same GPU skew
the numbers. Safari can't turn off vsync, so its numbers cap at the
display rate.
