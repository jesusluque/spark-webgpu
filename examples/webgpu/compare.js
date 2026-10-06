// Shared settings for compare-webgl.html and compare-webgpu.html.
//   ?file=<asset name>   a file from examples/assets.json (default butterfly.spz)
//   ?n=<count>           instead, a synthetic cloud of n packed splats
//   &flat=1              with zero z scales (2D Gaussians, for enable2DGS)
//   ?w=&h=               canvas size
//   ?lod=1               load the file with LoD splats (lod: true) and draw
//                        the LoD traversal's selection
//   ?rad=<url>|1         instead, a paged .rad file streamed through the LoD
//                        pager (1: the hobbiton scene)
//   ?count=<n>           lodSplatCount for both renderers
//   ?focalDistance=&apertureAngle=&covSplats=1&enable2DGS=1
//                        renderer options for both (SparkRenderer and
//                        WgpuSplatRenderer take the same names; covSplats
//                        loads the file as cov ExtSplats meshes on WebGL)
//   ?ext=1               load the file as ExtSplats (both backends)
//   ?scale=x,y,z         non-uniform object scale (needs covSplats)
//   ?dist=<d>            object distance from the camera (default 3; e.g.
//                        1.2 for a dense close-up, the worst overdraw)
// Both pages expose window.__bench(frames): renders that many frames as fast
// as possible, each waited on until the GPU finishes, with the object
// turning a little every frame so both backends regenerate and re-sort.
export function setupCompare() {
  const params = new URLSearchParams(location.search);
  const file = params.get("file") ?? "butterfly.spz";
  const n = params.get("n") ? Number(params.get("n")) : null;
  const flat = params.get("flat") === "1";
  const lod = params.get("lod") === "1";
  const radParam = params.get("rad");
  const rad =
    radParam === "1"
      ? "https://storage.googleapis.com/forge-dev-public/asundqui/rad/260219/tijerin_w6_hobbiton-lod.rad"
      : radParam;
  const lodSplatCount = params.get("count")
    ? Number(params.get("count"))
    : undefined;
  const size = {
    w: Number(params.get("w") ?? 800),
    h: Number(params.get("h") ?? 600),
  };
  const scale = params.get("scale")?.split(",").map(Number);
  const dist = Number(params.get("dist") ?? 3);
  // The pose examples/hello-world gives the butterfly, without the spin.
  const pose = (object) => {
    if (scale) object.scale.set(...scale);
    if (n) {
      object.position.set(0, 0, -dist);
      return;
    }
    if (rad) {
      object.quaternion.set(1, 0, 0, 0);
      object.position.set(0, 0, -1);
      return;
    }
    object.quaternion.set(1, 0, 0, 0);
    object.position.set(0, 0, -dist);
    object.rotation.y += 0.6;
  };
  const options = {};
  for (const key of ["focalDistance", "apertureAngle"]) {
    if (params.has(key)) options[key] = Number(params.get(key));
  }
  for (const key of ["covSplats", "enable2DGS", "accumExtSplats"]) {
    if (params.has(key)) options[key] = params.get(key) === "1";
  }
  const ext = params.get("ext") === "1" || options.covSplats === true;
  return { file, n, flat, lod, rad, lodSplatCount, size, pose, options, ext };
}

// A deterministic cloud of n splats in a 2-unit ball, packed as PackedSplats;
// `flat`: 2D splats, their z scale 0.
export function syntheticPacked(utils, n, flat = false) {
  const packed = new Uint32Array(n * 4);
  let s = 12345;
  const rnd = () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 2 ** 32;
  };
  for (let i = 0; i < n; i++) {
    let x = 0;
    let y = 0;
    let z = 0;
    do {
      x = rnd() * 2 - 1;
      y = rnd() * 2 - 1;
      z = rnd() * 2 - 1;
    } while (x * x + y * y + z * z > 1);
    const sc = 0.004 + 0.01 * rnd();
    const q = [rnd() - 0.5, rnd() - 0.5, rnd() - 0.5, rnd() - 0.5];
    const l = Math.hypot(...q);
    const [qx, qy, qz, qw] = q.map((v) => v / l);
    utils.setPackedSplat(
      packed,
      i,
      x,
      y,
      z,
      sc,
      sc * (0.3 + rnd()),
      flat ? 0 : sc,
      qx,
      qy,
      qz,
      qw,
      0.3 + 0.6 * rnd(),
      0.5 + 0.5 * x,
      0.5 + 0.5 * y,
      0.5 + 0.5 * z,
    );
  }
  return packed;
}

// Yields to the event loop without setTimeout's 4 ms clamp, so worker and
// mapAsync results can arrive between benchmark frames.
const channel = new MessageChannel();
export function yieldTask() {
  return new Promise((resolve) => {
    channel.port1.onmessage = () => resolve();
    channel.port2.postMessage(0);
  });
}

export async function bench(frames, step) {
  const times = [];
  for (let i = 0; i < frames; i++) {
    const t0 = performance.now();
    await step(i);
    times.push(performance.now() - t0);
    await yieldTask();
  }
  times.sort((a, b) => a - b);
  const mean = times.reduce((a, b) => a + b, 0) / times.length;
  return {
    frames,
    meanMs: mean,
    medianMs: times[times.length >> 1],
    p90Ms: times[Math.floor(times.length * 0.9)],
  };
}

// Resolves once `state()` (a JSON-able LoD summary with a `pending` flag)
// has stayed the same, and not pending, for `ms`: the LoD selection and any
// page streaming have settled, so both backends draw the same splats.
export async function waitSettled(state, ms = 1500) {
  let last = "";
  let since = performance.now();
  while (true) {
    await new Promise((resolve) => setTimeout(resolve, 100));
    const s = state();
    const key = JSON.stringify(s);
    if (key !== last || s.pending) {
      last = key;
      since = performance.now();
    } else if (performance.now() - since >= ms) {
      return s;
    }
  }
}
