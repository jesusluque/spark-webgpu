// Shared settings for compare-webgl.html and compare-webgpu.html.
//   ?file=<asset name>   a file from examples/assets.json (default butterfly.spz)
//   ?n=<count>           instead, a synthetic cloud of n packed splats
//   ?w=&h=               canvas size
// Both pages expose window.__bench(frames): renders that many frames as fast
// as possible, each waited on until the GPU finishes, with the object
// turning a little every frame so both backends regenerate and re-sort.
export function setupCompare() {
  const params = new URLSearchParams(location.search);
  const file = params.get("file") ?? "butterfly.spz";
  const n = params.get("n") ? Number(params.get("n")) : null;
  const size = {
    w: Number(params.get("w") ?? 800),
    h: Number(params.get("h") ?? 600),
  };
  // The pose examples/hello-world gives the butterfly, without the spin.
  const pose = (object) => {
    if (n) {
      object.position.set(0, 0, -3);
      return;
    }
    object.quaternion.set(1, 0, 0, 0);
    object.position.set(0, 0, -3);
    object.rotation.y += 0.6;
  };
  return { file, n, size, pose };
}

// A deterministic cloud of n splats in a 2-unit ball, packed as PackedSplats.
export function syntheticPacked(utils, n) {
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
      sc,
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
