// Shared settings for compare-webgl.html and compare-webgpu.html.
// ?file=<asset name> (default butterfly.spz), ?w=&h= for the canvas size.
export function setupCompare() {
  const params = new URLSearchParams(location.search);
  const file = params.get("file") ?? "butterfly.spz";
  const size = {
    w: Number(params.get("w") ?? 800),
    h: Number(params.get("h") ?? 600),
  };
  // The pose examples/hello-world gives the butterfly, without the spin.
  const pose = (object) => {
    object.quaternion.set(1, 0, 0, 0);
    object.position.set(0, 0, -3);
    object.rotation.y += 0.6;
  };
  return { file, size, pose };
}
