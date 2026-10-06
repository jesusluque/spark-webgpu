// The examples/envmap scene for envmap-webgl.html and envmap.html: two
// fireplace splats around a rubber duck that reflects them through an
// environment map rendered from its position.
//   &t=<seconds>   a fixed duck rotation for screenshots; turns when absent
//   ?plastic=1     the rough, non-metal duck

export const CLEAR_COLOR = 0x1b2037;

export function envmapParams() {
  const params = new URLSearchParams(location.search);
  return {
    time: params.has("t") ? Number(params.get("t")) : null,
    plastic: params.get("plastic") === "1",
  };
}

// Poses for the two splat meshes (objects carrying the splats).
export function poseBackgrounds(background, background2) {
  background.quaternion.set(1, 0, 0, 0);
  background.position.set(0.5, 0, -1);
  background.scale.setScalar(0.5);
  background2.quaternion.set(1, 0, 0, 0);
  background2.rotation.y = Math.PI;
  background2.position.set(-0.5, 0, 0.0);
  background2.scale.setScalar(0.5);
}

export async function loadDuck(GLTFLoader, getAssetFileURL) {
  const gltf = await new GLTFLoader().loadAsync(
    await getAssetFileURL("rubberduck.glb"),
  );
  const duck = gltf.scene;
  duck.position.set(0, 0.45, -0.4);
  return duck;
}

export function setDuckMaterial(duck, envMap, plastic) {
  for (const obj of duck.children) {
    if (envMap) obj.material.envMap = envMap;
    obj.material.metalness = plastic ? 0.0 : 1.0;
    obj.material.roughness = plastic ? 0.2 : 0.02;
  }
}
