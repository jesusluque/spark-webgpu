import { SparkRenderer, SplatMesh } from "@sparkjsdev/spark";
import {
  atheneaOutputPlugin,
  atheneaRelightPlugin,
  fx,
  plugins,
} from "@sparkjsdev/spark/webgpu";
import GUI from "lil-gui";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { HDRLoader } from "three/addons/loaders/HDRLoader.js";
// The relightable Corvette, shared by examples/webgpu-site/corvette.html (the
// public page) and examples/webgpu/athenea-corvette.html (the dev page): the
// page brings the renderer and its import map, this builds the scene.
//
// corvette.json (next to the clouds) says what to load:
//   parts: [{ name, file, ior?, catcher?, label }]   the .athc clouds
//   upAxis: "Z"          athenea's stage is Z-up; the clouds are kept as baked
//                        and the car is turned here
//   camera: { position, target, horizontalFov }   athenea's /World/Camera, Y-up
//   hdri: [names], hdriDefault, domeRotation (degrees, athenea's DomeLight turn)
//                        names at any size: the page loads the 4k original
//                        of each from `hdriBase` (site.js HDRI_BASE)
//   ground: { height }   the dome's floor is projected onto y = 0 from this height
import * as THREE from "three/webgpu";
import {
  HALF_MAX,
  HDRIS_4K,
  HDRI_BASE,
  addColourCorrector,
  fullResHdri,
  hdriLabel,
  isWorkstation,
  mayaControls,
} from "./site.js";

const { TSL } = THREE;

export async function createCorvette({
  renderer,
  base,
  params,
  status,
  hdriBase = params.get("hdriBase") ?? HDRI_BASE,
}) {
  window.__athenea = { loaded: false, error: null };
  const info = await (
    await fetch(`${base}${params.get("scene") ?? "corvette.json"}`)
  ).json();

  const scene = new THREE.Scene();
  const cam = info.camera;
  const camera = new THREE.PerspectiveCamera(
    40,
    innerWidth / innerHeight,
    0.05,
    200,
  );
  camera.position.fromArray(cam.position);
  const hfov = THREE.MathUtils.degToRad(cam.horizontalFov);

  const target = new THREE.RenderTarget(1, 1, {
    type: THREE.HalfFloatType,
    depthTexture: new THREE.DepthTexture(1, 1),
  });
  const output = new THREE.QuadMesh(
    new THREE.MeshBasicNodeMaterial({ colorNode: TSL.texture(target.texture) }),
  );
  const chain = new fx.FxChain(renderer.backend.device);
  function resize() {
    const aspect = innerWidth / innerHeight;
    // athenea's horizontal field on a wide window, the same vertical one on a
    // tall window (as on a 16:9 frame).
    const vfov = 2 * Math.atan(Math.tan(hfov / 2) / Math.max(aspect, 16 / 9));
    camera.fov = THREE.MathUtils.radToDeg(vfov);
    camera.aspect = aspect;
    camera.updateProjectionMatrix();
    renderer.setSize(innerWidth, innerHeight);
    const size = renderer.getDrawingBufferSize(new THREE.Vector2());
    target.setSize(size.x, size.y);
  }
  resize();
  addEventListener("resize", resize);

  // The dome behind the car, looked up as athenea's domeUv looks it up and
  // turned with the dome; below the horizon its floor is projected onto the
  // ground plane (as three's GroundedSkybox does), so the car stands on it.
  const turn = TSL.uniform(new THREE.Vector2(1, 0)); // cos, sin of the rotation
  const intensity = TSL.uniform(1);
  const groundHeight = TSL.uniform(info.ground?.height ?? 1.6);
  const grounded = TSL.uniform(1);
  const skyTexture = TSL.texture(
    new THREE.DataTexture(
      new Float32Array(4),
      1,
      1,
      THREE.RGBAFormat,
      THREE.FloatType,
    ),
  );
  const eye = TSL.cameraPosition;
  const ray = TSL.positionWorld.sub(eye).normalize();
  const hit = eye.add(ray.mul(eye.y.negate().div(TSL.min(ray.y, -1e-4))));
  const fromCentre = hit.sub(TSL.vec3(0, groundHeight, 0)).normalize();
  const onGround = ray.y
    .lessThan(0)
    .and(grounded.greaterThan(0.5))
    .and(eye.y.greaterThan(0));
  const d = TSL.select(onGround, fromCentre, ray);
  const local = TSL.vec3(
    turn.x.mul(d.x).sub(turn.y.mul(d.z)),
    d.y,
    turn.y.mul(d.x).add(turn.x.mul(d.z)),
  );
  const skyUv = TSL.vec2(
    TSL.fract(
      TSL.atan(local.z, local.x)
        .add(Math.PI / 2)
        .div(2 * Math.PI),
    ),
    TSL.acos(TSL.clamp(local.y, -1, 1)).div(Math.PI),
  );
  const backdrop = new THREE.Mesh(
    new THREE.SphereGeometry(100, 64, 32),
    new THREE.MeshBasicNodeMaterial({
      side: THREE.BackSide,
      depthWrite: false,
    }),
  );
  // Linear radiance into the HalfFloat target, held at its largest finite
  // value (a sun past it would turn to infinity there).
  backdrop.material.colorNode = TSL.min(
    skyTexture.sample(skyUv).rgb.mul(intensity),
    TSL.vec3(HALF_MAX),
  );
  scene.add(backdrop);

  // The whole car by default: below its splat count, the LoD replaces the
  // small splats with merged ones even close up (?lod=<splats> to cap it).
  // Float end to end (hdr): the relit light, past 1 in the reflections, is
  // kept per splat and blended in linear light into the HalfFloat target
  // over the unclipped dome; athenea's display transform comes last.
  const spark = new SparkRenderer({
    renderer,
    hdr: true,
    lodSplatCount:
      Number(params.get("lod")) || Math.max(info.splats ?? 0, 2_500_000),
  });
  scene.add(spark);
  // athenea's stage is Z-up: the car turns to three's Y-up.
  const car = new THREE.Group();
  if ((info.upAxis ?? "Z") === "Z") car.rotation.x = -Math.PI / 2;
  scene.add(car);

  // Per-pixel slope and sharp coat (?detail=1): the paint's 0.03-0.07 rough
  // coat reads its mirror per pixel; several times slower.
  // The clouds' transfers, cells and fields are directions of athenea's
  // Z-up stage: the relighting runs in the car's frame (`frame`), the dome
  // and the sun taken into it.
  const relight = atheneaRelightPlugin({
    pixelDetail: params.get("detail") === "1",
    frame: car,
  });
  const display = atheneaOutputPlugin({
    // athenea's look for the car: the Standard view on sRGB.
    view: params.get("view") ?? "standard",
    exposure: 0,
  });
  const skip = new Set(
    ["glass", "trim", "catcher"].filter((k) => params.get(k) === "0"),
  );
  const parts = info.parts.filter((p) => !skip.has(p.group ?? p.name));
  const meshes = {};
  for (const part of parts) {
    const mesh = new SplatMesh({ url: `${base}${part.file}`, extSplats: true });
    meshes[part.name] = mesh;
    car.add(mesh);
    // A glass cloud's index is the cloud's, not its file's (corvette.json).
    if (part.ior) relight.setIor(mesh, part.ior);
    if (part.catcher) relight.setCatcher(mesh, true);
  }
  const megabytes = parts.reduce((s, p) => s + (p.bytes ?? 0), 0) / 1e6;
  status.textContent = `loading the car (${megabytes.toFixed(0)} MB)…`;

  const state = {
    hdri: fullResHdri(params.get("hdri") ?? info.hdriDefault ?? info.hdri[0]),
    rotation: info.domeRotation ?? 0,
    intensity: 1,
    sun: false,
    sunAzimuth: 40,
    sunElevation: 35,
    sunIntensity: 3,
    indirect: true,
    ground: true,
  };
  const hdrLoader = new HDRLoader().setDataType(THREE.FloatType);
  // A 4k float dome is 128 MB as a texture: only the one shown is kept (the
  // browser's HTTP cache keeps the files), a request in flight shared.
  const hdriCache = new Map();
  async function setHdri(name) {
    state.hdri = name;
    if (!hdriCache.has(name)) {
      status.textContent = `loading ${name}…`;
      hdriCache.set(name, hdrLoader.loadAsync(`${hdriBase}${name}`));
    }
    const tex = await hdriCache.get(name);
    for (const [other, pending] of hdriCache) {
      if (other === state.hdri) continue;
      hdriCache.delete(other);
      pending.then((t) => t.dispose()).catch(() => {});
    }
    if (state.hdri !== name) return;
    const { width, height, data } = tex.image;
    relight.set({ hdri: { width, height, data, channels: 4 } });
    tex.flipY = false;
    skyTexture.value = tex;
    applySky();
  }
  function applySky() {
    relight.set({
      rotation: THREE.MathUtils.degToRad(state.rotation),
      intensity: state.intensity,
      indirect: state.indirect,
    });
    const r = THREE.MathUtils.degToRad(state.rotation);
    turn.value.set(Math.cos(r), Math.sin(r));
    intensity.value = state.intensity;
    grounded.value = state.ground ? 1 : 0;
    if (meshes.catcher) meshes.catcher.visible = state.ground;
    const az = THREE.MathUtils.degToRad(state.sunAzimuth);
    const el = THREE.MathUtils.degToRad(state.sunElevation);
    relight.set({
      sun: state.sun
        ? {
            direction: [
              Math.cos(el) * Math.sin(az),
              Math.sin(el),
              Math.cos(el) * Math.cos(az),
            ],
            intensity: state.sunIntensity,
          }
        : null,
    });
  }

  try {
    await setHdri(state.hdri);
    await Promise.all(Object.values(meshes).map((m) => m.initialized));
    await spark.webgpuReady;
    window.__athenea.loaded = true;
  } catch (error) {
    window.__athenea.error = String(error);
    status.textContent = `error: ${error} (the car's files are not under ${base}?)`;
    console.error(error);
  }

  let corrector = null;
  const maya = isWorkstation() && params.get("maya") !== "0";
  if (params.get("gui") !== "0") {
    const gui = new GUI({ title: "Corvette" });
    if (innerWidth < 700) gui.close();
    // Every dome at full resolution, the scene's own first.
    const hdris = [
      ...new Set([...(info.hdri ?? []).map(fullResHdri), ...HDRIS_4K]),
    ];
    gui.add(state, "hdri", hdris).name("HDRI").onChange(setHdri);
    gui
      .add(state, "rotation", -180, 180, 1)
      .name("dome rotation")
      .onChange(applySky);
    gui
      .add(state, "intensity", 0, 4, 0.05)
      .name("dome intensity")
      .onChange(applySky);
    gui.add(state, "ground").name("ground + shadow").onChange(applySky);
    gui
      .add({ detail: relight.options.pixelDetail }, "detail")
      .name("per-pixel detail (slow)")
      .onChange((on) => relight.set({ pixelDetail: on }));
    // A direct-only transfer (t16, the light set) has no bounce to turn off.
    if (parts.some((p) => (p.transferCount ?? 0) >= 36)) {
      gui.add(state, "indirect").name("indirect + field").onChange(applySky);
    }
    const s = gui.addFolder("analytic sun").close();
    s.add(state, "sun").onChange(applySky);
    s.add(state, "sunAzimuth", -180, 180, 1).onChange(applySky);
    s.add(state, "sunElevation", -10, 90, 1).onChange(applySky);
    s.add(state, "sunIntensity", 0, 20, 0.1).onChange(applySky);
    gui
      .add(
        spark,
        "lodSplatCount",
        200_000,
        Math.max(info.splats ?? 0, 2_500_000),
        100_000,
      )
      .name("detail (splats)");
    const p = gui.addFolder("parts").close();
    for (const part of parts) {
      if (part.catcher) continue;
      p.add(meshes[part.name], "visible").name(part.label ?? part.name);
    }
    const o = gui.addFolder("output").close();
    for (const c of display.ui) {
      const v = { [c.label]: c.get() };
      if (c.type === "select") o.add(v, c.label, c.options).onChange(c.set);
      else o.add(v, c.label, c.min, c.max, c.step).onChange(c.set);
    }
    corrector = addColourCorrector(gui, chain, fx);
    if (maya) {
      gui
        .add(
          { help: "Alt+LMB tumble · Alt+MMB track · Alt+RMB dolly · F frame" },
          "help",
        )
        .name("Maya camera")
        .disable();
    }
  }

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.fromArray(cam.target);
  controls.enableDamping = true;
  controls.minDistance = 1.5;
  controls.maxDistance = 20;
  controls.maxPolarAngle = THREE.MathUtils.degToRad(88);
  controls.update();
  if (maya) {
    const home = {
      position: camera.position.clone(),
      target: controls.target.clone(),
    };
    mayaControls(controls, () => {
      camera.position.copy(home.position);
      controls.target.copy(home.target);
      controls.update();
    });
  }
  if (params.get("shot") === "1") controls.enabled = false;

  let host = null;
  let frames = 0;
  let last = performance.now();
  Object.assign(window.__athenea, {
    renderer,
    target,
    chain,
    corrector,
    relight,
    display,
    meshes,
    spark,
    camera,
    controls,
    setHdri,
    state,
    applySky,
  });
  renderer.setAnimationLoop(() => {
    controls.update();
    renderer.setRenderTarget(target);
    renderer.render(scene, camera);
    const splats = spark.webgpu?.splats;
    if (!host && splats) {
      host = new plugins.PluginHost({ capabilities: splats.capabilities });
      host.register(relight).register(display).attach(splats);
      host.applyFx(chain);
      corrector?.keepLast();
      window.__athenea.host = host;
      const why = host
        .resolve(splats.meshes[0])
        .inactive.find((r) => r.id === relight.id);
      if (why)
        status.textContent = `relight inactive: ${why.reason} ${why.detail ?? ""}`;
    }
    chain.applyToRenderTarget(renderer, target);
    renderer.setRenderTarget(null);
    output.render(renderer);
    window.__sparkFrames = (window.__sparkFrames ?? 0) + 1;
    frames++;
    const now = performance.now();
    if (now - last > 500 && window.__athenea.loaded) {
      const fps = (frames * 1000) / (now - last);
      status.textContent = `${hdriLabel(state.hdri)} · ${(info.splats / 1e6).toFixed(2)}M splats · ${fps.toFixed(0)} fps`;
      window.__athenea.fps = fps;
      frames = 0;
      last = now;
      if (host && window.__sparkFrames > 30) window.__ready = true;
    }
  });
}
