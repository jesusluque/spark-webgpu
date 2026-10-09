import { SparkRenderer, SplatMesh, workerPool } from "@sparkjsdev/spark";
import {
  atheneaOutputPlugin,
  atheneaRelightPlugin,
  atheneaSkinPlugin,
  fetchAthcClip,
  fx,
  plugins,
  readAthcSkeleton,
} from "@sparkjsdev/spark/webgpu";
import GUI from "lil-gui";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { HDRLoader } from "three/addons/loaders/HDRLoader.js";
// The animated, relightable sparrow (examples/webgpu-site/sparrow.html):
// athenea's skinned sparrow cloud with a zonal light transfer per splat,
// posed every frame on the GPU by athenea's splat skinning
// (atheneaSkinPlugin) and relit under any HDRI (atheneaRelightPlugin).
//
// sparrow.json (next to the cloud; scripts/build-sparrow.mjs writes it):
//   cloud, cloudBytes, splats, upAxis   the .athc (its skeleton embeds the
//                        first clip)
//   camera: { eye, target, up, verticalFov, near, far }   athenea's camera,
//                        in the stage's (Z-up) coordinates
//   hdri, hdriDefault, hdriMobile, domeRotation
//   clips: [{ name, label, file, bytes, from, to, timeCodesPerSecond,
//            seconds, loops }]   one ATCL file a clip, fetched when played
import * as THREE from "three/webgpu";
import {
  HALF_MAX,
  HDRIS_4K,
  HDRI_4K_BYTES,
  HDRI_BASE,
  addColourCorrector,
  fullResHdri,
  hdriDome,
  hdriLabel,
  isWorkstation,
  mayaControls,
} from "./site.js";

const { TSL } = THREE;

export async function createSparrow({
  renderer,
  base,
  params,
  status,
  progress,
  mobile = false,
  hdriBase = params.get("hdriBase") ?? (mobile ? `${base}hdri/` : HDRI_BASE),
}) {
  window.__athenea = { loaded: false, error: null };
  if (mobile) workerPool.maxWorkers = 1;
  // Phones load the lighter cloud (sparrow-mobile.json: the same clips and
  // camera, athenea's resolution-256 bake) when it is published.
  const sceneJson = async (name) => {
    try {
      const response = await fetch(`${base}${name}`, { cache: "no-cache" });
      return response.ok ? await response.json() : null;
    } catch {
      return null; // a missing file may come back without CORS headers
    }
  };
  const info =
    (params.has("scene") ? await sceneJson(params.get("scene")) : null) ??
    (mobile ? await sceneJson("sparrow-mobile.json") : null) ??
    (await sceneJson("sparrow.json"));
  const domeName = mobile
    ? (name) => name.replace(/_(2k|4k|8k)\.hdr$/, "_1k.hdr")
    : fullResHdri;

  const scene = new THREE.Scene();
  const cam = info.camera;
  const camera = new THREE.PerspectiveCamera(
    cam.verticalFov ?? 40,
    innerWidth / innerHeight,
    cam.near ?? 0.005,
    cam.far ?? 60,
  );
  // athenea's stage is Z-up: the bird turns to three's Y-up. The relighting
  // runs in the stage's frame (its transfer is the bake's directions).
  const stage = new THREE.Group();
  if ((info.upAxis ?? "Z") === "Z") stage.rotation.x = -Math.PI / 2;
  scene.add(stage);
  stage.updateMatrixWorld(true);
  const toWorld = (v) => stage.localToWorld(new THREE.Vector3().fromArray(v));
  // athenea's camera, moved along its view (?distance=, default 0.65 of
  // the way): the 18 cm bird fills more of a wide window.
  const home = {
    target: toWorld(cam.target),
    position: toWorld(cam.target).lerp(
      toWorld(cam.eye),
      Number(params.get("distance")) || 0.65,
    ),
  };
  camera.position.copy(home.position);

  const target = new THREE.RenderTarget(1, 1, {
    type: THREE.HalfFloatType,
    depthTexture: new THREE.DepthTexture(1, 1),
  });
  const output = new THREE.QuadMesh(
    new THREE.MeshBasicNodeMaterial({ colorNode: TSL.texture(target.texture) }),
  );
  const chain = new fx.FxChain(renderer.backend.device);
  function resize() {
    camera.aspect = innerWidth / innerHeight;
    // athenea's square frame: its field across the window's short side.
    const v = THREE.MathUtils.degToRad(cam.verticalFov ?? 40);
    camera.fov = THREE.MathUtils.radToDeg(
      camera.aspect >= 1 ? v : 2 * Math.atan(Math.tan(v / 2) / camera.aspect),
    );
    camera.updateProjectionMatrix();
    renderer.setSize(innerWidth, innerHeight);
    const size = renderer.getDrawingBufferSize(new THREE.Vector2());
    target.setSize(size.x, size.y);
  }
  resize();
  addEventListener("resize", resize);

  // The dome behind the bird, looked up as athenea's domeUv looks it up and
  // turned with the dome.
  const turn = TSL.uniform(new THREE.Vector2(1, 0)); // cos, sin of the rotation
  const intensity = TSL.uniform(1);
  const d = TSL.positionWorld.sub(TSL.cameraPosition).normalize();
  // Filtered, with mipmaps (site.js hdriDome; thread BT).
  const dome = hdriDome(THREE, d, turn);
  const backdrop = new THREE.Mesh(
    new THREE.SphereGeometry(20, 64, 32),
    new THREE.MeshBasicNodeMaterial({
      side: THREE.BackSide,
      depthWrite: false,
    }),
  );
  // Linear radiance into the HalfFloat target, held at its largest finite
  // value (a sun past it would turn to infinity there).
  backdrop.material.colorNode = TSL.min(
    dome.color.mul(intensity),
    TSL.vec3(HALF_MAX),
  );
  scene.add(backdrop);

  // Float end to end (hdr): relit light past 1 is kept per splat and
  // blended in linear light; athenea's display transform comes last.
  const spark = new SparkRenderer({
    renderer,
    hdr: true,
    // ?blur=0.15: the anti-alias blur's variance in px^2 (Spark's 0.3 by
    // default); ?aa=box: the pixel's box instead (WgpuSplatRenderer's
    // aaFilter). Thread BO: both match a supersampled render better far
    // away, at some temporal shimmer (research/simplify-measurements.md).
    ...(params.has("blur") ? { blurAmount: Number(params.get("blur")) } : {}),
    // Phones draw at most 600k (as the Corvette); a desktop draws them all.
    lodSplatCount:
      Number(params.get("lod")) ||
      (mobile ? 600_000 : Math.max(info.splats ?? 0, 1e6)),
  });
  scene.add(spark);

  // The bird's size from its JSON (cloudBytes), for the download bar.
  if (info.cloudBytes) {
    progress?.expect?.(
      info.cloudBytes + (mobile ? 1.6 * 1024 * 1024 : HDRI_4K_BYTES),
    );
    status.textContent = `loading the bird (${(info.cloudBytes / (1024 * 1024)).toFixed(0)} MB)…`;
  }
  const url = `${base}${info.cloud}`;
  const skeleton = await readAthcSkeleton({ url });
  if (!skeleton) throw new Error(`${info.cloud}: no skeleton`);
  const bird = new SplatMesh({
    url,
    extSplats: true,
    onProgress: progress?.track(info.cloud),
  });
  stage.add(bird);
  // The clips: the cloud's own first, the others fetched when played.
  const clips = info.clips.filter(
    (c) => c.loops !== false || params.get("all") === "1",
  );
  const byName = Object.fromEntries(info.clips.map((c) => [c.name, c]));
  const loading = new Map();
  async function ensureClip(name) {
    if (skeleton.clips.some((c) => c.name === name)) return;
    if (!loading.has(name)) {
      loading.set(
        name,
        fetchAthcClip(`${base}${byName[name].file}`).then(
          ({ joints, clip }) => {
            if (joints !== skeleton.joints.length) {
              throw new Error(
                `${name}: ${joints} joints, the skeleton ${skeleton.joints.length}`,
              );
            }
            clip.name = name;
            if (!skeleton.clips.some((c) => c.name === name))
              skeleton.clips.push(clip);
          },
        ),
      );
    }
    await loading.get(name);
  }
  const firstClip =
    byName[params.get("clip")]?.name ??
    info.defaultClip ??
    skeleton.clips[0].name;
  await ensureClip(firstClip);

  const state = {
    clip: firstClip,
    playing: params.get("pause") !== "1",
    speed: Number(params.get("speed") ?? 1),
    time: 0,
    follow: params.get("follow") !== "0",
    hdri: domeName(
      params.get("hdri") ?? info.hdriDefault ?? "golden_gate_hills_4k.hdr",
    ),
    rotation: info.domeRotation ?? 0,
    intensity: 1,
    sun: false,
    sunAzimuth: 40,
    sunElevation: 35,
    sunIntensity: 3,
  };
  const skin = atheneaSkinPlugin({
    skeleton,
    clip: state.clip,
    playing: state.playing,
    speed: state.speed,
  });
  const relight = atheneaRelightPlugin({
    ...(params.has("footprint")
      ? { footprint: Number(params.get("footprint")) }
      : {}),
    pixelDetail: params.get("detail") === "1",
    frame: stage,
  });
  const display = atheneaOutputPlugin({
    view: params.get("view") ?? "agx",
    exposure: Number(params.get("exposure") ?? 0),
  });

  const hdrLoader = new HDRLoader().setDataType(THREE.FloatType);
  // A 4k float dome is 128 MB (85 MB as its half copy with mips on the
  // GPU): only the one shown is kept.
  const hdriCache = new Map();
  async function setHdri(name) {
    state.hdri = name;
    if (!hdriCache.has(name)) {
      status.textContent = `loading ${name}…`;
      hdriCache.set(name, hdrLoader.loadAsync(`${hdriBase}${name}`));
    }
    let tex;
    try {
      tex = await hdriCache.get(name);
    } catch (error) {
      // A dome not on the server (yet): the site's default one instead.
      hdriCache.delete(name);
      const fallback = domeName("golden_gate_hills_4k.hdr");
      if (name === fallback) throw error;
      console.warn(`${name}: ${error}; showing ${fallback}`);
      return setHdri(fallback);
    }
    for (const [other, pending] of hdriCache) {
      if (other === state.hdri) continue;
      hdriCache.delete(other);
      pending.then((t) => t.dispose()).catch(() => {});
    }
    if (state.hdri !== name) return;
    const { width, height, data } = tex.image;
    relight.set({ hdri: { width, height, data, channels: 4 } });
    dome.set(tex);
    applySky();
  }
  function applySky() {
    relight.set({
      rotation: THREE.MathUtils.degToRad(state.rotation),
      intensity: state.intensity,
    });
    const r = THREE.MathUtils.degToRad(state.rotation);
    turn.value.set(Math.cos(r), Math.sin(r));
    intensity.value = state.intensity;
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

  // Where the bird is: the bind-space origin (between its feet) as the
  // pelvis joint's skinning transform carries it, in the world. "follow"
  // moves the camera with it, so a clip that wanders stays in frame.
  const pelvis = Math.max(
    0,
    skeleton.joints.findIndex((j) => j.endsWith("Pelvis")),
  );
  const birdAt = () => {
    const c = skeleton.clips.find((x) => x.name === state.clip);
    const per = skeleton.joints.length * 16;
    const n = c.times.length;
    const t = skin.timeCode;
    let k = 0;
    while (k < n && c.times[k] <= t) k++;
    const k0 = Math.max(0, Math.min(k - 1, n - 1));
    const k1 = Math.min(k, n - 1);
    const a = k1 > k0 ? (t - c.times[k0]) / (c.times[k1] - c.times[k0]) : 0;
    const p = [12, 13, 14].map((e) => {
      const x0 = c.xforms[k0 * per + pelvis * 16 + e];
      return x0 + (c.xforms[k1 * per + pelvis * 16 + e] - x0) * a;
    });
    return stage.localToWorld(new THREE.Vector3().fromArray(p));
  };
  let followed = null;

  let timeCtl = null;
  let playCtl = null;
  async function setClip(name) {
    status.textContent = `loading ${byName[name]?.label ?? name}…`;
    await ensureClip(name);
    state.clip = name;
    const c = skeleton.clips.find((x) => x.name === name);
    skin.set({ clip: name, time: c.times[0] });
    followed = null;
    timeCtl?.min(c.times[0]).max(c.times[c.times.length - 1]);
    timeCtl?.updateDisplay();
  }

  try {
    await setHdri(state.hdri);
    await bird.initialized;
    workerPool.trim();
    await spark.webgpuReady;
    window.__athenea.loaded = true;
  } catch (error) {
    window.__athenea.error = String(error);
    status.textContent = `error: ${error} (the bird's files are not under ${base}?)`;
    console.error(error);
  }

  let corrector = null;
  const maya = isWorkstation() && params.get("maya") !== "0";
  if (params.get("gui") !== "0") {
    const gui = new GUI({ title: "Sparrow" });
    if (innerWidth < 700) gui.close();
    const a = gui.addFolder("animation");
    const labels = Object.fromEntries(clips.map((c) => [c.label, c.name]));
    a.add(state, "clip", labels).name("clip").onChange(setClip);
    playCtl = a
      .add(state, "playing")
      .name("play")
      .onChange((on) => skin.set({ playing: on }));
    a.add(state, "speed", 0, 2, 0.05)
      .name("speed")
      .onChange((s) => skin.set({ speed: s }));
    const range = skin.range();
    timeCtl = a
      .add(state, "time", range.from, range.to, 0.01)
      .name("time (frame)")
      .onChange((t) => {
        state.playing = false;
        playCtl.updateDisplay();
        skin.set({ playing: false, time: t });
      });
    a.add(state, "follow").name("camera follows the bird");
    // Every dome at full resolution, the bake's own first (on a phone, the
    // 1k domes next to the cloud).
    const hdris = [
      ...new Set(
        mobile
          ? (info.hdriMobile ?? []).map(domeName)
          : [...(info.hdri ?? []).map(fullResHdri), ...HDRIS_4K],
      ),
    ];
    const l = gui.addFolder("light");
    l.add(state, "hdri", hdris).name("HDRI").onChange(setHdri);
    l.add(state, "rotation", -180, 180, 1)
      .name("dome rotation")
      .onChange(applySky);
    l.add(state, "intensity", 0, 4, 0.05)
      .name("dome intensity")
      .onChange(applySky);
    const s = l.addFolder("analytic sun").close();
    s.add(state, "sun").onChange(applySky);
    s.add(state, "sunAzimuth", -180, 180, 1).onChange(applySky);
    s.add(state, "sunElevation", -10, 90, 1).onChange(applySky);
    s.add(state, "sunIntensity", 0, 20, 0.1).onChange(applySky);
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
  controls.target.copy(home.target);
  controls.enableDamping = true;
  controls.minDistance = 0.05;
  controls.maxDistance = 4;
  controls.update();
  const frameBird = () => {
    // Back to athenea's view, about where the bird is now.
    const at = state.follow && followed ? followed : home.target;
    camera.position.copy(home.position).sub(home.target).add(at);
    controls.target.copy(at);
    controls.update();
  };
  if (maya) mayaControls(controls, frameBird);
  if (params.get("shot") === "1") controls.enabled = false;

  let host = null;
  let frames = 0;
  let last = performance.now();
  Object.assign(window.__athenea, {
    renderer,
    target,
    chain,
    corrector,
    skin,
    relight,
    display,
    bird,
    spark,
    camera,
    controls,
    skeleton,
    setClip,
    setHdri,
    state,
    applySky,
    info,
  });
  function frame() {
    if (state.follow && window.__athenea.loaded && host) {
      const at = birdAt();
      if (followed) {
        const delta = at.clone().sub(followed);
        camera.position.add(delta);
        controls.target.add(delta);
      }
      followed = at;
    }
    controls.update();
    renderer.setRenderTarget(target);
    renderer.render(scene, camera);
    const splats = spark.webgpu?.splats;
    if (splats && params.has("aa")) splats.options.aaFilter = params.get("aa");
    if (!host && splats && window.__athenea.loaded) {
      host = new plugins.PluginHost({ capabilities: splats.capabilities });
      host.register(skin).register(relight).register(display).attach(splats);
      host.applyFx(chain);
      corrector?.keepLast();
      window.__athenea.host = host;
      const why = host
        .resolve(splats.meshes[0])
        .inactive.find((r) => r.id === skin.id || r.id === relight.id);
      if (why)
        status.textContent = `${why.id} inactive: ${why.reason} ${why.detail ?? ""}`;
    }
    chain.applyToRenderTarget(renderer, target);
    renderer.setRenderTarget(null);
    output.render(renderer);
    if (window.__athenea.captureNext) {
      window.__athenea.captureNext = false;
      window.__athenea.captured = renderer.domElement.toDataURL("image/png");
    }
  }
  window.__athenea.frame = frame;
  renderer.setAnimationLoop(() => {
    frame();
    window.__sparkFrames = (window.__sparkFrames ?? 0) + 1;
    frames++;
    const now = performance.now();
    if (now - last > 500 && window.__athenea.loaded) {
      const fps = (frames * 1000) / (now - last);
      const label = byName[state.clip]?.label ?? state.clip;
      status.textContent = `${label} · ${hdriLabel(state.hdri)} · ${((info.splats ?? 0) / 1e3).toFixed(0)}k splats · ${fps.toFixed(0)} fps`;
      window.__athenea.fps = fps;
      if (state.playing) {
        state.time = skin.timeCode;
        timeCtl?.updateDisplay();
      }
      frames = 0;
      last = now;
      if (host && window.__sparkFrames > 30) window.__ready = true;
    }
  });
}
