import { SparkRenderer, SplatMesh, workerPool } from "@sparkjsdev/spark";
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
//   ground: { height, radius }   the dome's floor: a disc of this radius on
//                        y = 0 seen from this height (three's GroundedSkybox)
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
  progress,
  mobile = false,
  // On a phone the scene's own 1k domes (next to its clouds), not the 4k set.
  hdriBase = params.get("hdriBase") ?? (mobile ? `${base}hdri/` : HDRI_BASE),
}) {
  const domeName = mobile
    ? (name) => name.replace(/_(1k|2k|4k)\.hdr$/, "_1k.hdr")
    : fullResHdri;
  window.__athenea = { loaded: false, error: null };
  if (mobile) workerPool.maxWorkers = 1;
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
  // athenea's camera, pulled back along its view (?distance=, default 1.5x)
  // so the whole car and its floor are in frame.
  {
    const target = new THREE.Vector3().fromArray(cam.target);
    const away = new THREE.Vector3().fromArray(cam.position).sub(target);
    // The light set starts farther back (2.2x), the detailed one at 1.5x.
    const pull =
      Number(params.get("distance")) ||
      (params.get("quality") === "hd" ? 1.5 : 2.2);
    camera.position.copy(target).addScaledVector(away, pull);
  }
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
  const groundRadius = TSL.uniform(info.ground?.radius ?? 25);
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
  // three's GroundedSkybox (examples/jsm/objects/GroundedSkybox.js): a
  // sphere of groundRadius centred on the camera that shot the dome,
  // groundHeight above the floor; below 1.5 heights under that centre its
  // vertices are pushed onto the floor (y = 0) and the band above blends
  // into the sphere. Each vertex keeps the dome direction it was made from
  // (skyDir), so the floor's scale follows the shooting height.
  function groundedGeometry(height, radius, resolution = 192) {
    const geometry = new THREE.SphereGeometry(
      radius,
      2 * resolution,
      resolution,
    );
    const pos = geometry.getAttribute("position");
    const dirs = new Float32Array(pos.count * 3);
    const v = new THREE.Vector3();
    const y1 = (-height * 3) / 2;
    for (let i = 0; i < pos.count; i++) {
      v.fromBufferAttribute(pos, i);
      const n = v.clone().normalize();
      dirs.set([n.x, n.y, n.z], i * 3);
      if (v.y < 0) {
        const f = v.y < y1 ? -height / v.y : 1 - (v.y * v.y) / (3 * y1 * y1);
        v.multiplyScalar(f);
        pos.setXYZ(i, v.x, v.y, v.z);
      }
    }
    geometry.setAttribute("skyDir", new THREE.BufferAttribute(dirs, 3));
    return geometry;
  }
  // The floor disc's radius on the ground (where the push onto y = 0 ends).
  const floorDiscOf = (height, radius) =>
    (2 / 3) * Math.sqrt(Math.max(radius * radius - 2.25 * height * height, 0));
  const radiusForDisc = (height, disc) => 1.5 * Math.hypot(disc, height);
  const eye = TSL.cameraPosition;
  const ray = TSL.positionWorld.sub(eye).normalize();
  const d = TSL.select(
    grounded.greaterThan(0.5),
    TSL.attribute("skyDir", "vec3").normalize(),
    ray,
  );
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
    groundedGeometry(groundHeight.value, groundRadius.value),
    new THREE.MeshBasicNodeMaterial({
      side: THREE.DoubleSide,
      depthWrite: false,
    }),
  );
  backdrop.position.y = groundHeight.value;
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
      Number(params.get("lod")) ||
      (mobile ? 600_000 : Math.max(info.splats ?? 0, 2_500_000)),
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
    const mesh = new SplatMesh({
      url: `${base}${part.file}`,
      extSplats: true,
      onProgress: progress?.track(part.file),
    });
    meshes[part.name] = mesh;
    car.add(mesh);
    // A glass cloud's index is the cloud's, not its file's (corvette.json).
    if (part.ior) relight.setIor(mesh, part.ior);
    if (part.catcher) relight.setCatcher(mesh, true);
    // On a phone one cloud at a time, in one worker: each decode grows a
    // worker's WebAssembly memory, and four at once got the tab killed.
    if (mobile) await mesh.initialized;
  }
  const megabytes = parts.reduce((s, p) => s + (p.bytes ?? 0), 0) / 1e6;
  status.textContent = `loading the car (${megabytes.toFixed(0)} MB)…`;

  const state = {
    hdri: domeName(params.get("hdri") ?? "golden_gate_hills_4k.hdr"),
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
    // Give the decoders' memory back once everything is on the GPU: each
    // worker keeps the WebAssembly heap of the largest cloud it decoded
    // (with four, about 0.4 GB for the light set and 2 GB for the detailed
    // one, never shrunk). The LoD traversal's worker is not a free one.
    workerPool.trim();
    await spark.webgpuReady;
    window.__athenea.loaded = true;
  } catch (error) {
    window.__athenea.error = String(error);
    status.textContent = `error: ${error} (the car's files are not under ${base}?)`;
    console.error(error);
  }

  // The dome's floor: its radius and the height it was shot from, set from
  // the menu or with the handle, a ring on the ground: drag it to resize the
  // floor, Shift + drag to raise or lower the camera that shot the dome.
  const floor = {
    height: groundHeight.value,
    radius: groundRadius.value,
    handle: params.get("floorHandle") === "1",
  };
  const ringMaterial = new THREE.MeshBasicNodeMaterial({
    color: 0x8ab4ff,
    transparent: true,
    opacity: 0.85,
    depthTest: false,
  });
  const ring = new THREE.Mesh(
    new THREE.RingGeometry(0.985, 1, 128),
    ringMaterial,
  );
  ring.rotation.x = -Math.PI / 2;
  ring.position.y = 0.002;
  ring.renderOrder = 10;
  scene.add(ring);
  let builtFloor = "";
  const applyFloor = () => {
    groundHeight.value = floor.height;
    floor.radius = Math.min(Math.max(floor.radius, 1.6 * floor.height), 180);
    groundRadius.value = floor.radius;
    const key = `${floor.height}|${floor.radius}`;
    if (key !== builtFloor) {
      builtFloor = key;
      backdrop.geometry.dispose();
      backdrop.geometry = groundedGeometry(floor.height, floor.radius);
      backdrop.position.y = floor.height;
    }
    ring.scale.setScalar(
      Math.max(floorDiscOf(floor.height, floor.radius), 0.1),
    );
    ring.visible = floor.handle && state.ground;
  };
  applyFloor();
  let floorControllers = [];

  let corrector = null;
  const maya = isWorkstation() && params.get("maya") !== "0";
  if (params.get("gui") !== "0") {
    const gui = new GUI({ title: "Corvette" });
    if (innerWidth < 700) gui.close();
    // Every dome at full resolution, the scene's own first (on a phone, the
    // scene's 1k domes only).
    const hdris = [
      ...new Set(
        mobile
          ? (info.hdri ?? []).map(domeName)
          : [...(info.hdri ?? []).map(fullResHdri), ...HDRIS_4K],
      ),
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
    gui
      .add(state, "ground")
      .name("ground + shadow")
      .onChange(() => {
        applySky();
        applyFloor();
      });
    const g = gui.addFolder("floor").close();
    floorControllers = [
      g
        .add(floor, "radius", 3, 180, 0.5)
        .name("dome radius (m)")
        .onChange(applyFloor),
      g
        .add(floor, "height", 0.1, 20, 0.05)
        .name("shot height = floor scale (m)")
        .onChange(applyFloor),
    ];
    g.add(floor, "handle").name("show handle (drag ring)").onChange(applyFloor);
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

  // Dragging the floor ring (registered after the Maya handler, so it has
  // the last word on whether the camera moves).
  {
    const caster = new THREE.Raycaster();
    const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
    const pointer = new THREE.Vector2();
    const onFloorAt = (event) => {
      const r = renderer.domElement.getBoundingClientRect();
      pointer.set(
        ((event.clientX - r.left) / r.width) * 2 - 1,
        -((event.clientY - r.top) / r.height) * 2 + 1,
      );
      caster.setFromCamera(pointer, camera);
      return caster.ray.intersectPlane(plane, new THREE.Vector3());
    };
    let drag = null;
    const dom = renderer.domElement;
    dom.addEventListener(
      "pointerdown",
      (event) => {
        if (!ring.visible || event.altKey || event.button !== 0) return;
        const p = onFloorAt(event);
        if (!p) return;
        const r = Math.hypot(p.x, p.z);
        const disc = floorDiscOf(floor.height, floor.radius);
        if (Math.abs(r - disc) > Math.max(0.12 * disc, 0.4)) return;
        drag = { y: event.clientY, height: floor.height };
        controls.enabled = false;
        dom.setPointerCapture(event.pointerId);
        event.stopImmediatePropagation();
      },
      { capture: true },
    );
    dom.addEventListener("pointermove", (event) => {
      if (!drag) return;
      if (event.shiftKey) {
        floor.height = THREE.MathUtils.clamp(
          drag.height * 2 ** ((drag.y - event.clientY) / 150),
          0.1,
          20,
        );
      } else {
        const p = onFloorAt(event);
        if (p) {
          const disc = THREE.MathUtils.clamp(Math.hypot(p.x, p.z), 1, 110);
          floor.radius = radiusForDisc(floor.height, disc);
        }
      }
      applyFloor();
      for (const c of floorControllers) c.updateDisplay();
    });
    const end = () => {
      if (!drag) return;
      drag = null;
      controls.enabled = true;
    };
    dom.addEventListener("pointerup", end);
    dom.addEventListener("pointercancel", end);
  }

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
  // One frame of the loop below (window.__athenea.frame, for benchmarks
  // that drive frames themselves).
  function frame() {
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
  }
  window.__athenea.frame = frame;
  renderer.setAnimationLoop(() => {
    frame();
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
