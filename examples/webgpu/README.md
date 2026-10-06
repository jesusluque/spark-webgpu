# Spark on WebGPU

`SparkRenderer` draws with Spark's WebGPU backend when its renderer is three's
`WebGPURenderer`. An app written for WebGL changes two lines:

```diff
-import * as THREE from "three";
+import * as THREE from "three/webgpu";
 ...
-const renderer = new THREE.WebGLRenderer();
+const renderer = new THREE.WebGPURenderer();
 const spark = new SparkRenderer({ renderer });
 scene.add(spark);
 scene.add(new SplatMesh({ url }));
 renderer.setAnimationLoop(() => renderer.render(scene, camera));
```

`"three"` and `"three/webgpu"` share `three.core.js`, so Spark and the app see
the same three classes. With an import map, add
`"three/webgpu": ".../three.webgpu.js"` and `"three/tsl": ".../three.tsl.js"`.
`new SparkRenderer({ renderer, backend: "webgpu" })` requires a WebGPURenderer.
If the browser has no WebGPU, WebGPURenderer falls back to WebGL2. Spark can't
draw on that fallback and logs an error.

## How it works

`SparkRenderer` is still a transparent mesh in the scene, but three draws
nothing for it. three calls its `onBeforeRender` while it records its render
pass. At that point `SparkWebGPU` (`src/webgpu/SparkWebGPU.ts`) does three
things:

- It maps the scene's visible `SplatMesh`es and `SplatGenerator`s to
  `WgpuSplatRenderer` meshes. PackedSplats and ExtSplats are uploaded once and
  shared. LoD and paged meshes go through `WgpuLod`. Modifiers, edits and
  skinning go through `splatMeshDyno`.
- It submits generate and sort.
- It draws the splats against three's depth buffer.

- **On the canvas**, three renders into a linear half-float target and then
  converts it to the canvas. The splats are drawn after that conversion, in
  sRGB space, as WebGL Spark blends them. As a result, transparent objects in
  front of the splats end up under them. With `antialias: true` the splats
  test against the first sample of three's multisampled depth.
- **Into a RenderTarget or through `PostProcessing`'s `pass()`**, the splats
  are drawn inside three's own pass, sorted
  with the transparent objects. Blending happens in the target's linear space,
  which is also what WebGL Spark does for render targets.

## Differences to know

- **`WebGPURenderer` defaults.** The canvas clears to transparent: set
  `scene.background` or `renderer.setClearColor(color, 1)` if the page isn't
  black. `render()` does nothing before `await renderer.init()`, but
  `setAnimationLoop` waits for init. Apps that render on demand should await
  `init()` first (see `on-demand.html`).
- **Custom dyno code.** Dynos with hand-written GLSL also need WGSL: pass
  `wgsl: { globals, statements }` to `new dyno.Dyno(...)`. See
  `splat-shader-effects.html`. Built-in dynos and modifiers already have both.
  A graph that fails to compile drops its mesh and logs the error.
- **three's ShaderMaterial objects.** `ShaderMaterial` objects such as `Sky`
  have node equivalents (`SkyMesh`; see `sogs.html`).
- **Large paged scenes.** The page pool is limited by the device's
  `maxStorageBufferBindingSize` (128 MB by default, 32–42 pages). Pass
  `requiredLimits: splatRequiredLimits(adapter)` to `WebGPURenderer` to get
  more pages.
- **Render targets on meshes.** The `target` option, `renderTarget` and
  `readTarget` work as on WebGL. three's WebGPURenderer (r180) shows a
  target's texture upside down on a mesh's UVs compared to WebGL, and doesn't
  pick up a `map` that replaced a plain texture after the material compiled
  (see `multiple-viewpoints.html`).
- **Raycasting.** Raycasting works for packed, ext and LoD meshes, but not
  for paged (`.rad`) ones, whose splats live only on the GPU.

## Not supported yet

These throw or warn:

- `renderCubeMap`/`renderEnvMap`, `getLodTreeLevel`;
- custom `vertexShader`/`fragmentShader`;
- WebXR and array cameras;
- `covSplats`, `enable2DGS`, `accumExtSplats`;
- SplatMeshes with a custom `SplatSource`.

Splats are not tone mapped, as on WebGL.

## Examples

| Example | Changed besides the renderer | Mean diff vs WebGL (/255) |
|---|---|---|
| hello-world.html | – | 0.03 |
| multiple-splats.html | – | 0.12 |
| raycasting.html (clicked) | – | 0.08 |
| interactivity.html (GLB, lights, shadows) | asset paths | 0.16 |
| extsplats.html | – | 0.13 |
| nonlod.html | – | 0.16 |
| lod-example.html (examples/lod) | – | 0.25 |
| multi-lod.html, streaming-lod.html | – | 0.58, 0.67 |
| depth-of-field.html | – | 0.97 |
| on-demand.html | `await renderer.init()` | 0.70 |
| splat-shader-effects.html | WGSL for its GLSL dyno | 0.46 |
| procedural-splats.html | – | random stars differ |
| sogs.html | `SkyMesh` for `Sky` | 0.97 |
| debug-color.html, lod-on-demand.html | – | 0.13, 0.23 |
| particle-animation.html | – | 3.09 (time-driven noise) |
| glsl.html | WGSL for its GLSL dynos, opaque clear color | 0.16 |
| dynamic-lighting.html (lighting on, debug SDFs) | – | 0.22, 0.23 |
| mobile-joystick.html (still, after look + walk) | opaque clear color | 0.36, 0.38 |
| multiple-viewpoints.html (still, turning) | the screens' map and UVs (three, see above) | 0.02, 0.05 |
| interactive-deform.html (still, drag, bounce, rotate) | WGSL for its GLSL dyno, main.js inlined | 0.08, 0.09, 0.09, 0.09 |
| interactive-ripples.html (still, 30 and 90 frames after a click) | WGSL for its GLSL dyno, main.js inlined | 0.33, 0.33, 0.34 |
| interactive-holes.html (still, 1 click, 5 clicks, reset) | WGSL for its GLSL dyno (impulse loop unrolled) | 0.50, 0.57, 0.63, 0.50 |
| splat-dissolve-effects.html (t = 0, 10, 40 s) | WGSL for its GLSL dyno, opaque clear color | 0.15, 0.39, 0.38 |
| splat-reveal-effects.html (Magic, Spread, Unroll, Twister, Rain) | WGSL for its GLSL dyno, opaque clear color | 0.36, 0.41, 0.34, 1.51, 2.38 (fast particles) |
| lofi.html (default, next/prev world, bad weather, mid-transition) | WGSL for its 4 GLSL dynos | 0.56, 1.18, 0.98, 1.01, 0.83 |
| splat-transitions.html (spherical, explosion, flow, morph; 11 states) | WGSL for its GLSL dynos | 0.08–0.39 |

`spark-renderer.html` puts a cube through a splat mesh. Use
`?backend=webgl` for the WebGL version, `&mode=target` or `&mode=post` for the
in-pass paths, and `&msaa=1` for antialiasing.
