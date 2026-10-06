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

- **On the canvas** (with or without `antialias`), three renders into a
  linear half-float target and then converts it to the canvas. WebGL Spark
  blends splats in sRGB space on its canvas. So where the SparkRenderer comes
  in three's transparent order, `SparkWebGPU` ends three's pass, draws the
  splats into an 8-bit layer of their own (premultiplied sRGB colour and
  transmittance), composites that over the target in sRGB
  (`src/webgpu/SrgbComposite.ts`) and resumes three's pass, as three's own
  `copyFramebufferToTexture` does. Transparent objects sorted after the
  splats (by `renderOrder` or depth) draw over them, as on WebGL. With tone
  mapping the splats instead go on the canvas after three's output pass, so
  they are not tone mapped (as on WebGL), but over all transparent objects.
- **Into a RenderTarget or through `PostProcessing`'s `pass()`**, the splats
  are drawn inside three's own pass, sorted with the transparent objects,
  and blend in the target's linear space, as WebGL Spark does in render
  targets. `new SparkRenderer({ srgbBlend: true })` (or
  `WgpuSplatRenderer`'s `srgbBlend` option) blends them in sRGB there too, as
  on WebGL's canvas: envmap.html's main view goes from 7.62 to 1.54/255
  against WebGL with it.

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
- **Raycasting.** Raycasting works for packed, ext and LoD meshes, but not
  for paged (`.rad`) ones, whose splats live only on the GPU.

- **Offscreen renders.** The `target` option with `renderTarget()` /
  `readTarget()`, and `renderCubeMap()` / `readCubeTargets()`, work as on
  WebGL: the SparkRenderer in the scene draws as the one rendering
  (`sparkOverride`). `rawColor: true` writes splat colours as they are,
  with no colour conversion, for data packed into RGB (render-cube-depth).
- **Baking colours.** `RgbaArray.render()` runs GLSL readers. On WebGPU,
  `spark.getRgba({ generator })` gives a generator's RGBA as it generates
  it (modifiers, recolor), in a GPU-backed `RgbaArray` usable as
  `splatRgba` and readable with `read()`; `fromPackedSplats()` decodes on
  the CPU (splat-painter).
- **Portals.** The splat shader has `DISK_PORTAL_FRAGMENT_SHADER`'s disk
  clip built in: a SparkRenderer with the `diskCenter`, `diskNormal`,
  `diskRadius` and `diskTwoSided` extraUniforms and no `fragmentShader`
  gets it (portal.html). `SparkPortals` works unchanged.
- **three ShaderMaterials** become TSL node materials. three/webgpu converts
  every material's output to sRGB, which WebGL's ShaderMaterial and
  MeshDepthMaterial skip (splat-portal, render-cube-depth). three/webgpu has
  no `MeshDepthMaterial`.
- **Generator time.** WebGL SparkRenderer's own `Timer` is never updated
  (`ownsTimer` is inverted in SparkRenderer.ts), so generators and
  `onFrame` see time 0 there; on WebGPU they get `performance.now()`.

## Not supported yet

These throw or warn:

- `renderEnvMap` (use `WgpuCubeMap.renderEnvMap`), `getLodTreeLevel`;
- custom `vertexShader`/`fragmentShader` (the portal disk clip is built in);
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
| particle-simulation.html | – | 0.20 (clock at 0) |
| splat-flow.html | WGSL for its transition dyno | 0.32, 0.25 mid-transition |
| viewer.html (`?url=`, file input) | clear colour | 0.48, 0.17 |
| splat-painter.html (paint, erase, undo strokes) | `spark.getRgba` for `RgbaArray.render` | 0.57; exported SPZ: same splats, 88 bytes of 40 MB differ |
| render-cube-depth.html (depth on) | WGSL dyno, TSL depth material, linear output + `rawColor` in depth mode | 1.05 off, 6.13 on (low packed-depth bits); cube readback means within 0.3/255 |
| portal.html | no fragmentShader (built-in disk clip) | 0.32 (local assets substituted) |
| newportal.html | adapter storage limits | 0.33 with a non-paged scene; paged LoD streams differently |
| splat-portal.html (+ splat-portal/main.js) | TSL portal material | 0.58 |
| editor.html (normal colour, clip, grid) | – | 0.01 empty, 0.46 with a file, 0.30 with debug options |

`spark-renderer.html` puts a cube through a splat mesh. Use
`?backend=webgl` for the WebGL version, and `&mode=target`, `&mode=post` or
`&msaa=1` for the in-pass paths, `&srgb=1` for `srgbBlend`. Against WebGL,
outside the transparent plane (three's own blending differs there): canvas
and MSAA 0.31/255; target and post 0.84 linear, 0.30 with `&srgb=1`. A
transparent plane with `renderOrder = 1` over the splats now draws over
them, as on WebGL (it went under them before).

Parity numbers use a frozen clock (`performance.now`, `Date.now`, rAF time)
and a seeded `Math.random`.
