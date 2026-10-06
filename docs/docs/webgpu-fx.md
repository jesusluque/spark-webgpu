# WebGPU effects (aofx)

On the WebGPU backend, a rendered frame can go through image effects written for the **AOFX SDK** (`aopenfx`), a C++ plugin interface for GPU effects whose kernels are written in Slang. Spark runs the same Slang kernels, compiled to WGSL, under a TypeScript port of the SDK's effect model. So an effect is written once in Slang and runs in an aofx host on Metal or CUDA and in the browser on WebGPU.

The code is in `src/webgpu/fx/` (exported as the `fx` namespace) and `slang/fx/`. `examples/webgpu/fx.html` runs a chain of effects over a splat render, with a panel built from each effect's description.

## Applying effects to a frame

`FxChain` applies an ordered list of effects to a texture, in place. Render the scene and the splats into a `RenderTarget`, apply the chain to it, and show the target:

```typescript
import * as THREE from "three/webgpu";
import { fx, GpuSplatSource, WgpuSplatRenderer } from "@sparkjsdev/spark";

const chain = new fx.FxChain(renderer.backend.device);
chain.add(new fx.Grade(), { gamma: [1.2, 1.2, 1.2, 1] });
chain.add(new fx.Blur(), { size: [6, 6] });

renderer.setAnimationLoop(() => {
  renderer.setRenderTarget(target);
  renderer.render(scene, camera);
  splats.render(camera, target);
  chain.applyToRenderTarget(renderer, target);   // in place
  renderer.setRenderTarget(null);
  output.render(renderer);                        // a QuadMesh showing target.texture
});
```

- `add(effect, params, options)` appends a step with the effect's defaults, overridden by `params`. `FxChain.defaults(effect, width, height)` gives every default, with normalized ones (such as Transform's centre) scaled to the picture.
- Each step's pass-through input is the step before it; the first sees the frame. Other inputs can be wired to any node through `options.inputs`, such as a generator for Merge's `A`:

    ```typescript
    const wheel = new fx.ColorWheel();
    chain.add(new fx.Merge(), { operation: fx.MERGE_OVER }, {
      inputs: { A: { effect: wheel, params: fx.FxChain.defaults(wheel) } },
    });
    ```

- `options.enabled: false` skips a step, and `options.channels` (`[r, g, b, a]`) says which channels it may change; the others come back from its input, as in an aofx host.
- `chain.context = { scaleX, scaleY, projectWidth, projectHeight }` sets the render scale. Parameters in pixels are canonical (say CSS pixels) and the frame renders at `scaleX` of them (the device pixel ratio), as an aofx host renders a proxy.
- Values an effect attaches to its picture, such as `ImageStatistics`' measurements, arrive in `chain.values` a frame later, since WebGPU readback is asynchronous.

`applyToRenderTarget` takes the target's texture from three's backend, so render into the target first. `apply(texture, options)` works on any `GPUTexture` with `COPY_SRC` and `COPY_DST` usage in one of `rgba16float`, `rgba8unorm(-srgb)`, `bgra8unorm(-srgb)` or `rgba32float`.

### The example effects

The eight examples of the aofx SDK are ported, on their Slang kernels unchanged:

| Effect | What it does |
| ------ | ------------ |
| `Invert` | one minus the picture, with (un)premultiply, a mask and mix |
| `Crop` | a box with soft edges; blacks out the rest or cuts the picture to it |
| `Blur` | a separable Gaussian blur, radius at render scale |
| `Grade` | lift, gamma, gain and the rest of a grade |
| `Merge` | A onto B with thirty-nine operations (`MERGE_OPERATIONS`) |
| `Transform` | translate, rotate, scale and skew, with nine filters (`TRANSFORM_FILTERS`) |
| `CornerPin`, `CornerPinSS` | a perspective warp from four corners, plain or supersampled |
| `Constant`, `CheckerBoard`, `ColorBars`, `ColorWheel`, `NoOp`, `ImageStatistics` | the generators, and a measuring pass-through |

## How it runs

The host half of the aofx contract, in TypeScript:

- **`Gpu`** is what an effect asks the host for: `load(module, entry)` a kernel, `run(kernel, grid, buffers, uniforms)` it, `scratch` pictures for the frame, `keep` buffers across frames, and `read` a buffer back (a promise, where aofx's `read` blocks). Dispatches go through Spark's `KernelRegistry`.
- **`FxGraph`** pulls a node: it works out the node's region of definition from its inputs', asks the effect how much of each input that window needs (`regionOfInterest`), renders those, and either hands back the pass-through input (`isIdentity`, or a disabled node) or allocates the output and calls `process`. It applies the R/G/B/A switches and clamps regions to the project.
- **`TextureBridge`** moves a texture into the picture layout and back with `slang/fx/bridge.slang`: copy to a buffer, unpack to premultiplied linear `float4` with y up, and the reverse. Copies rather than texture bindings, so one kernel serves every supported format and no storage-texture format is needed.
- **`FxChain`** links steps into a graph over the bridged frame and submits.

A **picture** (`FxBuffer`) is a storage buffer of premultiplied `float4` pixels, rows bottom to top, with a `stride` in pixels and the `rect` it covers in the picture's coordinates, y up, as in aofx and OpenFX.

## Writing an effect

An effect is a Slang kernel in `slang/fx/` and an `Effect` subclass. The kernel follows the aofx SDK's rules, so the same file builds in an aofx bundle:

- no `import`s and no `groupshared` (a reduction folds rows in a second dispatch instead; see `imagestats.slang`);
- one parameter struct of four-byte members in declaration order, with explicit padding and no arrays (use `float4` members);
- buffer bindings are positional and belong to the file: a dispatch hands over every buffer the file declares, in order.

A brightness effect, as `slang/fx/brightness.slang`:

```hlsl
struct BrightnessParams {
    uint  srcWidth;
    uint  srcHeight;
    uint  srcStride;
    int   srcOffsetX;
    int   srcOffsetY;
    uint  dstWidth;
    uint  dstHeight;
    uint  dstStride;
    float gain;
    uint  pad0;
    uint  pad1;
    uint  pad2;
};

StructuredBuffer<float4>         src;
RWStructuredBuffer<float4>       dst;
ConstantBuffer<BrightnessParams> params;

[shader("compute")]
[numthreads(16, 16, 1)]
void brightnessMain(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.dstWidth || tid.y >= params.dstHeight) {
        return;
    }
    // Source and target need not share an origin; outside the source is transparent.
    const int sx = int(tid.x) + params.srcOffsetX;
    const int sy = int(tid.y) + params.srcOffsetY;
    float4 rgba = float4(0.0);
    if (sx >= 0 && sy >= 0 && sx < int(params.srcWidth) && sy < int(params.srcHeight)) {
        rgba = src[uint(sy) * params.srcStride + uint(sx)];
    }
    dst[tid.y * params.dstStride + tid.x] = float4(rgba.rgb * params.gain, rgba.a);
}
```

`npm run build:slang` compiles it to `src/webgpu/generated/fx/brightness.ts`. The effect describes itself and processes a request:

```typescript
import brightnessModule from "../../generated/fx/brightness";
import { UniformWriter } from "../../uniforms";
import { Effect, type RenderRequest } from "../Effect";
import { COLOR_PLANE, type EffectDesc, isValidBuffer } from "../types";

export class Brightness extends Effect {
  protected describeEffect(): EffectDesc {
    return {
      identifier: "org.example.brightness",
      label: "Brightness",
      grouping: "Colour",
      description: "Multiplies the colour by a gain.",
      inputs: [{ name: "Source", label: "Source", passThrough: true }],
      outputs: [COLOR_PLANE],
      params: [{
        name: "gain", label: "Gain", type: "double", defaults: [1],
        displayMin: [0], displayMax: [4], hint: "What the colour is multiplied by.",
      }],
    };
  }

  isIdentity(request: RenderRequest) {
    return request.number("gain", 1) === 1;
  }

  process(request: RenderRequest): boolean {
    const source = request.input("Source")?.buffer;
    const target = request.output("Color")?.buffer;
    if (!isValidBuffer(source) || !isValidBuffer(target)) return false;
    const gpu = request.gpu;
    const kernel = gpu.load(brightnessModule, "brightnessMain");
    const u = UniformWriter.for(brightnessModule).setAll({
      srcWidth: source.width,
      srcHeight: source.height,
      srcStride: source.stride,
      srcOffsetX: target.rect.x1 - source.rect.x1,
      srcOffsetY: target.rect.y1 - source.rect.y1,
      dstWidth: target.width,
      dstHeight: target.height,
      dstStride: target.stride,
      gain: request.number("gain", 1),
    });
    // The grid counts pixels; buffers in the order the file declares them.
    return gpu.run(kernel, [target.width, target.height, 1], [source, target], u);
  }
}
```

The other `Effect` methods are optional: `regionOfDefinition` (default: the union of the inputs) for effects that move pixels or generate them, and `regionOfInterest` (default: the output's own rectangle) for effects that read around a pixel, such as a blur. A port of an aofx effect reads line for line like its C++ `process`: the TypeScript types in `fx/types.ts` mirror the SDK's `Types.h` and `Descriptor.h`.

### Reflection checks

Before anything is recorded, `Gpu.run` checks the dispatch against the kernel's reflection as an aofx host checks its kernel trailer, and throws an error naming the problem instead of rendering black:

- the number of buffers against the buffers the file declares (positional form), or each name (record form);
- each buffer's size against its element size, and against the device's `maxStorageBufferBindingSize`;
- that a picture's rows fit its buffer for its width, height and stride;
- the uniform block's size against the struct's, and uniforms given to a kernel without a block;
- the grid in whole threads, and the workgroups it makes against `maxComputeWorkgroupsPerDimension`.

On top of those it checks what WebGPU refuses later with an error far from the cause: a written buffer bound under a second name. Bind `gpu.placeholder` for an output nobody asked for.

## Relation to the aofx SDK

The aofx SDK (`aopenfx/sdk/include/aofx/`) is C++ headers for plugins that a host loads as bundles. Its CMake (`aofx_add_kernel`) compiles each Slang kernel with slangc to a `.metallib` for Metal and to PTX for CUDA and embeds it in the plugin, with a reflection trailer the host checks dispatches against.

Spark replaces the native half and keeps the rest:

| | aofx host | Spark |
| - | --------- | ----- |
| Kernels | Slang → Metal / PTX, embedded | the same Slang → WGSL, `tools/slang-build` |
| Reflection | binary trailer (`AofxKernelTrailer.cmake`) | typed JSON in the generated module |
| Effect model | `Effect`, `RenderRequest`, `EffectDesc` in C++ | the same in TypeScript (`fx/Effect.ts`, `fx/types.ts`) |
| Host | the C++ host's `Gpu`, graph and planes | `fx/Gpu.ts`, `fx/FxGraph.ts`, `fx/TextureBridge.ts` |

So the kernels in `slang/fx/` are copies of `aopenfx/examples/*/*.slang`, and each effect in `src/webgpu/fx/effects/` is a port of the example's `.cpp`. To bring an aofx effect to Spark, copy its `.slang` into `slang/fx/`, run `npm run build:slang`, and port its C++ `Effect` to TypeScript; to take a Spark effect to an aofx host, the `.slang` file builds there unchanged.

What the WebGPU host doesn't have: blocking reads (`Gpu.read` returns a promise, and attached values arrive a frame later), and the aofx SDK's audio, tracks and other planes beyond those the example effects use.
