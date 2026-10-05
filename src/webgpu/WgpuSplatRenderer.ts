// Splat rendering on three.js's WebGPURenderer with the Slang kernels:
// generate (compute) -> sort -> draw (own render pass after three's).
//
// Usage:
//   const renderer = new WebGPURenderer({ antialias: false });
//   await renderer.init();
//   const splats = new WgpuSplatRenderer(renderer);
//   splats.add(GpuSplatSource.fromExt(device, a, b, count), mesh);
//   renderer.setAnimationLoop(() => {
//     renderer.setRenderTarget(target);     // RenderTarget with a DepthTexture
//     renderer.render(scene, camera);
//     splats.render(camera, target);
//     renderer.setRenderTarget(null);
//     output.render(renderer);              // QuadMesh showing target.texture
//   });
//
// The draw pass loads the texture three just rendered into and blends
// premultiplied splats over it. Into a RenderTarget with a depthTexture the
// splats are occluded by the scene (three's canvas depth buffer is internal,
// so on the canvas they draw over it). Sorting is on the CPU for now, from a
// metric readback one frame behind, as SparkRenderer does with its WASM sort.

import * as THREE from "three";
import { GpuSorter } from "./GpuSorter";
import type { GpuSplatSource } from "./GpuSplatSource";
import { KernelRegistry } from "./KernelRegistry";
import { type GpuCapabilities, capabilitiesOf } from "./capabilities";
import { sortBackToFront } from "./cpuSort";
import drawModule from "./generated/draw/splat_draw";
import generateModule from "./generated/kernels/generate";
import { createStorage, upload } from "./gpuBuffers";
import {
  type ReflectedRenderPipeline,
  createBindGroups,
  createReflectedRenderPipeline,
} from "./renderPipeline";
import { UniformWriter } from "./uniforms";

const GEN_SRC_EXT = 1;
const GEN_OUT_EXT = 2;
const GEN_USE_LOD = 4;
const GEN_LOD_OPACITY = 8;
const GEN_SORT_RADIAL = 16;

const DRAW_EXT = 1;
const DRAW_ORTHOGRAPHIC = 16;
const DRAW_ENCODE_LINEAR = 32;
const DRAW_PREMULTIPLIED = 64;

export interface WgpuSplatMesh {
  source: GpuSplatSource;
  object: THREE.Object3D;
  recolor: THREE.Vector4;
  /** Source indices to draw, from LOD traversal; all splats when null. */
  lodIndices: Uint32Array | null;
  lodBuffer: GPUBuffer | null;
}

export interface WgpuSplatRendererOptions {
  /** Test against the render target's depthTexture, if it has one (default true). */
  depthTest?: boolean;
  sortRadial?: boolean;
  /**
   * "gpu" (default): radix sort on the GPU in the same frame, drawn with an
   * indirect count. "cpu": metric readback and a JS radix sort, a frame behind.
   */
  sort?: "gpu" | "cpu";
  maxStdDev?: number;
  minPixelRadius?: number;
  maxPixelRadius?: number;
  minAlpha?: number;
  blurAmount?: number;
  preBlurAmount?: number;
  falloff?: number;
  clipXY?: number;
  focalAdjustment?: number;
}

interface WebGPURendererLike {
  backend: {
    isWebGPUBackend?: boolean;
    device: GPUDevice;
    context: GPUCanvasContext;
    /** three's per-resource backend data; `.texture` for textures. */
    get(resource: object): { texture?: GPUTexture } | undefined;
  };
}

export class WgpuSplatRenderer {
  readonly device: GPUDevice;
  readonly registry: KernelRegistry;
  readonly meshes: WgpuSplatMesh[] = [];
  /** Counters for debugging and benchmarks. */
  readonly stats = { frames: 0, draws: 0, sorts: 0, drawn: 0, sortMs: 0 };
  options: Required<WgpuSplatRendererOptions>;
  /** What the device allows; consulted for the sort path and sizes. */
  readonly capabilities: GpuCapabilities;

  private capacity = 0;
  private accumulator: GPUBuffer | null = null;
  private metric: GPUBuffer | null = null;
  private ordering: GPUBuffer | null = null;
  private sorter: GpuSorter;
  private drawUniform: GPUBuffer;
  private pipelines = new Map<string, ReflectedRenderPipeline>();
  private emptyBuffer: GPUBuffer;

  // Sort state: a readback in flight, and the order last uploaded.
  private sortPending = false;
  private mappingVersion = 0;
  private drawCount = 0;
  private drawVersion = -1;

  constructor(
    readonly renderer: WebGPURendererLike,
    options: WgpuSplatRendererOptions = {},
  ) {
    if (!renderer.backend?.isWebGPUBackend) {
      throw new Error(
        "WgpuSplatRenderer needs WebGPURenderer on its WebGPU backend (await renderer.init())",
      );
    }
    this.device = renderer.backend.device;
    this.registry = new KernelRegistry(this.device);
    this.sorter = new GpuSorter(this.registry);
    this.options = {
      depthTest: true,
      sortRadial: true,
      sort: "gpu",
      maxStdDev: Math.sqrt(8),
      minPixelRadius: 0,
      maxPixelRadius: 512,
      minAlpha: 0.5 / 255,
      blurAmount: 0.3,
      preBlurAmount: 0,
      falloff: 1,
      clipXY: 1.4,
      focalAdjustment: 1,
      ...options,
    };
    this.capabilities = capabilitiesOf(this.device);
    if (this.options.sort === "gpu" && !this.capabilities.gpuSort) {
      console.warn(
        `WgpuSplatRenderer: GPU sort unavailable (${this.capabilities.gpuSortReason}); sorting on the CPU`,
      );
      this.options.sort = "cpu";
    }
    this.drawUniform = this.device.createBuffer({
      label: "splat draw params",
      size: UniformWriter.for(drawModule).data.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.emptyBuffer = createStorage(this.device, 16, "empty");
  }

  add(
    source: GpuSplatSource,
    object: THREE.Object3D = new THREE.Object3D(),
  ): WgpuSplatMesh {
    const mesh: WgpuSplatMesh = {
      source,
      object,
      recolor: new THREE.Vector4(1, 1, 1, 1),
      lodIndices: null,
      lodBuffer: null,
    };
    this.meshes.push(mesh);
    this.mappingVersion += 1;
    return mesh;
  }

  remove(mesh: WgpuSplatMesh) {
    const i = this.meshes.indexOf(mesh);
    if (i >= 0) {
      this.meshes.splice(i, 1);
      mesh.lodBuffer?.destroy();
      this.mappingVersion += 1;
    }
  }

  /** Restricts a mesh to the given source indices (a LOD traversal result). */
  setLodIndices(mesh: WgpuSplatMesh, indices: Uint32Array | null) {
    mesh.lodBuffer?.destroy();
    mesh.lodIndices = indices;
    mesh.lodBuffer = indices
      ? upload(this.device, indices, "lod indices")
      : null;
    this.mappingVersion += 1;
  }

  private meshCount(m: WgpuSplatMesh) {
    return m.lodIndices ? m.lodIndices.length : m.source.count;
  }

  private ensureCapacity(total: number) {
    if (total <= this.capacity) return;
    const { maxSplats } = this.capabilities;
    if (total > maxSplats) {
      throw new Error(
        `WgpuSplatRenderer: ${total} splats is over this device's ${maxSplats} (maxStorageBufferBindingSize; see splatRequiredLimits)`,
      );
    }
    this.capacity = Math.min(
      Math.max(total, Math.ceil(this.capacity * 1.5)),
      maxSplats,
    );
    this.accumulator?.destroy();
    this.metric?.destroy();
    this.ordering?.destroy();
    this.accumulator = createStorage(
      this.device,
      this.capacity * 32,
      "accumulator",
    );
    this.metric = createStorage(this.device, this.capacity * 4, "sort metric");
    this.ordering = createStorage(this.device, this.capacity * 4, "ordering");
    this.drawCount = 0;
    this.drawVersion = -1;
  }

  /**
   * Generates, sorts and draws all meshes over what three rendered this frame:
   * into `target` when given (a RenderTarget three rendered into first; its
   * depthTexture, if any, occludes the splats), else onto the canvas.
   */
  render(camera: THREE.Camera, target?: THREE.RenderTarget) {
    const total = this.meshes.reduce((n, m) => n + this.meshCount(m), 0);
    this.stats.frames += 1;
    if (total === 0) return;
    this.ensureCapacity(total);

    camera.updateMatrixWorld();
    const cameraPos = new THREE.Vector3().setFromMatrixPosition(
      camera.matrixWorld,
    );
    const cameraDir = new THREE.Vector3(0, 0, -1).transformDirection(
      camera.matrixWorld,
    );

    const encoder = this.device.createCommandEncoder({ label: "splats" });
    this.generate(encoder, cameraPos, cameraDir);
    if (this.options.sort === "gpu") {
      const pass = encoder.beginComputePass({ label: "sort" });
      this.sorter.encode(pass, this.metric as GPUBuffer, total);
      pass.end();
      this.draw(encoder, camera, target, total);
      this.registry.submit(encoder.finish());
      return;
    }
    this.draw(encoder, camera, target);
    const version = this.mappingVersion;
    const readback = this.sortPending ? null : this.copyMetric(encoder, total);
    this.registry.submit(encoder.finish());
    if (readback) this.sortFrom(readback, total, version);
  }

  // The GPU textures to draw into: three's for a RenderTarget, or the canvas.
  private resolveTarget(target?: THREE.RenderTarget): {
    color: GPUTexture;
    depth: GPUTexture | null;
    linear: boolean;
  } {
    const { backend } = this.renderer;
    if (!target) {
      return {
        color: backend.context.getCurrentTexture(),
        depth: null,
        linear: false,
      };
    }
    const color = backend.get(target.texture)?.texture as
      | GPUTexture
      | undefined;
    if (!color) {
      throw new Error(
        "WgpuSplatRenderer: render the scene into the target first",
      );
    }
    if ((target.samples ?? 0) > 1) {
      throw new Error(
        "WgpuSplatRenderer: multisampled targets are not supported yet",
      );
    }
    const depth = target.depthTexture
      ? ((backend.get(target.depthTexture)?.texture as GPUTexture) ?? null)
      : null;
    return {
      color,
      depth,
      linear: target.texture.colorSpace !== THREE.SRGBColorSpace,
    };
  }

  private generate(
    encoder: GPUCommandEncoder,
    cameraPos: THREE.Vector3,
    cameraDir: THREE.Vector3,
  ) {
    const kernel = this.registry.get(generateModule, "generate");
    const pass = encoder.beginComputePass({ label: "generate" });
    const position = new THREE.Vector3();
    const rotation = new THREE.Quaternion();
    const scale = new THREE.Vector3();
    const inverse = new THREE.Matrix4();
    let base = 0;
    for (const mesh of this.meshes) {
      const { source, object } = mesh;
      object.updateMatrixWorld();
      object.matrixWorld.decompose(position, rotation, scale);
      const viewObject = cameraPos
        .clone()
        .applyMatrix4(inverse.copy(object.matrixWorld).invert());
      const count = this.meshCount(mesh);
      let flags = GEN_OUT_EXT;
      if (source.format === "ext") flags |= GEN_SRC_EXT;
      if (mesh.lodIndices) flags |= GEN_USE_LOD;
      if (source.lodOpacity) flags |= GEN_LOD_OPACITY;
      if (this.options.sortRadial) flags |= GEN_SORT_RADIAL;
      const params = UniformWriter.for(generateModule).setAll({
        numSplats: count,
        outBase: base,
        flags,
        numSh: source.numSh,
        srcCount: source.count,
        rotate: [rotation.x, rotation.y, rotation.z, rotation.w],
        translateScale: [
          position.x,
          position.y,
          position.z,
          (scale.x + scale.y + scale.z) / 3,
        ],
        recolor: mesh.recolor.toArray(),
        encoding: source.encoding,
        shMax: source.shMax,
        viewObject: [viewObject.x, viewObject.y, viewObject.z, 0],
        viewCenter: [cameraPos.x, cameraPos.y, cameraPos.z, 0],
        viewDir: [cameraDir.x, cameraDir.y, cameraDir.z, 0],
        outOrigin: [0, 0, 0, 0],
      });
      kernel.dispatch(pass, {
        grid: [count],
        buffers: {
          src: source.src,
          sh: source.sh ?? this.emptyBuffer,
          lodIndices: mesh.lodBuffer ?? this.emptyBuffer,
          outSplats: this.accumulator as GPUBuffer,
          sortMetric: this.metric as GPUBuffer,
        },
        uniforms: params.data,
      });
      base += count;
    }
    pass.end();
  }

  private copyMetric(encoder: GPUCommandEncoder, total: number): GPUBuffer {
    const staging = this.device.createBuffer({
      label: "sort metric readback",
      size: total * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    encoder.copyBufferToBuffer(
      this.metric as GPUBuffer,
      0,
      staging,
      0,
      total * 4,
    );
    this.sortPending = true;
    return staging;
  }

  // Back to front: largest metric first; inactive splats (+inf) left out.
  private async sortFrom(staging: GPUBuffer, total: number, version: number) {
    try {
      await staging.mapAsync(GPUMapMode.READ);
      const metric = new Float32Array(staging.getMappedRange().slice(0));
      staging.unmap();
      if (version !== this.mappingVersion || !this.ordering) return;
      const t0 = performance.now();
      const order = sortBackToFront(metric, total);
      this.stats.sortMs = performance.now() - t0;
      this.stats.sorts += 1;
      this.device.queue.writeBuffer(
        this.ordering,
        0,
        order.buffer,
        0,
        order.byteLength,
      );
      this.drawCount = order.length;
      this.drawVersion = version;
    } finally {
      staging.destroy();
      this.sortPending = false;
    }
  }

  private pipeline(
    format: GPUTextureFormat,
    depthFormat: GPUTextureFormat | null,
  ) {
    const key = `${format}/${depthFormat}`;
    let p = this.pipelines.get(key);
    if (!p) {
      const blend: GPUBlendState = {
        color: { srcFactor: "one", dstFactor: "one-minus-src-alpha" },
        alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha" },
      };
      p = createReflectedRenderPipeline(this.device, drawModule, {
        vertex: "splatVertex",
        fragment: "splatFragment",
        targets: [{ format, blend }],
        depthStencil: depthFormat
          ? {
              format: depthFormat,
              depthWriteEnabled: false,
              depthCompare: "less-equal",
            }
          : undefined,
      });
      this.pipelines.set(key, p);
    }
    return p;
  }

  private draw(
    encoder: GPUCommandEncoder,
    camera: THREE.Camera,
    renderTarget?: THREE.RenderTarget,
    gpuSorted?: number,
  ) {
    const gpu = gpuSorted !== undefined;
    if (
      !gpu &&
      (this.drawVersion !== this.mappingVersion || this.drawCount === 0)
    )
      return;
    const { color: target, depth, linear } = this.resolveTarget(renderTarget);
    const size = { x: target.width, y: target.height };
    const depthTexture = this.options.depthTest ? depth : null;
    const depthView = depthTexture?.createView();
    const depthFormat = depthTexture?.format ?? null;

    const view = camera.matrixWorldInverse;
    const viewQuat = new THREE.Quaternion().setFromRotationMatrix(view);
    const viewPos = new THREE.Vector3().setFromMatrixPosition(view);
    const p = camera.projectionMatrix.elements;
    const basis = new THREE.Matrix3().setFromMatrix4(view).elements;
    const o = this.options;
    const params = UniformWriter.for(drawModule).setAll({
      proj0: p.slice(0, 4),
      proj1: p.slice(4, 8),
      proj2: p.slice(8, 12),
      proj3: p.slice(12, 16),
      renderToViewQuat: [viewQuat.x, viewQuat.y, viewQuat.z, viewQuat.w],
      renderToViewPos: [viewPos.x, viewPos.y, viewPos.z, 0],
      renderToViewBasis0: basis.slice(0, 3),
      renderToViewBasis1: basis.slice(3, 6),
      renderToViewBasis2: basis.slice(6, 9),
      renderWidth: size.x,
      renderHeight: size.y,
      maxStdDev: o.maxStdDev,
      minPixelRadius: o.minPixelRadius,
      maxPixelRadius: o.maxPixelRadius,
      minAlpha: o.minAlpha,
      blurAmount: o.blurAmount,
      preBlurAmount: o.preBlurAmount,
      focalDistance: 0,
      apertureAngle: 0,
      clipXY: o.clipXY,
      focalAdjustment: o.focalAdjustment,
      falloff: o.falloff,
      flags:
        DRAW_EXT |
        DRAW_PREMULTIPLIED |
        (linear ? DRAW_ENCODE_LINEAR : 0) |
        ((camera as THREE.OrthographicCamera).isOrthographicCamera
          ? DRAW_ORTHOGRAPHIC
          : 0),
    });
    this.device.queue.writeBuffer(this.drawUniform, 0, params.data);

    const rp = this.pipeline(target.format, depthFormat);
    const groups = createBindGroups(this.device, rp, {
      ordering: gpu ? this.sorter.ordering : (this.ordering as GPUBuffer),
      splats: this.accumulator as GPUBuffer,
      params: this.drawUniform,
    });
    const pass = encoder.beginRenderPass({
      label: "splats",
      colorAttachments: [
        { view: target.createView(), loadOp: "load", storeOp: "store" },
      ],
      depthStencilAttachment: depthView
        ? {
            view: depthView,
            depthReadOnly: true,
            ...(depthFormat?.includes("stencil")
              ? { stencilReadOnly: true }
              : {}),
          }
        : undefined,
    });
    pass.setPipeline(rp.pipeline);
    groups.forEach((g, i) => pass.setBindGroup(i, g));
    if (gpu) {
      pass.drawIndirect(this.sorter.drawArgs, 0);
    } else {
      pass.draw(4, this.drawCount);
    }
    this.stats.draws += 1;
    this.stats.drawn = gpu ? (gpuSorted as number) : this.drawCount;
    pass.end();
  }

  dispose() {
    for (const m of this.meshes) m.lodBuffer?.destroy();
    this.accumulator?.destroy();
    this.metric?.destroy();
    this.ordering?.destroy();
    this.drawUniform.destroy();
    this.emptyBuffer.destroy();
    this.sorter.destroy();
    this.registry.destroy();
  }
}

export { GpuSplatSource } from "./GpuSplatSource";
export { sortBackToFront } from "./cpuSort";
