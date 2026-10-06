// SparkRenderer on three's WebGPURenderer: the scene's SplatMeshes and
// SplatGenerators drawn by WgpuSplatRenderer inside three's own render pass.
//
// SparkRenderer stays a transparent THREE.Mesh in the scene, but with an
// empty instanced geometry three never draws. three calls its onBeforeRender
// while recording its render pass, between the opaque objects and the
// transparent ones sorted after it; there we submit generate and sort (they
// run before three's pass, submitted when the render finishes) and record the
// splat draw into three's open pass. The splats then test against the depth
// three is using, on the canvas, in render targets and in post-processing
// passes alike, and the user's renderer.render(scene, camera) is unchanged.
//
// Each frame the scene is walked as SparkRenderer.prepareGenerate does:
// visible SplatMeshes become WgpuSplatRenderer meshes (their PackedSplats or
// ExtSplats uploaded once, shared by meshes drawing the same splats), LoD and
// paged ones go through WgpuLod, their modifiers, edits and skinning through
// splatMeshDyno, and other SplatGenerators through splatGeneratorDyno.

import * as THREE from "three";
import type { ExtSplats } from "../ExtSplats";
import { PackedSplats } from "../PackedSplats";
import type { SparkRenderer } from "../SparkRenderer";
import { type SplatEdit, isSplatEdit } from "../SplatEdit";
import { SplatGenerator } from "../SplatGenerator";
import { SplatMesh } from "../SplatMesh";
import { GpuSplatSource } from "./GpuSplatSource";
import { WgpuLod, type WgpuLodMesh } from "./WgpuLod";
import {
  type SplatPassTarget,
  type WgpuSplatMesh,
  WgpuSplatRenderer,
} from "./WgpuSplatRenderer";
import type { WgpuDyno } from "./dyno/DynoKernels";
import { splatGeneratorDyno, splatMeshDyno } from "./dyno/adapters";

// The parts of three's WebGPURenderer (r180) this module uses. The render
// context, its backend data and the backend utils are internal to three.
interface RenderContextLike {
  textures: THREE.Texture[] | null;
  renderTarget?: THREE.RenderTarget | null;
  depthTexture: THREE.Texture | null;
  width: number;
  height: number;
  viewport: boolean;
  viewportValue: THREE.Vector4;
}

interface BackendLike {
  isWebGPUBackend?: boolean;
  device: GPUDevice;
  context: GPUCanvasContext;
  get(resource: object): {
    texture?: GPUTexture;
    format?: GPUTextureFormat;
    currentPass?: GPURenderPassEncoder | null;
    descriptor?: GPURenderPassDescriptor;
    currentSets?: unknown;
  };
  utils: {
    getCurrentColorFormat(rc: RenderContextLike): GPUTextureFormat;
    getCurrentDepthStencilFormat(
      rc: RenderContextLike,
    ): GPUTextureFormat | undefined;
    getSampleCountRenderContext(rc: RenderContextLike): number;
    getCurrentColorSpace(rc: RenderContextLike): string;
  };
  pipelineUtils?: { _activePipelines?: WeakMap<object, unknown> };
}

export interface WebGPURendererLike {
  isWebGPURenderer?: boolean;
  backend: BackendLike;
  info: { frame: number };
  xr?: { isPresenting?: boolean };
  outputColorSpace: string;
  getOutputRenderTarget(): THREE.RenderTarget | null;
  _currentRenderContext?: RenderContextLike | null;
  /** Where three renders the scene before its output (color space) pass. */
  _frameBufferTarget?: THREE.RenderTarget | null;
}

/** Whether `renderer` is three's WebGPURenderer (either backend). */
export function isWebGPURenderer(
  renderer: unknown,
): renderer is WebGPURendererLike {
  return !!(renderer as WebGPURendererLike | undefined)?.isWebGPURenderer;
}

type BaseSplats = PackedSplats | ExtSplats;

// A SplatMesh or SplatGenerator as drawn this frame.
interface Entry {
  /** What the entry was built from; rebuilt when it changes. */
  key: unknown[];
  dyno: WgpuDyno;
  mesh?: WgpuSplatMesh;
  lodMesh?: WgpuLodMesh;
  /** Uploaded splats shared with other meshes drawing the same ones. */
  shared?: SharedSource;
  /** A WgpuLod.add in flight. */
  pending?: boolean;
  detached?: boolean;
}

interface SharedSource {
  splats: BaseSplats;
  source: GpuSplatSource;
  users: number;
}

const warned = new Set<string>();
function warnOnce(message: string) {
  if (warned.has(message)) return;
  warned.add(message);
  console.warn(`SparkRenderer (WebGPU): ${message}`);
}

export class SparkWebGPU {
  splats?: WgpuSplatRenderer;
  lod?: WgpuLod;
  private entries = new Map<SplatGenerator, Entry>();
  private shared = new Map<BaseSplats, SharedSource>();
  private initWatched = new WeakSet<SplatMesh>();
  private globalEdits: SplatEdit[] = [];
  private lastFrame = -1;
  private lastTime = performance.now() / 1000;
  private failed = false;

  constructor(
    readonly spark: SparkRenderer,
    readonly renderer: WebGPURendererLike,
  ) {}

  /** SparkRenderer.onBeforeRender on WebGPU: syncs the scene and draws. */
  onBeforeRender(scene: THREE.Scene, camera: THREE.Camera) {
    if (this.failed) return;
    const { renderer } = this;
    const { backend } = renderer;
    if (!backend.isWebGPUBackend) {
      this.failed = true;
      console.error(
        "SparkRenderer: WebGPURenderer fell back to WebGL; splats need its WebGPU backend",
      );
      return;
    }
    if (
      (camera as THREE.ArrayCamera).isArrayCamera ||
      renderer.xr?.isPresenting
    ) {
      warnOnce("XR and array cameras are not supported yet");
      return;
    }
    const rc = renderer._currentRenderContext;
    const data = rc ? backend.get(rc) : undefined;
    const pass = data?.currentPass;
    if (!rc || !data || !pass) {
      warnOnce("no open render pass to draw into (render bundles?)");
      return;
    }

    const splats = this.ensureRenderer();
    const width = rc.viewport ? rc.viewportValue.z : rc.width;
    const height = rc.viewport ? rc.viewportValue.w : rc.height;

    const frame = renderer.info.frame;
    if (frame !== this.lastFrame) {
      // Once per frame, as SparkRenderer updates on a new frame.
      this.lastFrame = frame;
      this.applyOptions();
      this.sync(scene, camera);
      if (this.lod?.meshes.length && this.spark.enableDriveLod) {
        this.lod.update(camera, { x: width, y: height });
      }
    }

    const { utils } = backend;
    const target: SplatPassTarget = {
      format: utils.getCurrentColorFormat(rc),
      depthFormat: data.descriptor?.depthStencilAttachment
        ? (utils.getCurrentDepthStencilFormat(rc) ?? null)
        : null,
      sampleCount: utils.getSampleCountRenderContext(rc),
      width,
      height,
      linear: utils.getCurrentColorSpace(rc) !== THREE.SRGBColorSpace,
      extraFormats: rc.textures
        ?.slice(1)
        .map((t) => backend.get(t).format as GPUTextureFormat),
      depthCompare: (camera as { reversedDepth?: boolean }).reversedDepth
        ? "greater-equal"
        : "less-equal",
    };
    this.spark.dirty = false;
    if (
      rc.renderTarget &&
      rc.renderTarget === renderer._frameBufferTarget &&
      !renderer.getOutputRenderTarget() &&
      target.sampleCount === 1 &&
      (scene as THREE.Scene).isScene
    ) {
      this.drawAfterOutput(scene, camera, rc);
      return;
    }
    splats.renderInPass(camera, pass, target);
    // three skips setting a pipeline or bind group it believes is still set.
    backend.pipelineUtils?._activePipelines?.delete(pass);
    data.currentSets = {
      attributes: {},
      bindingGroups: [],
      pipeline: null,
      index: null,
    };
  }

  // three renders the scene into a linear half-float target, then converts it
  // to the canvas. Splats are blended in sRGB space, as WebGL Spark does on
  // the canvas and as they were trained, so they go on the canvas after that
  // output pass, tested against the scene's depth, which three keeps.
  // Transparent objects in front of the splats are then drawn under them.
  private drawAfterOutput(
    scene: THREE.Scene,
    camera: THREE.Camera,
    rc: RenderContextLike,
  ) {
    const { backend } = this.renderer;
    const depth = rc.depthTexture
      ? (backend.get(rc.depthTexture).texture ?? null)
      : null;
    const viewport = rc.viewport ? rc.viewportValue.clone() : null;
    const size = { x: rc.width, y: rc.height };
    const previous = scene.onAfterRender;
    // Restored first (as an own property when it was the prototype's).
    scene.onAfterRender = (...args) => {
      scene.onAfterRender = previous;
      previous.apply(scene, args);
      this.drawOnCanvas(camera, depth, viewport, size);
    };
  }

  private drawOnCanvas(
    camera: THREE.Camera,
    depth: GPUTexture | null,
    viewport: THREE.Vector4 | null,
    size: { x: number; y: number },
  ) {
    const { renderer } = this;
    const splats = this.splats as WgpuSplatRenderer;
    const color = renderer.backend.context.getCurrentTexture();
    const encoder = splats.device.createCommandEncoder({
      label: "splats on canvas",
    });
    const depthFormat = depth?.format ?? null;
    const pass = encoder.beginRenderPass({
      label: "splats",
      colorAttachments: [
        { view: color.createView(), loadOp: "load", storeOp: "store" },
      ],
      depthStencilAttachment: depth
        ? {
            view: depth.createView(),
            depthReadOnly: true,
            ...(depthFormat?.includes("stencil")
              ? { stencilReadOnly: true }
              : {}),
          }
        : undefined,
    });
    if (viewport) {
      pass.setViewport(viewport.x, viewport.y, viewport.z, viewport.w, 0, 1);
    }
    splats.renderInPass(camera, pass, {
      format: color.format,
      depthFormat,
      sampleCount: 1,
      width: viewport ? viewport.z : size.x,
      height: viewport ? viewport.w : size.y,
      linear: renderer.outputColorSpace !== THREE.SRGBColorSpace,
      depthCompare: (camera as { reversedDepth?: boolean }).reversedDepth
        ? "greater-equal"
        : "less-equal",
    });
    pass.end();
    splats.device.queue.submit([encoder.finish()]);
  }

  private ensureRenderer() {
    if (!this.splats) {
      const spark = this.spark;
      this.splats = new WgpuSplatRenderer(
        this.renderer as unknown as ConstructorParameters<
          typeof WgpuSplatRenderer
        >[0],
        { depthTest: spark.material.depthTest },
      );
      this.lod = new WgpuLod(this.splats, {
        pagedExtSplats: spark.pagedExtSplats,
        maxPagedSplats: spark.maxPagedSplats,
        numLodFetchers: spark.numLodFetchers,
        onDirty: () => spark.setDirty(),
      });
      if (spark.covSplats || spark.accumExtSplats) {
        warnOnce("covSplats and accumExtSplats are ignored");
      }
      if (spark.enable2DGS) warnOnce("enable2DGS is not supported yet");
    }
    return this.splats;
  }

  // SparkRenderer's settings, which apps change at any time.
  private applyOptions() {
    const spark = this.spark;
    const o = (this.splats as WgpuSplatRenderer).options;
    o.maxStdDev = spark.maxStdDev;
    o.minPixelRadius = spark.minPixelRadius;
    o.maxPixelRadius = spark.maxPixelRadius;
    o.minAlpha = spark.minAlpha;
    o.blurAmount = spark.blurAmount;
    o.preBlurAmount = spark.preBlurAmount;
    o.falloff = spark.falloff;
    o.clipXY = spark.clipXY;
    o.focalAdjustment = spark.focalAdjustment;
    o.focalDistance = spark.focalDistance;
    o.apertureAngle = spark.apertureAngle;
    o.sortRadial = spark.sortRadial;
    o.lodInflate = spark.lodInflate;
    const l = (this.lod as WgpuLod).options;
    l.lodSplatCount = spark.lodSplatCount;
    l.lodSplatScale = spark.lodSplatScale;
    l.lodRenderScale = spark.lodRenderScale;
    l.lodTraverseMode = spark.lodTraverseMode;
    l.behindFoveate = spark.behindFoveate;
    l.coneFov0 = spark.coneFov0;
    l.coneFov = spark.coneFov;
    l.coneFoveate = spark.coneFoveate;
    l.enableLodFetching = spark.enableLodFetching;
  }

  // Matches the WgpuSplatRenderer meshes to the scene's visible generators.
  private sync(scene: THREE.Scene, camera: THREE.Camera) {
    const inLayers = (node: THREE.Object3D) =>
      !camera.layers || camera.layers.test(node.layers);
    const all = new Set<SplatGenerator>();
    scene.traverse((node) => {
      if (node instanceof SplatGenerator && inLayers(node)) all.add(node);
    });
    const visible: SplatGenerator[] = [];
    const globalEdits: SplatEdit[] = [];
    scene.traverseVisible((node) => {
      if (node instanceof SplatGenerator && inLayers(node)) visible.push(node);
      if (isSplatEdit(node)) {
        let ancestor = node.parent;
        while (ancestor && !(ancestor instanceof SplatMesh)) {
          ancestor = ancestor.parent;
        }
        if (!ancestor) globalEdits.push(node);
      }
    });
    this.globalEdits = globalEdits;

    const time = performance.now() / 1000;
    const deltaTime = time - this.lastTime;
    this.lastTime = time;
    for (const node of all) {
      if (node instanceof SplatMesh) {
        // SplatMesh.update runs onFrame; frameUpdate is replaced here.
        node.onFrame?.({ mesh: node, time, deltaTime });
      }
    }

    const shown = new Set<SplatGenerator>();
    for (const node of visible) {
      if (this.updateEntry(node)) shown.add(node);
    }
    for (const [node, entry] of this.entries) {
      if (shown.has(node)) continue;
      this.detach(entry);
      if (!all.has(node)) this.entries.delete(node);
    }
  }

  // Brings a visible generator's entry up to date; whether it is drawn.
  private updateEntry(node: SplatGenerator): boolean {
    let entry = this.entries.get(node);
    let key: unknown[];
    let build: () => Entry | null;
    if (node instanceof SplatMesh) {
      if (!node.isInitialized) {
        this.watchInit(node);
        return false;
      }
      if (node.covSplats) warnOnce("covSplats meshes are drawn as ext splats");
      const base = node.packedSplats ?? node.extSplats;
      const lodSplats = base?.lodSplats;
      const useLod = this.spark.enableLod && !!(node.paged || lodSplats);
      const splats = node.paged ?? base;
      if (node.splats && node.splats !== splats) {
        warnOnce("SplatMeshes with a custom SplatSource are not supported");
        return false;
      }
      if (!splats || (!useLod && (base as BaseSplats).numSplats === 0)) {
        return false;
      }
      key = [useLod, splats, lodSplats];
      build = () => {
        const dyno = splatMeshDyno(node, {
          globalEdits: () => this.globalEdits,
        });
        if (useLod) return this.addLod(node, { key: [], dyno });
        return this.addBase(node, base as BaseSplats, dyno);
      };
    } else {
      if (!node.generator || node.numSplats <= 0) return false;
      key = [node.numSplats];
      build = () => {
        const dyno = splatGeneratorDyno(node);
        const splats = this.splats as WgpuSplatRenderer;
        return {
          key,
          dyno,
          mesh: splats.addGenerator(node.numSplats, dyno, node),
        };
      };
    }

    if (entry && !sameKey(entry.key, key)) {
      this.detach(entry);
      entry = undefined;
    }
    const fresh = !entry;
    if (!entry) {
      const built = build();
      if (!built) return false;
      built.key = key;
      entry = built;
      this.entries.set(node, entry);
    } else if (entry.detached) {
      this.reattach(node, entry);
    }

    if (node instanceof SplatMesh) {
      // SplatMesh.updateGenerator(): construct its dyno blocks again.
      if (node.generatorDirty) {
        node.generatorDirty = false;
        if (!fresh) this.renewMesh(node, entry);
      }
      const mesh = entry.mesh ?? entry.lodMesh?.mesh;
      mesh?.recolor.set(
        node.recolor.r,
        node.recolor.g,
        node.recolor.b,
        node.opacity,
      );
      if (entry.shared && mesh) {
        this.refreshShared(entry.shared);
        if (mesh.source !== entry.shared.source) {
          mesh.source = entry.shared.source;
          this.splats?.markDirty();
        }
      }
      const lodMesh = entry.lodMesh;
      if (lodMesh) {
        lodMesh.lodScale = node.lodScale;
        lodMesh.behindFoveate = node.behindFoveate;
        lodMesh.coneFov0 = node.coneFov0;
        lodMesh.coneFov = node.coneFov;
        lodMesh.coneFoveate = node.coneFoveate;
        this.lod?.setEnableLod(lodMesh, node.enableLod !== false);
      }
    }
    return true;
  }

  // A new WgpuSplatMesh for the same splats: DynoKernels compiles a mesh's
  // graphs once, keyed by the mesh.
  private renewMesh(node: SplatMesh, entry: Entry) {
    const splats = this.splats as WgpuSplatRenderer;
    const old = entry.mesh ?? entry.lodMesh?.mesh;
    if (!old) return;
    const mesh = splats.add(old.source, node, entry.dyno);
    if (old.lodIndices) splats.setLodIndices(mesh, old.lodIndices);
    splats.remove(old);
    if (entry.lodMesh) entry.lodMesh.mesh = mesh;
    else entry.mesh = mesh;
  }

  private addBase(node: SplatMesh, base: BaseSplats, dyno: WgpuDyno): Entry {
    const shared = this.acquire(base);
    const mesh = (this.splats as WgpuSplatRenderer).add(
      shared.source,
      node,
      dyno,
    );
    return { key: [], dyno, mesh, shared };
  }

  // Fills `entry` once WgpuLod has the mesh (after loading .rad metadata).
  private addLod(node: SplatMesh, entry: Entry): Entry {
    entry.pending = true;
    const lod = this.lod as WgpuLod;
    const splats = node.paged ?? node.packedSplats ?? node.extSplats;
    lod
      .add(splats as PackedSplats, node, { lodScale: node.lodScale })
      .then((lodMesh) => {
        entry.pending = false;
        if (entry.detached) {
          lod.remove(lodMesh);
          return;
        }
        lodMesh.mesh.dyno = entry.dyno;
        entry.lodMesh = lodMesh;
        this.spark.setDirty();
      })
      .catch((error) => {
        console.error("SparkRenderer (WebGPU): LoD mesh failed", error);
      });
    return entry;
  }

  private reattach(node: SplatGenerator, entry: Entry) {
    entry.detached = false;
    const splats = this.splats as WgpuSplatRenderer;
    if (entry.shared) {
      // Uploaded again if no other mesh kept the splats.
      entry.shared = this.acquire(entry.shared.splats);
      entry.mesh = splats.add(entry.shared.source, node, entry.dyno);
    } else if (node instanceof SplatMesh) {
      // LoD: added again, as WgpuLod releases its trees on removal.
      if (!entry.pending) this.addLod(node, entry);
    } else {
      entry.mesh = splats.addGenerator(node.numSplats, entry.dyno, node);
    }
  }

  private detach(entry: Entry) {
    if (entry.detached) return;
    entry.detached = true;
    if (entry.mesh) this.splats?.remove(entry.mesh);
    entry.mesh = undefined;
    if (entry.lodMesh) this.lod?.remove(entry.lodMesh);
    entry.lodMesh = undefined;
    if (entry.shared) this.release(entry.shared);
  }

  private acquire(splats: BaseSplats): SharedSource {
    let shared = this.shared.get(splats);
    if (!shared) {
      shared = { splats, source: this.upload(splats), users: 0 };
      this.shared.set(splats, shared);
    }
    shared.users += 1;
    return shared;
  }

  private release(shared: SharedSource) {
    shared.users -= 1;
    if (shared.users > 0) return;
    shared.source.destroy();
    this.shared.delete(shared.splats);
  }

  private upload(splats: BaseSplats): GpuSplatSource {
    const device = (this.splats as WgpuSplatRenderer).device;
    if (splats instanceof PackedSplats) {
      // The WebGL path consumes needsUpdate when it uploads its texture.
      splats.needsUpdate = false;
      return GpuSplatSource.fromPackedSplats(device, splats);
    }
    return GpuSplatSource.fromExtSplats(device, splats);
  }

  // Re-uploads splats edited since (PackedSplats.needsUpdate, a new count).
  private refreshShared(shared: SharedSource) {
    const { splats } = shared;
    const edited =
      (splats instanceof PackedSplats && splats.needsUpdate) ||
      splats.numSplats !== shared.source.count;
    if (!edited) return;
    shared.source.destroy();
    shared.source = this.upload(splats);
  }

  private watchInit(mesh: SplatMesh) {
    if (this.initWatched.has(mesh)) return;
    this.initWatched.add(mesh);
    const onInitialized = () => {
      mesh.removeEventListener("initialized", onInitialized);
      this.initWatched.delete(mesh);
      this.spark.setDirty();
    };
    mesh.addEventListener("initialized", onInitialized);
  }

  dispose() {
    for (const entry of this.entries.values()) this.detach(entry);
    this.entries.clear();
    this.lod?.dispose();
    this.splats?.dispose();
    this.lod = undefined;
    this.splats = undefined;
  }
}

function sameKey(a: unknown[], b: unknown[]) {
  return a.length === b.length && a.every((x, i) => x === b[i]);
}
