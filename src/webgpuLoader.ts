// Spark's WebGPU backend (src/webgpu) is a chunk of its own that WebGL apps
// never download: SparkRenderer loads it with import() when given a
// WebGPURenderer. Importing "@sparkjsdev/spark/webgpu" loads it up front.

import type { WebGPUBackend } from "./webgpu/index";
import type { WebGPURendererLike } from "./webgpu/threeRenderer";

type WebGPUModule = typeof import("./webgpu/index");

let backend: WebGPUBackend | undefined;
let loading: Promise<WebGPUModule> | undefined;

/** Whether `renderer` is three's WebGPURenderer (either backend). */
export function isWebGPURenderer(
  renderer: unknown,
): renderer is WebGPURendererLike {
  return !!(renderer as WebGPURendererLike | undefined)?.isWebGPURenderer;
}

/** @internal Called by src/webgpu/index.ts when it is evaluated. */
export function registerWebGPU(loaded: WebGPUBackend) {
  backend = loaded;
}

/** @internal The WebGPU backend if it has loaded, else undefined. */
export function loadedWebGPU(): WebGPUBackend | undefined {
  return backend;
}

/**
 * Loads Spark's WebGPU backend, resolving to the "@sparkjsdev/spark/webgpu"
 * module (WgpuSplatRenderer, WgpuLod, fx, ...). SparkRenderer loads it by
 * itself when given a WebGPURenderer, but once this has resolved (or that
 * module was imported statically) its `webgpu` is set from the constructor
 * on rather than after `spark.webgpuReady`.
 */
export function loadWebGPU(): Promise<WebGPUModule> {
  loading ??= import("./webgpu/index");
  return loading;
}

/** @internal loadWebGPU() resolving to what SparkRenderer uses. */
export async function loadWebGPUBackend(): Promise<WebGPUBackend> {
  await loadWebGPU();
  if (!backend) throw new Error("Spark: the WebGPU backend didn't register");
  return backend;
}
