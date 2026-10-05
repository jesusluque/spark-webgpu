// What this WebGPU device can do, in the terms the splat renderer and the fx
// host care about, so they pick a path or cap a size instead of failing at
// validation. One place to look, because the answers differ by browser and
// by request: a device gets the spec's default limits (128 MiB storage
// bindings, 8 storage buffers a stage) unless it asks for the adapter's, and
// three's WebGPURenderer asks for none. examples/webgpu/caps.html prints both
// sets in any browser.
//
// Nothing here needs an optional feature. The kernels use plain WGSL:
// atomics (sort_radix's histogram and the indirect count), 256-thread
// workgroups, at most 4.2 KiB of workgroup storage, two read-only storage
// buffers in the vertex stage, drawIndirect from a buffer compute wrote with
// firstInstance 0 (so no indirect-first-instance), and rgba16float render
// targets with depth. Subgroups, timestamp queries, float32-filterable and
// bgra8unorm-storage are reported for callers that want them, never assumed.

/** Workgroup threads sort_radix's histogram and scatter entries declare. */
export const SORT_WORKGROUP_THREADS = 256;
/**
 * Largest workgroup storage of a sort entry: sHist (16 atomic u32) + sScan
 * (128 vec4<u32>) + sBase (16 u32) + sScanBuf (512 u32), from the WGSL.
 */
export const SORT_WORKGROUP_BYTES = 16 * 4 + 128 * 16 + 16 * 4 + 512 * 4;
/** Bytes per splat in the largest per-splat buffer (the ext accumulator). */
export const ACCUMULATOR_BYTES_PER_SPLAT = 32;
/** Packed SH: three uint4 per splat. */
export const SH_BYTES_PER_SPLAT = 48;
/** Storage buffers the draw's vertex stage reads (ordering, splats). */
export const VERTEX_STORAGE_BUFFERS = 2;
/** Most storage buffers one kernel binds (generate, the fx bridge). */
export const COMPUTE_STORAGE_BUFFERS = 5;

type LimitsLike = Partial<Record<keyof GPUSupportedLimits, number>> &
  Record<string, unknown>;

export interface GpuCapabilities {
  readonly features: ReadonlySet<string>;
  readonly limits: Readonly<Record<string, number>>;
  /** The GPU sort fits the device; otherwise sort on the CPU. */
  readonly gpuSort: boolean;
  /** Why gpuSort is false, for a message. */
  readonly gpuSortReason: string;
  /** Most splats one draw can hold, from the storage binding and buffer limits. */
  readonly maxSplats: number;
  /** Most splats with spherical harmonics. */
  readonly maxSplatsWithSh: number;
  /** Storage buffers the vertex stage may bind. */
  readonly vertexStorageBuffers: number;
  /** The draw's vertex-stage storage reads are allowed. */
  readonly vertexStorage: boolean;
  /** Largest square fx picture (16 bytes a pixel) one binding holds. */
  readonly maxFxPixels: number;
  readonly subgroups: boolean;
  readonly timestampQuery: boolean;
  readonly float32Filterable: boolean;
  readonly bgra8unormStorage: boolean;
  readonly shaderF16: boolean;
}

function readLimits(limits: GPUSupportedLimits | LimitsLike) {
  const out: Record<string, number> = {};
  // GPUSupportedLimits keeps its values on the prototype, so a for-in over
  // it finds them where Object.keys would not.
  for (const k in limits) {
    const v = (limits as Record<string, unknown>)[k];
    if (typeof v === "number") out[k] = v;
  }
  return out;
}

/** Capabilities of a device (what it was created with) or an adapter (what it could ask for). */
export function capabilitiesOf(
  gpu:
    | Pick<GPUDevice, "features" | "limits">
    | Pick<GPUAdapter, "features" | "limits">,
): GpuCapabilities {
  const features = new Set<string>(gpu.features as unknown as Iterable<string>);
  const limits = readLimits(gpu.limits);
  const lim = (k: string, fallback: number) => limits[k] ?? fallback;

  const reasons: string[] = [];
  if (lim("maxComputeInvocationsPerWorkgroup", 256) < SORT_WORKGROUP_THREADS) {
    reasons.push("maxComputeInvocationsPerWorkgroup < 256");
  }
  if (lim("maxComputeWorkgroupSizeX", 256) < SORT_WORKGROUP_THREADS) {
    reasons.push("maxComputeWorkgroupSizeX < 256");
  }
  if (lim("maxComputeWorkgroupStorageSize", 16384) < SORT_WORKGROUP_BYTES) {
    reasons.push(`maxComputeWorkgroupStorageSize < ${SORT_WORKGROUP_BYTES}`);
  }
  if (lim("maxStorageBuffersPerShaderStage", 8) < COMPUTE_STORAGE_BUFFERS) {
    reasons.push("maxStorageBuffersPerShaderStage < 5");
  }

  const binding = lim("maxStorageBufferBindingSize", 134217728);
  const buffer = lim("maxBufferSize", 268435456);
  const perBuffer = Math.min(binding, buffer);
  // generate runs one 256-thread group per 256 splats along x only.
  const dispatchable = lim("maxComputeWorkgroupsPerDimension", 65535) * 256;
  // Compatibility-mode devices report vertex-stage storage separately, and
  // may allow none; core WebGPU uses the per-stage limit.
  const vertexStorageBuffers =
    limits.maxStorageBuffersInVertexStage ??
    lim("maxStorageBuffersPerShaderStage", 8);

  return {
    features,
    limits,
    gpuSort: reasons.length === 0,
    gpuSortReason: reasons.join(", "),
    maxSplats: Math.min(
      Math.floor(perBuffer / ACCUMULATOR_BYTES_PER_SPLAT),
      dispatchable,
    ),
    maxSplatsWithSh: Math.min(
      Math.floor(perBuffer / SH_BYTES_PER_SPLAT),
      dispatchable,
    ),
    vertexStorageBuffers,
    vertexStorage: vertexStorageBuffers >= VERTEX_STORAGE_BUFFERS,
    maxFxPixels: Math.floor(perBuffer / 16),
    subgroups: features.has("subgroups"),
    timestampQuery: features.has("timestamp-query"),
    float32Filterable: features.has("float32-filterable"),
    bgra8unormStorage: features.has("bgra8unorm-storage"),
    shaderF16: features.has("shader-f16"),
  };
}

/**
 * Limits worth asking an adapter for when creating the device (three's
 * `new WebGPURenderer({ requiredLimits })`): the storage sizes that bound how
 * many splats fit, raised to what the adapter allows. A device gets the
 * spec defaults (128 MiB bindings, 4M splats) unless it asks.
 */
export function splatRequiredLimits(
  adapter: Pick<GPUAdapter, "limits">,
): Record<string, number> {
  const a = readLimits(adapter.limits);
  const out: Record<string, number> = {};
  for (const k of [
    "maxStorageBufferBindingSize",
    "maxBufferSize",
    "maxStorageBuffersPerShaderStage",
  ]) {
    if (a[k] !== undefined) out[k] = a[k];
  }
  return out;
}

/** A canvas format and alpha mode that every WebGPU implementation accepts. */
export function canvasConfiguration(): {
  format: GPUTextureFormat;
  alphaMode: GPUCanvasAlphaMode;
} {
  // bgra8unorm on Apple GPUs in both browsers; rgba8unorm elsewhere. Both
  // are always valid canvas formats, and "premultiplied" is what the splat
  // draw writes and what Safari and Chrome both support.
  const format =
    typeof navigator !== "undefined" && navigator.gpu
      ? navigator.gpu.getPreferredCanvasFormat()
      : "bgra8unorm";
  return { format, alphaMode: "premultiplied" };
}
