// Shape of the modules tools/slang-build writes to src/webgpu/generated/.

export type BindingKind =
  | "storage"
  | "read-only-storage"
  | "uniform"
  /** Added at runtime (e.g. dyno uniforms), bound via DispatchArgs.bindings. */
  | "external"
  | "unsupported";

export interface UniformField {
  readonly name: string;
  readonly offset: number;
  readonly size: number;
  /** "float32", "uint32" or "int32" for scalars, vectors and matrices. */
  readonly scalar?: string;
  readonly components?: number;
  /** Set instead of scalar for structs and arrays, which can't be packed by name. */
  readonly kind?: string;
}

export interface BindingReflection {
  readonly name: string;
  readonly group: number;
  readonly binding: number;
  readonly kind: BindingKind;
  /** Storage buffers: bytes per element. */
  readonly elementBytes?: number;
  /** Uniform blocks: size of the block in bytes. */
  readonly bytes?: number;
  readonly fields?: readonly UniformField[];
  readonly type?: string;
  /** External bindings: the bind group layout entry, less binding/visibility. */
  readonly layout?: Omit<GPUBindGroupLayoutEntry, "binding" | "visibility">;
}

export interface EntryReflection {
  readonly name: string;
  readonly stage: "compute" | "vertex" | "fragment" | string;
  readonly workgroupSize: readonly number[] | null;
  readonly uses: readonly string[];
  /** Compute entries: workgroup storage they declare, as WebGPU counts it. */
  readonly workgroupStorageBytes?: number;
}

export interface KernelReflection {
  readonly entries: readonly EntryReflection[];
  readonly bindings: readonly BindingReflection[];
}

export interface KernelModule {
  readonly name: string;
  readonly wgsl: string;
  readonly reflection: KernelReflection;
}
