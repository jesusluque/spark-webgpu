// Compute pipelines built from the modules tools/slang-build generates, with
// each dispatch checked against the kernel's reflection before it is recorded.
//
// This is aopenfx's Gpu::run carried to WebGPU: the grid counts threads, not
// workgroups (the registry divides by the kernel's workgroup size), buffers are
// bound by the names the Slang source gives them, and a dispatch whose buffers
// or uniform block disagree with the kernel is refused with an error instead
// of rendering black.

import type {
  BindingReflection,
  EntryReflection,
  KernelModule,
} from "./KernelModule";

export type Grid = readonly [number, number?, number?];

export interface DispatchArgs {
  /** Threads to run in x, y, z; the registry rounds up to whole workgroups. */
  grid: Grid;
  /** Storage buffers by binding name. */
  buffers?: Record<string, GPUBuffer>;
  /** Contents of the kernel's uniform block, exactly its reflected size. */
  uniforms?: ArrayBufferView | ArrayBuffer;
  /** Resources for "external" bindings, by name. */
  bindings?: Record<string, GPUBindingResource>;
}

/** Problems with a dispatch, or an empty list when it matches the kernel. */
export function validateDispatch(
  module: KernelModule,
  entry: EntryReflection,
  args: DispatchArgs,
): string[] {
  const errors: string[] = [];
  const where = `${module.name}:${entry.name}`;
  for (const b of usedBindings(module, entry)) {
    if (b.kind === "external") {
      if (!args.bindings?.[b.name]) {
        errors.push(`${where}: missing binding '${b.name}'`);
      }
    } else if (b.kind === "uniform") {
      const bytes = args.uniforms?.byteLength ?? 0;
      if (bytes !== b.bytes) {
        errors.push(
          `${where}: uniform block '${b.name}' is ${b.bytes} bytes, got ${bytes}`,
        );
      }
    } else if (b.kind === "storage" || b.kind === "read-only-storage") {
      const buf = args.buffers?.[b.name];
      if (!buf) {
        errors.push(`${where}: missing buffer '${b.name}'`);
      } else if (b.elementBytes && buf.size % b.elementBytes !== 0) {
        errors.push(
          `${where}: buffer '${b.name}' is ${buf.size} bytes, not a multiple of its ${b.elementBytes}-byte element`,
        );
      }
    } else {
      errors.push(
        `${where}: binding '${b.name}' has unsupported type ${b.type}`,
      );
    }
  }
  const [x, y = 1, z = 1] = args.grid;
  if (![x, y, z].every((n) => Number.isInteger(n) && n >= 0)) {
    errors.push(`${where}: grid ${args.grid.join("x")} is not whole threads`);
  }
  return errors;
}

/** The bindings an entry point uses, in group/binding order. */
export function usedBindings(
  module: KernelModule,
  entry: EntryReflection,
): BindingReflection[] {
  const uses = new Set(entry.uses);
  return module.reflection.bindings
    .filter((b) => uses.has(b.name))
    .sort((a, b) => a.group - b.group || a.binding - b.binding);
}

/**
 * One bind group layout per group up to the highest one `bindings` use:
 * buffers by their kind, "external" bindings by their own layout.
 */
export function bindGroupLayouts(
  device: GPUDevice,
  label: string,
  bindings: readonly BindingReflection[],
  visibility: (b: BindingReflection) => GPUShaderStageFlags,
): GPUBindGroupLayout[] {
  const groupCount = bindings.reduce((n, b) => Math.max(n, b.group + 1), 0);
  return Array.from({ length: groupCount }, (_, g) =>
    device.createBindGroupLayout({
      label: `${label}@${g}`,
      entries: bindings
        .filter((b) => b.group === g)
        .map((b) => ({
          ...(b.layout ?? {
            buffer: {
              type:
                b.kind === "unsupported" || b.kind === "external"
                  ? undefined
                  : b.kind,
            },
          }),
          binding: b.binding,
          visibility: visibility(b),
        })),
    }),
  );
}

export function workgroupCount(entry: EntryReflection, grid: Grid): number[] {
  const size = entry.workgroupSize ?? [1, 1, 1];
  const [x, y = 1, z = 1] = grid;
  return [x, y, z].map((n, i) => Math.ceil(n / (size[i] || 1)));
}

/** One compute entry point of a module. */
export class Kernel {
  readonly bindings: BindingReflection[];
  readonly groups: number[];

  constructor(
    readonly registry: KernelRegistry,
    readonly module: KernelModule,
    readonly entry: EntryReflection,
    readonly pipeline: GPUComputePipeline,
    readonly layouts: GPUBindGroupLayout[],
  ) {
    this.bindings = usedBindings(module, entry);
    this.groups = [...new Set(this.bindings.map((b) => b.group))];
  }

  /** Records the dispatch into `pass`. Throws if it doesn't match the kernel. */
  dispatch(pass: GPUComputePassEncoder, args: DispatchArgs) {
    const errors = validateDispatch(this.module, this.entry, args);
    if (errors.length) throw new Error(errors.join("\n"));

    const [gx, gy, gz] = workgroupCount(this.entry, args.grid);
    if (gx === 0 || gy === 0 || gz === 0) return;
    this.bind(pass, args);
    pass.dispatchWorkgroups(gx, gy, gz);
  }

  /**
   * Records a dispatch whose workgroup counts a kernel wrote to `indirect`
   * (three u32 at `offset`); args.grid is not used.
   */
  dispatchIndirect(
    pass: GPUComputePassEncoder,
    args: Omit<DispatchArgs, "grid">,
    indirect: GPUBuffer,
    offset = 0,
  ) {
    const full = { ...args, grid: [0] as const };
    const errors = validateDispatch(this.module, this.entry, full);
    if (errors.length) throw new Error(errors.join("\n"));
    this.bind(pass, full);
    pass.dispatchWorkgroupsIndirect(indirect, offset);
  }

  private bind(pass: GPUComputePassEncoder, args: DispatchArgs) {
    const device = this.registry.device;
    pass.setPipeline(this.pipeline);
    for (const group of this.groups) {
      const entries: GPUBindGroupEntry[] = [];
      for (const b of this.bindings) {
        if (b.group !== group) continue;
        const resource: GPUBindingResource =
          b.kind === "external"
            ? (args.bindings as Record<string, GPUBindingResource>)[b.name]
            : b.kind === "uniform"
              ? this.registry.uniforms.push(args.uniforms as ArrayBufferView)
              : { buffer: (args.buffers as Record<string, GPUBuffer>)[b.name] };
        entries.push({ binding: b.binding, resource });
      }
      pass.setBindGroup(
        group,
        device.createBindGroup({ layout: this.layouts[group], entries }),
      );
    }
  }

  /** Dispatches on its own pass and submits. */
  run(args: DispatchArgs) {
    const device = this.registry.device;
    const encoder = device.createCommandEncoder({ label: this.entry.name });
    const pass = encoder.beginComputePass({ label: this.entry.name });
    this.dispatch(pass, args);
    pass.end();
    this.registry.submit(encoder.finish());
  }
}

/**
 * Uniform blocks for the dispatches recorded before the next submit, packed
 * into shared buffers at 256-byte offsets so each dispatch keeps its own.
 */
export class UniformArena {
  private chunks: GPUBuffer[] = [];
  private chunk = 0;
  private offset = 0;

  constructor(
    readonly device: GPUDevice,
    readonly chunkBytes = 64 * 1024,
  ) {}

  push(data: ArrayBufferView | ArrayBuffer): GPUBufferBinding {
    const bytes = data.byteLength;
    const align = this.device.limits.minUniformBufferOffsetAlignment;
    if (bytes > this.chunkBytes) {
      throw new Error(
        `uniform block of ${bytes} bytes is over the arena chunk`,
      );
    }
    if (this.offset + bytes > this.chunkBytes) {
      this.chunk += 1;
      this.offset = 0;
    }
    if (this.chunk === this.chunks.length) {
      this.chunks.push(
        this.device.createBuffer({
          label: "UniformArena",
          size: this.chunkBytes,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        }),
      );
    }
    const buffer = this.chunks[this.chunk];
    const offset = this.offset;
    const view = ArrayBuffer.isView(data)
      ? data
      : new Uint8Array(data as ArrayBuffer);
    this.device.queue.writeBuffer(
      buffer,
      offset,
      view.buffer,
      view.byteOffset,
      bytes,
    );
    this.offset = Math.ceil((offset + bytes) / align) * align;
    return { buffer, offset, size: bytes };
  }

  /** After a submit: the writes are queued ahead of it, so space is free. */
  reset() {
    this.chunk = 0;
    this.offset = 0;
  }

  destroy() {
    for (const b of this.chunks) b.destroy();
    this.chunks = [];
    this.reset();
  }
}

export class KernelRegistry {
  readonly uniforms: UniformArena;
  private shaderModules = new Map<KernelModule, GPUShaderModule>();
  private kernels = new Map<string, Kernel>();

  constructor(readonly device: GPUDevice) {
    this.uniforms = new UniformArena(device);
  }

  /** The compute entry `entryName` of `module`, built on first use. */
  get(module: KernelModule, entryName: string): Kernel {
    const key = `${module.name}:${entryName}`;
    const cached = this.kernels.get(key);
    if (cached) return cached;

    const entry = module.reflection.entries.find((e) => e.name === entryName);
    if (!entry) {
      const names = module.reflection.entries.map((e) => e.name).join(", ");
      throw new Error(`${module.name}: no entry '${entryName}' (has ${names})`);
    }
    if (entry.stage !== "compute") {
      throw new Error(`${key} is a ${entry.stage} entry, not compute`);
    }

    const layouts = bindGroupLayouts(
      this.device,
      key,
      usedBindings(module, entry),
      () => GPUShaderStage.COMPUTE,
    );
    const pipeline = this.device.createComputePipeline({
      label: key,
      layout: this.device.createPipelineLayout({ bindGroupLayouts: layouts }),
      compute: { module: this.shaderModule(module), entryPoint: entry.name },
    });
    const kernel = new Kernel(this, module, entry, pipeline, layouts);
    this.kernels.set(key, kernel);
    return kernel;
  }

  shaderModule(module: KernelModule): GPUShaderModule {
    let sm = this.shaderModules.get(module);
    if (!sm) {
      sm = this.device.createShaderModule({
        label: module.name,
        code: module.wgsl,
      });
      this.shaderModules.set(module, sm);
    }
    return sm;
  }

  /** Submits and frees the uniform arena for the next batch of dispatches. */
  submit(...commands: GPUCommandBuffer[]) {
    this.device.queue.submit(commands);
    this.uniforms.reset();
  }

  destroy() {
    this.uniforms.destroy();
    this.kernels.clear();
    this.shaderModules.clear();
  }
}
