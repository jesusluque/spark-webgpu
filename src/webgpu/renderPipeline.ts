// Render pipelines from a module with a vertex and a fragment entry: the
// bind-group layout is the union of the bindings either entry uses, visible
// to the stages that use them.

import type { BindingReflection, KernelModule } from "./KernelModule";
import { usedBindings } from "./KernelRegistry";

export interface RenderPipelineOptions {
  vertex: string;
  fragment: string;
  targets: (GPUColorTargetState | null)[];
  primitive?: GPUPrimitiveState;
  depthStencil?: GPUDepthStencilState;
  multisample?: GPUMultisampleState;
  label?: string;
}

export interface ReflectedRenderPipeline {
  pipeline: GPURenderPipeline;
  layouts: GPUBindGroupLayout[];
  bindings: BindingReflection[];
}

export function createReflectedRenderPipeline(
  device: GPUDevice,
  module: KernelModule,
  options: RenderPipelineOptions,
): ReflectedRenderPipeline {
  const entry = (name: string) => {
    const e = module.reflection.entries.find((x) => x.name === name);
    if (!e) throw new Error(`${module.name}: no entry '${name}'`);
    return e;
  };
  const vs = entry(options.vertex);
  const fs = entry(options.fragment);
  const visibility = new Map<string, number>();
  for (const b of usedBindings(module, vs)) {
    visibility.set(
      b.name,
      (visibility.get(b.name) ?? 0) | GPUShaderStage.VERTEX,
    );
  }
  for (const b of usedBindings(module, fs)) {
    visibility.set(
      b.name,
      (visibility.get(b.name) ?? 0) | GPUShaderStage.FRAGMENT,
    );
  }
  const bindings = module.reflection.bindings
    .filter((b) => visibility.has(b.name))
    .sort((a, b) => a.group - b.group || a.binding - b.binding);

  const groupCount = bindings.reduce((n, b) => Math.max(n, b.group + 1), 0);
  const layouts: GPUBindGroupLayout[] = [];
  for (let g = 0; g < groupCount; g += 1) {
    layouts.push(
      device.createBindGroupLayout({
        label: `${module.name}@${g}`,
        entries: bindings
          .filter((b) => b.group === g)
          .map((b) => {
            if (b.kind === "unsupported" || b.kind === "external") {
              throw new Error(
                `${module.name}: unsupported binding '${b.name}'`,
              );
            }
            // Vertex shaders may not write storage buffers.
            return {
              binding: b.binding,
              visibility: visibility.get(b.name) as number,
              buffer: { type: b.kind },
            };
          }),
      }),
    );
  }
  const shaderModule = device.createShaderModule({
    label: module.name,
    code: module.wgsl,
  });
  const pipeline = device.createRenderPipeline({
    label: options.label ?? module.name,
    layout: device.createPipelineLayout({ bindGroupLayouts: layouts }),
    vertex: { module: shaderModule, entryPoint: vs.name },
    fragment: {
      module: shaderModule,
      entryPoint: fs.name,
      targets: options.targets,
    },
    primitive: options.primitive ?? { topology: "triangle-strip" },
    depthStencil: options.depthStencil,
    multisample: options.multisample,
  });
  return { pipeline, layouts, bindings };
}

/** Bind groups for a reflected pipeline, buffers by binding name. */
export function createBindGroups(
  device: GPUDevice,
  rp: ReflectedRenderPipeline,
  buffers: Record<string, GPUBuffer | GPUBufferBinding>,
): GPUBindGroup[] {
  return rp.layouts.map((layout, group) =>
    device.createBindGroup({
      layout,
      entries: rp.bindings
        .filter((b) => b.group === group)
        .map((b) => {
          const r = buffers[b.name];
          if (!r) throw new Error(`missing buffer '${b.name}'`);
          return {
            binding: b.binding,
            resource:
              "size" in r && !("buffer" in r)
                ? { buffer: r as GPUBuffer }
                : (r as GPUBufferBinding),
          };
        }),
    }),
  );
}
