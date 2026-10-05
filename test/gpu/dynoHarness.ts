// Runs a dyno graph compiled to WGSL on the test device: one invocation that
// stores the graph's `value` output as raw 32-bit words.

import type { Dyno, IOTypes } from "../../src/dyno/base";
import type { DynoType } from "../../src/dyno/types";
import { WgslDynoProgram, typeShape } from "../../src/dyno/wgsl";
import {
  dynoBindingReflections,
  dynoResources,
  textureLayouts,
} from "../../src/webgpu/dyno/bindings";
import { TextureCache } from "../../src/webgpu/dyno/textures";
import { device, readBack, storage } from "./device";

let cache: TextureCache | null = null;
const textures = () => {
  cache ??= new TextureCache(device as GPUDevice);
  return cache;
};

export async function compileErrors(code: string): Promise<string[]> {
  const d = device as GPUDevice;
  const module = d.createShaderModule({ code });
  const info = await module.getCompilationInfo();
  return info.messages
    .filter((m) => m.type === "error")
    .map((m) => {
      const line = code.split("\n")[m.lineNum - 1] ?? "";
      return `${m.lineNum}:${m.linePos} ${m.message}\n    ${line.trim()}`;
    });
}

function storeWords(type: DynoType): { words: number; store: string } {
  const shape = typeShape(type);
  if (!shape) throw new Error("harness: value must be a scalar/vector/matrix");
  const { scalar, rows, cols } = shape;
  const toU32 = (e: string) =>
    scalar === "u32"
      ? e
      : scalar === "bool"
        ? `select(0u, 1u, ${e})`
        : `bitcast<u32>(${e})`;
  const lines: string[] = [];
  for (let c = 0; c < cols; c++) {
    for (let r = 0; r < rows; r++) {
      const e = cols > 1 ? `v[${c}][${r}]` : rows > 1 ? `v[${r}]` : "v";
      lines.push(`    words[${c * rows + r}] = ${toU32(e)};`);
    }
  }
  return { words: rows * cols, store: lines.join("\n") };
}

export function harnessCode(program: WgslDynoProgram, type: DynoType) {
  const { store } = storeWords(type);
  return `${program.code}

@group(0) @binding(0) var<storage, read_write> words: array<u32>;

@compute @workgroup_size(1)
fn main() {
    let v = dyno_test(0);
${store}
}
`;
}

/** The graph's `value` output, as numbers (bools 0/1). */
export async function evalDyno(
  graph: Dyno<IOTypes, IOTypes>,
): Promise<number[]> {
  const d = device as GPUDevice;
  const type = graph.outTypes.value;
  const program = new WgslDynoProgram({
    functions: [
      { name: "dyno_test", graph, inputs: { index: "int" }, output: "value" },
    ],
  });
  const code = harnessCode(program, type);
  const errors = await compileErrors(code);
  if (errors.length) {
    throw new Error(`WGSL errors:\n${errors.join("\n")}\n\n${code}`);
  }
  program.update();
  const { words } = storeWords(type);
  const out = storage(words * 4);
  const layouts = textureLayouts(program);
  const dynoEntries = dynoBindingReflections(program, layouts);
  const bgl0 = d.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" },
      },
    ],
  });
  const bgl1 = d.createBindGroupLayout({
    entries: dynoEntries.map((b) => ({
      ...b.layout,
      binding: b.binding,
      visibility: GPUShaderStage.COMPUTE,
    })),
  });
  const pipeline = d.createComputePipeline({
    layout: d.createPipelineLayout({ bindGroupLayouts: [bgl0, bgl1] }),
    compute: { module: d.createShaderModule({ code }), entryPoint: "main" },
  });
  const resources = dynoResources(program, layouts, textures(), (data) => {
    const ubo = d.createBuffer({
      size: data.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    d.queue.writeBuffer(ubo, 0, data);
    return { buffer: ubo };
  });
  const encoder = d.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(
    0,
    d.createBindGroup({
      layout: bgl0,
      entries: [{ binding: 0, resource: { buffer: out } }],
    }),
  );
  pass.setBindGroup(
    1,
    d.createBindGroup({
      layout: bgl1,
      entries: dynoEntries.map((b) => ({
        binding: b.binding,
        resource: resources[b.name],
      })),
    }),
  );
  pass.dispatchWorkgroups(1);
  pass.end();
  d.queue.submit([encoder.finish()]);
  const raw = new DataView(await readBack(out));
  const shape = typeShape(type);
  return Array.from({ length: words }, (_, i) => {
    if (shape?.scalar === "f32") return raw.getFloat32(4 * i, true);
    if (shape?.scalar === "i32") return raw.getInt32(4 * i, true);
    return raw.getUint32(4 * i, true);
  });
}
