// Packs a kernel's uniform block by field name, at the offsets and with the
// scalar types its reflection gives, so host code never mirrors a Slang
// struct by hand. Fields left out are zero; unknown names throw.

import type { BindingReflection, KernelModule } from "./KernelModule";

export type UniformValue = number | boolean | ArrayLike<number>;

export function uniformBlock(
  module: KernelModule,
  name = "params",
): BindingReflection {
  const b = module.reflection.bindings.find(
    (x) => x.name === name && x.kind === "uniform",
  );
  if (!b) throw new Error(`${module.name}: no uniform block '${name}'`);
  return b;
}

const fieldMaps = new WeakMap<BindingReflection, Map<string, unknown>>();

export class UniformWriter {
  readonly data: ArrayBuffer;
  private readonly view: DataView;
  private readonly fields: Map<string, BindingReflection["fields"] & object>;

  constructor(readonly block: BindingReflection) {
    this.data = new ArrayBuffer(block.bytes ?? 0);
    this.view = new DataView(this.data);
    // Writers are made per dispatch; the name lookup is made once a block.
    let fields = fieldMaps.get(block);
    if (!fields) {
      fields = new Map((block.fields ?? []).map((f) => [f.name, f] as never));
      fieldMaps.set(block, fields);
    }
    this.fields = fields as typeof this.fields;
  }

  static for(module: KernelModule, name = "params") {
    return new UniformWriter(uniformBlock(module, name));
  }

  set(name: string, value: UniformValue): this {
    const f = this.fields.get(name) as unknown as {
      offset: number;
      scalar?: string;
      components?: number;
    };
    if (!f) {
      throw new Error(
        `uniform block '${this.block.name}' has no field '${name}'`,
      );
    }
    if (!f.scalar) {
      throw new Error(
        `uniform field '${name}' is not a scalar, vector or matrix`,
      );
    }
    const values =
      typeof value === "number" || typeof value === "boolean"
        ? [Number(value)]
        : Array.from(value);
    const n = f.components ?? 1;
    if (values.length > n) {
      throw new Error(
        `uniform field '${name}' has ${n} components, got ${values.length}`,
      );
    }
    values.forEach((v, i) => {
      const at = f.offset + 4 * i;
      if (f.scalar === "float32") this.view.setFloat32(at, v, true);
      else if (f.scalar === "int32") this.view.setInt32(at, v, true);
      else this.view.setUint32(at, v >>> 0, true);
    });
    return this;
  }

  setAll(values: Record<string, UniformValue>): this {
    for (const [k, v] of Object.entries(values)) this.set(k, v);
    return this;
  }
}
