// The WGSL code generation backend for dyno graphs.
//
// Dyno ops emit GLSL. Much of it is valid WGSL once GLSL constructor names are
// mapped (vec3(...) is fine, ivec3(...) becomes vec3i(...)), so by default an
// op's GLSL statements are used with that mapping. Ops whose GLSL isn't valid
// WGSL (out parameters, ?:, overloads, local declarations, builtins WGSL lacks
// or names differently) register a WGSL emitter (ops.ts), and user dynos give
// their WGSL in `wgsl: { globals, statements }` or branch on compile.target.
//
// Uniforms can't be loose globals in WGSL: scalars, vectors and matrices go
// into one uniform block and their names resolve to its fields; samplers
// become texture (and sampler) bindings. All in one bind group.

import {
  type Dyno,
  type DynoBackend,
  DynoBlock,
  type DynoGenerated,
  type GenerateContext,
  type IOTypes,
} from "../base";
import type { DynoType } from "../types";
import { typeLiteral } from "../types";
import { DynoConst, type DynoLiteral } from "../value";
import { wgslGlobals } from "./prelude";
import {
  glslExprToWgsl,
  shapeType,
  textureType,
  typeShape,
  uniformLayout,
  wgslType,
} from "./types";

export type WgslEmitter = (
  // biome-ignore lint/suspicious/noExplicitAny: emitters read op-specific fields
  dyno: any,
  context: GenerateContext<IOTypes, IOTypes>,
  backend: WgslBackend,
) => DynoGenerated;

// biome-ignore lint/complexity/noBannedTypes: keyed by op class
const emitters = new Map<Function, WgslEmitter>();

/** Gives a dyno class (and its subclasses) its WGSL code. */
export function registerWgsl(
  // biome-ignore lint/suspicious/noExplicitAny: any dyno class
  cls: abstract new (...args: any[]) => Dyno<any, any>,
  emitter: WgslEmitter,
) {
  emitters.set(cls, emitter);
}

function findEmitter(dyno: object): WgslEmitter | undefined {
  for (
    let proto = Object.getPrototypeOf(dyno);
    proto;
    proto = Object.getPrototypeOf(proto)
  ) {
    const emitter = emitters.get(proto.constructor);
    if (emitter) return emitter;
  }
  return undefined;
}

export interface WgslUniformField {
  name: string;
  type: DynoType;
  count?: number;
  offset: number;
  /** Bytes per array element (count set). */
  stride?: number;
  /** Struct uniforms: their non-texture fields, at offsets in the struct. */
  members?: { name: string; type: DynoType; offset: number }[];
  /** Struct uniforms: bytes the member takes in the block. */
  size?: number;
  uniform: { value: unknown };
}

// Struct types dyno uniforms may have (e.g. SplatEdit's SdfArray), by name.
const structs = new Map<string, Record<string, DynoType>>();

/**
 * Lets dyno uniforms of a struct type compile to WGSL. WGSL structs can't
 * hold textures, so its texture fields become bindings of their own, named
 * by wgslStructTexture; the other fields stay a struct in the uniform block,
 * with bools stored as u32.
 */
export function registerWgslStruct(
  type: { type: string },
  fields: Record<string, DynoType>,
) {
  structs.set(type.type, fields);
}

/** The texture binding for `field` of a struct uniform (an input's value). */
export function wgslStructTexture(uniform: string, field: string): string {
  const prefix = `${UNIFORM_BLOCK}.`;
  const name = uniform.startsWith(prefix)
    ? uniform.slice(prefix.length)
    : uniform;
  return `${name}_${field}`;
}

const roundUp = (n: number, k: number) => Math.ceil(n / k) * k;

function storedType(type: DynoType): string {
  const shape = typeShape(type);
  if (!shape) throw new Error(`No WGSL uniform type for ${typeLiteral(type)}`);
  return shapeType(
    shape.scalar === "bool" ? { ...shape, scalar: "u32" } : shape,
  );
}

export interface WgslTextureBinding {
  name: string;
  type: DynoType;
  binding: number;
  /** Binding of its sampler, for texture() on float textures. */
  samplerBinding?: number;
  uniform: { value: unknown };
}

export const UNIFORM_BLOCK = "dyno_uniforms";

export class WgslBackend implements DynoBackend {
  readonly target = "wgsl";
  readonly fields: WgslUniformField[] = [];
  readonly textures: WgslTextureBinding[] = [];
  private aliases = new Map<string, string>();
  private structDecls = new Map<string, string>();
  /** Globals that are already WGSL (from emitters and dyno.wgsl). */
  private ownGlobals = new Set<string>();
  private blockBytes = 0;
  private nextBinding = 1;

  /** Bind group of the uniform block (binding 0) and textures. */
  constructor(readonly group = 1) {}

  generate(
    dyno: Dyno<IOTypes, IOTypes>,
    context: GenerateContext<IOTypes, IOTypes>,
  ): DynoGenerated {
    if (dyno.wgsl) {
      const globals = dyno.wgsl.globals?.(context) ?? [];
      for (const g of globals) this.ownGlobals.add(g);
      return { globals, statements: dyno.wgsl.statements?.(context) };
    }
    const emitter = findEmitter(dyno);
    if (emitter) {
      const result = emitter(dyno, context, this);
      for (const g of result.globals ?? []) {
        if (!wgslGlobals.has(g)) this.ownGlobals.add(g);
      }
      return result;
    }
    const result = dyno.generate(context);
    if (dyno instanceof DynoBlock) return result;
    return {
      ...result,
      statements: result.statements?.map(glslExprToWgsl),
    };
  }

  declare(name: string, type: DynoType): string {
    return `var ${name}: ${wgslType(type)}`;
  }

  literal(value: DynoLiteral<DynoType>): string {
    const literal = glslExprToWgsl(value.getLiteral());
    // An abstract int would become f32 in float contexts; keep it i32.
    if (value instanceof DynoConst && value.type === "int") {
      return `i32(${literal})`;
    }
    return literal;
  }

  resolve(name: string): string {
    return this.aliases.get(name) ?? name;
  }

  /** Marks a global as WGSL, for emitters that build them on the fly. */
  addGlobal(global: string): string {
    this.ownGlobals.add(global);
    return global;
  }

  /** A uniform named `name`; returns nothing, as references resolve to it. */
  addUniform(
    name: string,
    type: DynoType,
    uniform: { value: unknown },
    count?: number,
  ) {
    if (textureType(type)) {
      const binding = this.nextBinding++;
      const sampled = typeLiteral(type).startsWith("sampler");
      const samplerBinding = sampled ? this.nextBinding++ : undefined;
      this.textures.push({ name, type, binding, samplerBinding, uniform });
      return;
    }
    const struct = structs.get(typeLiteral(type));
    if (struct) {
      this.addStructUniform(name, type, uniform, struct);
      return;
    }
    const shape = typeShape(type);
    if (!shape) {
      throw new Error(
        `Dyno uniform ${name}: type ${typeLiteral(type)} can't be a WGSL uniform (registerWgslStruct for structs)`,
      );
    }
    let { size, align } = uniformLayout(type);
    let stride: number | undefined;
    if (count != null) {
      stride = Math.ceil(size / align) * align;
      if (stride % 16 !== 0 || shape.scalar === "bool") {
        throw new Error(
          `Dyno uniform ${name}: WGSL uniform arrays need 16-byte elements, ${typeLiteral(type)}[${count}] has ${stride}; use a vec4 type`,
        );
      }
      size = stride * count;
      align = 16;
    }
    const offset = Math.ceil(this.blockBytes / align) * align;
    this.blockBytes = offset + size;
    this.fields.push({ name, type, count, offset, stride, uniform });

    // WGSL uniforms can't hold bool: stored as u32, compared back.
    const field = `${UNIFORM_BLOCK}.${name}`;
    if (shape.scalar === "bool") {
      const zero = shape.rows === 1 ? "0u" : `vec${shape.rows}u()`;
      this.aliases.set(name, `(${field} != ${zero})`);
    } else {
      this.aliases.set(name, field);
    }
  }

  private addStructUniform(
    name: string,
    type: DynoType,
    uniform: { value: unknown },
    struct: Record<string, DynoType>,
  ) {
    const members: { name: string; type: DynoType; offset: number }[] = [];
    let size = 0;
    let align = 4;
    for (const [field, fieldType] of Object.entries(struct)) {
      if (textureType(fieldType)) {
        const value = {
          get value() {
            return (uniform.value as Record<string, unknown>)?.[field];
          },
        };
        this.addUniform(`${name}_${field}`, fieldType, value);
        continue;
      }
      const layout = uniformLayout(fieldType);
      const offset = roundUp(size, layout.align);
      members.push({ name: field, type: fieldType, offset });
      size = offset + layout.size;
      align = Math.max(align, layout.align);
    }
    if (!members.length) return;
    const structName = `DynoU_${typeLiteral(type)}`;
    this.structDecls.set(
      structName,
      `struct ${structName} {\n${members.map((m) => `    ${m.name}: ${storedType(m.type)},`).join("\n")}\n}`,
    );
    // In a uniform block a struct member aligns to 16 and pads to 16.
    const memberSize = roundUp(roundUp(size, align), 16);
    const offset = roundUp(this.blockBytes, 16);
    this.blockBytes = offset + memberSize;
    this.fields.push({
      name,
      type,
      offset,
      members,
      size: memberSize,
      uniform,
    });
    this.aliases.set(name, `${UNIFORM_BLOCK}.${name}`);
  }

  /** Size of the uniform block, 0 when there are no uniforms. */
  get uniformBytes(): number {
    return Math.ceil(this.blockBytes / 16) * 16;
  }

  /** Module-scope declarations of the uniform block and texture bindings. */
  bindingDeclarations(): string[] {
    const decls: string[] = [];
    if (this.fields.length) {
      const members = this.fields.map((f) => {
        if (f.members) {
          return `    @align(16) @size(${f.size}) ${f.name}: DynoU_${typeLiteral(f.type)},`;
        }
        const t = storedType(f.type);
        return f.count != null
          ? `    @align(16) ${f.name}: array<${t}, ${f.count}>,`
          : `    ${f.name}: ${t},`;
      });
      decls.push(...this.structDecls.values());
      decls.push(
        `struct DynoUniforms {\n${members.join("\n")}\n}`,
        `@group(${this.group}) @binding(0) var<uniform> ${UNIFORM_BLOCK}: DynoUniforms;`,
      );
    }
    for (const t of this.textures) {
      decls.push(
        `@group(${this.group}) @binding(${t.binding}) var ${t.name}: ${textureType(t.type)};`,
      );
      if (t.samplerBinding != null) {
        decls.push(
          `@group(${this.group}) @binding(${t.samplerBinding}) var ${t.name}_sampler: sampler;`,
        );
      }
    }
    return decls;
  }

  /** `globals` in WGSL: shared GLSL constants mapped, unknown GLSL refused. */
  wgslGlobals(globals: Iterable<string>): string[] {
    const result: string[] = [];
    for (const g of globals) {
      if (this.ownGlobals.has(g)) {
        result.push(g);
        continue;
      }
      const mapped = wgslGlobals.get(g);
      if (mapped === undefined) {
        const first = g.trim().split("\n")[0];
        throw new Error(
          `Dyno global has no WGSL version (give the dyno a wgsl option or registerWgslGlobal): ${first}`,
        );
      }
      if (mapped) result.push(mapped);
    }
    return result;
  }
}
