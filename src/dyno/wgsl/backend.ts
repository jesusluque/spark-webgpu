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
  uniform: { value: unknown };
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
    const shape = typeShape(type);
    if (!shape) {
      throw new Error(
        `Dyno uniform ${name}: type ${typeLiteral(type)} can't be a WGSL uniform`,
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

  /** Size of the uniform block, 0 when there are no uniforms. */
  get uniformBytes(): number {
    return Math.ceil(this.blockBytes / 16) * 16;
  }

  /** Module-scope declarations of the uniform block and texture bindings. */
  bindingDeclarations(): string[] {
    const decls: string[] = [];
    if (this.fields.length) {
      const members = this.fields.map((f) => {
        const shape = typeShape(f.type);
        const scalar =
          shape?.scalar === "bool"
            ? { ...shape, scalar: "u32" as const }
            : shape;
        const t = shapeType(scalar as NonNullable<typeof shape>);
        return f.count != null
          ? `    @align(16) ${f.name}: array<${t}, ${f.count}>,`
          : `    ${f.name}: ${t},`;
      });
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
