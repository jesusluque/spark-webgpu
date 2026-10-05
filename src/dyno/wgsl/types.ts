// WGSL names and layouts for dyno's GLSL type names.

import type { DynoType } from "../types";
import { typeLiteral } from "../types";

export type WgslScalar = "f32" | "i32" | "u32" | "bool";

export interface TypeShape {
  scalar: WgslScalar;
  /** Components of a vector, rows of a matrix; 1 for a scalar. */
  rows: number;
  /** Columns of a matrix; 1 otherwise. */
  cols: number;
}

const SCALARS: Record<string, WgslScalar> = {
  float: "f32",
  int: "i32",
  uint: "u32",
  bool: "bool",
};
const VECTOR_SCALARS: Record<string, WgslScalar> = {
  "": "f32",
  i: "i32",
  u: "u32",
  b: "bool",
};

/** Shape of a scalar, vector or matrix type, else null. */
export function typeShape(type: DynoType): TypeShape | null {
  const t = typeLiteral(type);
  if (t in SCALARS) return { scalar: SCALARS[t], rows: 1, cols: 1 };
  const vec = /^([biu]?)vec([234])$/.exec(t);
  if (vec) {
    return { scalar: VECTOR_SCALARS[vec[1]], rows: Number(vec[2]), cols: 1 };
  }
  const mat = /^mat([234])(?:x([234]))?$/.exec(t);
  if (mat) {
    const cols = Number(mat[1]);
    return { scalar: "f32", rows: Number(mat[2] ?? cols), cols };
  }
  return null;
}

export function isMatrix(shape: TypeShape | null): boolean {
  return shape != null && shape.cols > 1;
}

export function isVector(shape: TypeShape | null): boolean {
  return shape != null && shape.cols === 1 && shape.rows > 1;
}

export function isScalar(shape: TypeShape | null): boolean {
  return shape != null && shape.cols === 1 && shape.rows === 1;
}

const SUFFIX: Record<WgslScalar, string> = {
  f32: "f",
  i32: "i",
  u32: "u",
  bool: "",
};

export function shapeType({ scalar, rows, cols }: TypeShape): string {
  if (cols > 1) return `mat${cols}x${rows}f`;
  if (rows === 1) return scalar;
  return scalar === "bool" ? `vec${rows}<bool>` : `vec${rows}${SUFFIX[scalar]}`;
}

const TEXTURES: Record<string, string> = {
  sampler2D: "texture_2d<f32>",
  usampler2D: "texture_2d<u32>",
  isampler2D: "texture_2d<i32>",
  sampler2DArray: "texture_2d_array<f32>",
  usampler2DArray: "texture_2d_array<u32>",
  isampler2DArray: "texture_2d_array<i32>",
  sampler3D: "texture_3d<f32>",
  usampler3D: "texture_3d<u32>",
  isampler3D: "texture_3d<i32>",
  samplerCube: "texture_cube<f32>",
  usamplerCube: "texture_cube<u32>",
  isamplerCube: "texture_cube<i32>",
  sampler2DShadow: "texture_depth_2d",
  sampler2DArrayShadow: "texture_depth_2d_array",
  samplerCubeShadow: "texture_depth_cube",
};

/** The texture type a GLSL sampler type becomes, else undefined. */
export function textureType(type: DynoType): string | undefined {
  return TEXTURES[typeLiteral(type)];
}

/** WGSL spelling of a dyno type; struct types keep their name. */
export function wgslType(type: DynoType): string {
  const shape = typeShape(type);
  if (shape) return shapeType(shape);
  return textureType(type) ?? typeLiteral(type);
}

/** `expr` of type `from` widened to vector type `to` when it is a scalar. */
export function splat(expr: string, from: DynoType, to: DynoType): string {
  const f = typeShape(from);
  const t = typeShape(to);
  if (isScalar(f) && t && !isScalar(t)) return `${shapeType(t)}(${expr})`;
  return expr;
}

/** Zero of a scalar, vector or matrix type. */
export function zero(type: DynoType): string {
  const shape = typeShape(type);
  if (!shape) throw new Error(`No WGSL zero for ${typeLiteral(type)}`);
  return `${shapeType(shape)}()`;
}

// GLSL constructor names in an expression -> WGSL, for literals and the
// statements of ops whose GLSL is otherwise valid WGSL.
const CONSTRUCTOR_RE =
  /\b(bvec[234]|ivec[234]|uvec[234]|mat[234](?:x[234])?|float|int|uint)\s*\(/g;

export function glslExprToWgsl(expr: string): string {
  return expr
    .replace(
      CONSTRUCTOR_RE,
      (_, name: string) => `${wgslType(name as DynoType)}(`,
    )
    .replace(/\bINFINITY\b/g, "dyno_inf()");
}

/** Size and alignment in a WGSL uniform block (std140-like). */
export function uniformLayout(type: DynoType): { size: number; align: number } {
  const shape = typeShape(type);
  if (!shape) {
    throw new Error(
      `Dyno uniform type ${typeLiteral(type)} has no WGSL layout`,
    );
  }
  const { rows, cols } = shape;
  const vecAlign = rows === 1 ? 4 : rows === 2 ? 8 : 16;
  const vecSize = 4 * rows;
  if (cols === 1) return { size: vecSize, align: vecAlign };
  return { size: cols * vecAlign, align: vecAlign };
}
