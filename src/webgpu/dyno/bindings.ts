// The bind group of a compiled dyno program (src/dyno/wgsl): its uniform
// block at binding 0, then each sampler uniform's texture and sampler.

import type * as THREE from "three";
import { UNIFORM_BLOCK, type WgslDynoProgram } from "../../dyno/wgsl";
import type { BindingReflection } from "../KernelModule";
import { type TextureCache, isFilterable } from "./textures";

export interface TextureLayout {
  sampleType: GPUTextureSampleType;
  viewDimension: GPUTextureViewDimension;
}

/** Texture layouts by texture binding name. */
export type TextureLayouts = Record<string, TextureLayout>;

/** The layout a GLSL sampler type implies; float textures unfilterable. */
export function defaultTextureLayout(type: unknown): TextureLayout {
  const t = String(type);
  const sampleType: GPUTextureSampleType = t.startsWith("u")
    ? "uint"
    : t.startsWith("i")
      ? "sint"
      : t.includes("Shadow")
        ? "depth"
        : "unfilterable-float";
  const viewDimension: GPUTextureViewDimension = t.includes("2DArray")
    ? "2d-array"
    : t.includes("3D")
      ? "3d"
      : t.includes("Cube")
        ? "cube"
        : "2d";
  return { sampleType, viewDimension };
}

/**
 * Layouts for the program's textures from the textures its uniforms hold
 * now: sampled float textures filter unless their data is 32-bit float.
 */
export function textureLayouts(program: WgslDynoProgram): TextureLayouts {
  const layouts: TextureLayouts = {};
  for (const t of program.backend.textures) {
    const layout = defaultTextureLayout(t.type);
    if (
      t.samplerBinding != null &&
      layout.sampleType === "unfilterable-float" &&
      isFilterable(t.uniform.value as THREE.Texture)
    ) {
      layout.sampleType = "float";
    }
    layouts[t.name] = layout;
  }
  return layouts;
}

/** Reflection entries ("external" kind) for the program's bind group. */
export function dynoBindingReflections(
  program: WgslDynoProgram,
  layouts: TextureLayouts,
): BindingReflection[] {
  const { group } = program.backend;
  const extra: BindingReflection[] = [];
  if (program.uniformBytes > 0) {
    extra.push({
      name: UNIFORM_BLOCK,
      group,
      binding: 0,
      kind: "external",
      bytes: program.uniformBytes,
      layout: { buffer: { type: "uniform" } },
    });
  }
  for (const t of program.backend.textures) {
    const layout = layouts[t.name] ?? defaultTextureLayout(t.type);
    extra.push({
      name: t.name,
      group,
      binding: t.binding,
      kind: "external",
      layout: { texture: layout },
    });
    if (t.samplerBinding != null) {
      extra.push({
        name: `${t.name}_sampler`,
        group,
        binding: t.samplerBinding,
        kind: "external",
        layout: {
          sampler: {
            type: layout.sampleType === "float" ? "filtering" : "non-filtering",
          },
        },
      });
    }
  }
  return extra;
}

/**
 * The program's resources for one dispatch, from its uniforms' current
 * values (call program.update() first): the uniform block via `pushUniforms`
 * and the textures through `textures`.
 */
export function dynoResources(
  program: WgslDynoProgram,
  layouts: TextureLayouts,
  textures: TextureCache,
  pushUniforms: (data: ArrayBuffer) => GPUBindingResource,
): Record<string, GPUBindingResource> {
  const resources: Record<string, GPUBindingResource> = {};
  if (program.uniformBytes > 0) {
    resources[UNIFORM_BLOCK] = pushUniforms(program.packUniforms());
  }
  for (const t of program.backend.textures) {
    const texture = t.uniform.value as THREE.Texture;
    const layout = layouts[t.name] ?? defaultTextureLayout(t.type);
    resources[t.name] = textures.view(texture, layout.viewDimension);
    if (t.samplerBinding != null) {
      resources[`${t.name}_sampler`] = textures.sampler(
        texture,
        layout.sampleType === "float",
      );
    }
  }
  return resources;
}
