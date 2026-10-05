// Dyno graphs in the generate kernel: compiles a mesh's generator and
// modifiers to WGSL, patches them into the generate module at its hooks, and
// gives the dispatch its pipeline and dyno bindings. Pipelines are cached by
// the generated code, so meshes with the same graph structure share one;
// each mesh keeps its own compiled program for its uniform values.

import type * as THREE from "three";
import type { GsplatGenerator, GsplatModifier } from "../../SplatGenerator";
import { type Dyno, type IOTypes, dynoBlock } from "../../dyno/base";
import { Gsplat } from "../../dyno/splats";
import { WgslDynoProgram, type WgslFunction } from "../../dyno/wgsl";
import type { KernelModule } from "../KernelModule";
import type { Kernel, KernelRegistry } from "../KernelRegistry";
import { type TextureLayouts, dynoResources, textureLayouts } from "./bindings";
import { type DynoHookFunctions, patchKernel } from "./patchKernel";
import { TextureCache } from "./textures";

export interface WgpuDynoFrame {
  camera: THREE.Camera;
  object: THREE.Object3D;
  /** Seconds, from performance.now(). */
  time: number;
  deltaTime: number;
}

/**
 * Dyno graphs for one mesh, as SplatMesh / SplatGenerator use them. Arrays
 * may be edited in place: the graph is recompiled when its members change.
 */
export interface WgpuDyno {
  /** Index -> Gsplat, replacing the read from the mesh's source. */
  generator?: GsplatGenerator;
  /** Applied in object space after the read (SplatMesh.objectModifiers). */
  objectModifiers?: GsplatModifier[];
  /** Applied in world space after transform and recolor. */
  worldModifiers?: GsplatModifier[];
  /** The generator emits world-space splats itself: skip the transform. */
  worldSpace?: boolean;
  /** Called each frame before generating, e.g. to set uniform values. */
  update?: (frame: WgpuDynoFrame) => void;
}

export interface DynoDispatch {
  kernel: Kernel;
  bindings: Record<string, GPUBindingResource>;
}

interface Compiled {
  graphs: unknown[];
  program: WgslDynoProgram;
  module: KernelModule;
  layouts: TextureLayouts;
}

function chain(modifiers: GsplatModifier[]): Dyno<IOTypes, IOTypes> {
  return dynoBlock({ gsplat: Gsplat }, { gsplat: Gsplat }, ({ gsplat }) => {
    let g = gsplat;
    for (const modifier of modifiers) {
      g = modifier.apply({ gsplat: g }).gsplat;
    }
    return { gsplat: g };
  }) as unknown as Dyno<IOTypes, IOTypes>;
}

function graphsOf(dyno: WgpuDyno): unknown[] {
  return [
    dyno.generator,
    ...(dyno.objectModifiers ?? []),
    null,
    ...(dyno.worldModifiers ?? []),
  ];
}

function sameGraphs(a: unknown[], b: unknown[]) {
  return a.length === b.length && a.every((x, i) => x === b[i]);
}

export class DynoKernels {
  readonly textures: TextureCache;
  private modules = new Map<string, KernelModule>();
  private compiled = new WeakMap<object, Compiled>();

  constructor(
    readonly registry: KernelRegistry,
    readonly base: KernelModule,
    readonly entry: string,
  ) {
    this.textures = new TextureCache(registry.device);
  }

  /** Whether the mesh's dyno has anything to run. */
  static active(dyno?: WgpuDyno): dyno is WgpuDyno {
    return Boolean(
      dyno &&
        (dyno.generator ||
          dyno.objectModifiers?.length ||
          dyno.worldModifiers?.length),
    );
  }

  /** The kernel and dyno bindings for `owner`'s next dispatch. */
  prepare(owner: object, dyno: WgpuDyno): DynoDispatch {
    const { program, module, layouts } = this.compile(owner, dyno);
    program.update();
    const bindings = dynoResources(program, layouts, this.textures, (data) =>
      this.registry.uniforms.push(data),
    );
    return { kernel: this.registry.get(module, this.entry), bindings };
  }

  // A WGSL error only shows as an invalid pipeline: say where it is, which
  // matters most for hand-written dyno WGSL.
  private reportErrors(module: KernelModule) {
    const lines = module.wgsl.split("\n");
    this.registry
      .shaderModule(module)
      .getCompilationInfo()
      .then((info) => {
        const errors = info.messages
          .filter((m) => m.type === "error")
          .map(
            (m) =>
              `${m.lineNum}:${m.linePos} ${m.message}\n    ${lines[m.lineNum - 1]?.trim()}`,
          );
        if (errors.length) {
          console.error(
            `${module.name}: dyno WGSL errors\n${errors.join("\n")}`,
          );
        }
      });
  }

  private compile(owner: object, dyno: WgpuDyno): Compiled {
    const graphs = graphsOf(dyno);
    const cached = this.compiled.get(owner);
    if (cached && sameGraphs(cached.graphs, graphs)) return cached;

    const functions: WgslFunction[] = [];
    const hooks: DynoHookFunctions = {};
    if (dyno.generator) {
      functions.push({
        name: "dyno_source",
        graph: dyno.generator as unknown as Dyno<IOTypes, IOTypes>,
        inputs: { index: "int" },
        output: "gsplat",
      });
      hooks.dynoSource = "dyno_source";
    }
    if (dyno.objectModifiers?.length) {
      functions.push({
        name: "dyno_object",
        graph: chain(dyno.objectModifiers),
        inputs: { gsplat: Gsplat },
        output: "gsplat",
      });
      hooks.dynoObjectModifier = "dyno_object";
    }
    if (dyno.worldModifiers?.length) {
      functions.push({
        name: "dyno_world",
        graph: chain(dyno.worldModifiers),
        inputs: { gsplat: Gsplat },
        output: "gsplat",
      });
      hooks.dynoWorldModifier = "dyno_world";
    }
    const program = new WgslDynoProgram({ functions, group: 1 });

    const layouts = textureLayouts(program);
    const key = `${program.code}\n${JSON.stringify(layouts)}`;
    let module = this.modules.get(key);
    if (!module) {
      module = patchKernel(
        this.base,
        program,
        hooks,
        `${this.base.name}+dyno${this.modules.size}`,
        layouts,
      );
      this.modules.set(key, module);
      this.reportErrors(module);
    }
    const compiled = { graphs, program, module, layouts };
    this.compiled.set(owner, compiled);
    return compiled;
  }
}
