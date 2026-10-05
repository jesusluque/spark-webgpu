// WgpuDyno views of Spark's dyno users, so their graphs run on WebGPU with
// the per-frame context SparkRenderer would otherwise update.

import type { GsplatModifier, SplatGenerator } from "../../SplatGenerator";
import type { SplatMesh } from "../../SplatMesh";
import { type Dyno, type IOTypes, dynoBlock } from "../../dyno/base";
import { Gsplat } from "../../dyno/splats";
import type { WgpuDyno, WgpuDynoFrame } from "./DynoKernels";

// A modifier per wrapped object, so the graph keeps its identity across
// frames and isn't recompiled.
const wrapped = new WeakMap<object, GsplatModifier>();
function asModifier(
  owner: object,
  modify: (g: Dyno<IOTypes, IOTypes>) => unknown,
): GsplatModifier {
  let m = wrapped.get(owner);
  if (!m) {
    m = dynoBlock({ gsplat: Gsplat }, { gsplat: Gsplat }, ({ gsplat }) => ({
      gsplat: modify(gsplat as never) as never,
    }));
    wrapped.set(owner, m);
  }
  return m;
}

/**
 * A SplatMesh's dyno pipeline on WebGPU: skinning and objectModifiers in
 * object space, SDF edits and worldModifiers in world space, as in
 * SplatMesh.constructGenerator, with its dyno context (time, transforms,
 * worldToView...) updated each frame. Pass it to WgpuSplatRenderer.add with
 * the mesh as the object. Modifier changes are picked up without
 * updateGenerator().
 */
export function splatMeshDyno(mesh: SplatMesh): WgpuDyno {
  return {
    get objectModifiers() {
      const mods = [...(mesh.objectModifiers ?? [])];
      const skinning = mesh.skinning;
      if (skinning) {
        mods.unshift(asModifier(skinning, (g) => skinning.modify(g as never)));
      }
      return mods;
    },
    get worldModifiers() {
      const mods = [...(mesh.worldModifiers ?? [])];
      const edits = mesh.rgbaDisplaceEdits;
      if (edits) {
        mods.unshift(asModifier(edits, (g) => edits.modify(g as never)));
      }
      return mods;
    },
    update(frame: WgpuDynoFrame) {
      updateSplatMeshContext(mesh, frame);
    },
  };
}

/** The dyno uniforms SplatMesh.update sets, for WebGPU frames. */
export function updateSplatMeshContext(mesh: SplatMesh, frame: WgpuDynoFrame) {
  const { context } = mesh;
  context.time.value = frame.time;
  context.deltaTime.value = frame.deltaTime;
  (mesh.constructor as typeof SplatMesh).dynoTime.value = frame.time;
  context.transform.update(mesh);
  const viewToWorld = frame.camera.matrixWorld;
  context.viewToWorld.updateFromMatrix(viewToWorld);
  context.worldToView.updateFromMatrix(viewToWorld.clone().invert());
  const worldToObject = mesh.matrixWorld.clone().invert();
  context.viewToObject.updateFromMatrix(worldToObject.multiply(viewToWorld));
  context.recolor.value.set(
    mesh.recolor.r,
    mesh.recolor.g,
    mesh.recolor.b,
    mesh.opacity,
  );
}

/**
 * A SplatGenerator (snow, static, user generators) on WebGPU: its generator
 * graph makes the splats, in world space as it applies its own transform, and
 * its frameUpdate runs each frame. Use WgpuSplatRenderer.addGenerator with
 * generator.numSplats.
 */
export function splatGeneratorDyno(generator: SplatGenerator): WgpuDyno {
  return {
    get generator() {
      return generator.generator;
    },
    worldSpace: true,
    update(frame: WgpuDynoFrame) {
      generator.updateMatrixWorld();
      generator.frameUpdate?.({
        renderer: null as never,
        object: generator,
        time: frame.time,
        deltaTime: frame.deltaTime,
        viewToWorld: frame.camera.matrixWorld,
        camera: frame.camera,
        globalEdits: [],
      });
    },
  };
}
