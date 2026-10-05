// The GLSL dyno emits, pinned by snapshot: the WGSL backend (src/dyno/wgsl)
// hooks into the same compiler and must leave the GLSL path byte-for-byte.

import * as THREE from "three";
import { describe, expect, it } from "vitest";
import { SplatTransformer } from "../../src/SplatGenerator";
import { DynoProgram, DynoProgramTemplate, dynoConst } from "../../src/dyno";
import type { Dyno, IOTypes } from "../../src/dyno/base";
import { snowBox } from "../../src/generators/snow";
import { staticBox } from "../../src/generators/static";
import { makeDepthColorModifier } from "../../src/modifiers/depthColor";
import { makeNormalColorModifier } from "../../src/modifiers/normalColor";
import { cases, graph, splatCases } from "../gpu/dynoCases";

const template = new DynoProgramTemplate(
  "{{ GLOBALS }}\nvoid main() {\n    {{ STATEMENTS }}\n}\n",
);

function glsl(g: Dyno<IOTypes, IOTypes>, inputs: Record<string, string>) {
  const outputs = Object.fromEntries(
    Object.keys(g.outTypes).map((k) => [k, `out_${k}`]),
  );
  return new DynoProgram({ graph: g, inputs, outputs, template }).shader;
}

describe("dyno GLSL", () => {
  for (const [name, type, f] of [...cases, ...splatCases]) {
    it(name, () => {
      expect(glsl(graph(type, f), { index: "index" })).toMatchSnapshot();
    });
  }

  it("modifiers", () => {
    const depth = makeDepthColorModifier(
      new SplatTransformer(),
      dynoConst("float", 0.1),
      dynoConst("float", 10),
      dynoConst("bool", true),
    );
    const normal = makeNormalColorModifier(new SplatTransformer());
    expect(glsl(depth, { gsplat: "gsplat" })).toMatchSnapshot();
    expect(glsl(normal, { gsplat: "gsplat" })).toMatchSnapshot();
  });

  it("generators", () => {
    const { snow } = snowBox({});
    const noise = staticBox({
      box: new THREE.Box3(
        new THREE.Vector3(-1, -1, -1),
        new THREE.Vector3(1, 1, 1),
      ),
      cells: new THREE.Vector3(4, 4, 4),
      dotScale: 0.01,
    });
    for (const g of [snow.generator, noise.generator]) {
      expect(glsl(g as never, { index: "index" })).toMatchSnapshot();
    }
  });
});
