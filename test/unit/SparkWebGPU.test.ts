import * as THREE from "three";
import { expect, test, vi } from "vitest";
import type { SparkRenderer } from "../../src/SparkRenderer";
import { SparkWebGPU } from "../../src/webgpu/SparkWebGPU";

// renderEnvMap on WebGPU prefilters with the PMREMGenerator given (three/
// webgpu's), reused while it is the same class, as WebGL reuses its own.
test("prefilterEnvMap uses the PMREMGenerator passed, once per class", () => {
  const renderer = {};
  const webgpu = new SparkWebGPU({} as SparkRenderer, renderer as never);
  const made: unknown[] = [];
  class Generator {
    constructor(r: unknown) {
      made.push(r);
    }
    fromCubemap = vi.fn(() => ({ texture: new THREE.Texture() }));
    dispose = vi.fn();
  }
  const cube = new THREE.CubeTexture();
  const a = webgpu.prefilterEnvMap(cube, Generator);
  const b = webgpu.prefilterEnvMap(cube, Generator);
  expect(made).toEqual([renderer]);
  expect(a).not.toBe(cube);
  expect(b).not.toBe(a);
  webgpu.dispose();
});

// Without three/webgpu's class (Spark's "three" being three's WebGL build),
// the cube map is returned for three/webgpu's materials to prefilter.
test("prefilterEnvMap without a WebGPU PMREMGenerator returns the cube", () => {
  const webgpu = new SparkWebGPU({} as SparkRenderer, {} as never);
  const cube = new THREE.CubeTexture();
  const version = cube.pmremVersion;
  expect(webgpu.prefilterEnvMap(cube)).toBe(cube);
  expect(cube.pmremVersion).toBe(version + 1);
});
