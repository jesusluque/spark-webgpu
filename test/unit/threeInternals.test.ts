import * as THREE from "three";
import { WebGPURenderer } from "three/webgpu";
import { expect, test } from "vitest";
import {
  THREE_REVISIONS,
  ThreeInternalsError,
  checkThreeRevision,
  openPass,
} from "../../src/webgpu/threeInternals";

test("recognises the installed three", () => {
  expect(THREE_REVISIONS).toContain(THREE.REVISION);
  expect(() => checkThreeRevision()).not.toThrow();
  expect(() => checkThreeRevision("999")).toThrow(ThreeInternalsError);
});

// The internals threeInternals.ts reads exist, with the shape it expects,
// on the installed three's WebGPURenderer and WebGPUBackend.
test("matches the installed WebGPURenderer's internals", () => {
  // A canvas stand-in: unit tests have no DOM, and nothing here draws.
  const canvas = { style: {}, addEventListener() {}, setAttribute() {} };
  const renderer = new WebGPURenderer({
    canvas: canvas as unknown as HTMLCanvasElement,
  });
  const backend = renderer.backend as unknown as Record<string, unknown>;
  expect(renderer).toHaveProperty("_currentRenderContext", null);
  expect(renderer).toHaveProperty("_frameBufferTarget", null);
  for (const key of ["get", "updateViewport", "generateMipmaps"]) {
    expect(typeof backend[key]).toBe("function");
  }
  const utils = backend.utils as Record<string, unknown>;
  for (const key of [
    "getCurrentColorFormat",
    "getCurrentDepthStencilFormat",
    "getSampleCountRenderContext",
    "getCurrentColorSpace",
  ]) {
    expect(typeof utils[key]).toBe("function");
  }
  const pipelineUtils = backend.pipelineUtils as Record<string, unknown>;
  expect(pipelineUtils._activePipelines).toBeInstanceOf(WeakMap);
});

test("fails with a clear error when an internal is gone", () => {
  const renderer = {
    backend: {
      isWebGPUBackend: true,
      get: () => undefined,
      device: {},
      context: {},
    },
    _currentRenderContext: null,
    _frameBufferTarget: null,
  };
  expect(() => openPass(renderer as never)).toThrow(
    /renderer\.backend\.utils is undefined.*threeInternals\.ts/,
  );
});
