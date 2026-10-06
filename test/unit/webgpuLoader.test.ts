import { expect, test } from "vitest";
import { SparkRenderer } from "../../src/SparkRenderer";
import { loadWebGPU } from "../../src/webgpuLoader";

// The WebGPU backend is a chunk of its own (src/webgpuLoader.ts): a
// SparkRenderer given a WebGPURenderer before it loads gets it on webgpuReady.
test("SparkRenderer loads the WebGPU backend on demand", async () => {
  const renderer = { isWebGPURenderer: true, info: { frame: 0 } };
  const spark = new SparkRenderer({ renderer: renderer as never });
  const disposed = new SparkRenderer({ renderer: renderer as never });
  disposed.dispose();
  expect(spark.isWebGPU).toBe(true);
  expect(spark.webgpu).toBeUndefined();
  await Promise.all([spark.webgpuReady, disposed.webgpuReady]);
  const { SparkWebGPU } = await loadWebGPU();
  expect(spark.webgpu).toBeInstanceOf(SparkWebGPU);
  expect(disposed.webgpu).toBeUndefined();

  // Once loaded, from the constructor on.
  const next = new SparkRenderer({ renderer: renderer as never });
  expect(next.webgpu).toBeInstanceOf(SparkWebGPU);
});
