// WGSL backend for dyno graphs, for the WebGPU kernels (src/webgpu/dyno).

export {
  registerWgsl,
  UNIFORM_BLOCK,
  WgslBackend,
  type WgslEmitter,
  type WgslTextureBinding,
  type WgslUniformField,
} from "./backend";
export { convert } from "./ops";
export { registerWgslGlobal, WGSL_PRELUDE } from "./prelude";
export { flattenValue, type WgslFunction, WgslDynoProgram } from "./program";
export { glslExprToWgsl, typeShape, wgslType } from "./types";
