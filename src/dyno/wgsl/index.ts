// WGSL backend for dyno graphs, for the WebGPU kernels (src/webgpu/dyno).

export {
  registerWgsl,
  registerWgslStruct,
  UNIFORM_BLOCK,
  WgslBackend,
  type WgslEmitter,
  type WgslTextureBinding,
  type WgslUniformField,
  wgslStructTexture,
} from "./backend";
export { convert } from "./ops";
export { registerWgslGlobal, WGSL_PRELUDE } from "./prelude";
export { flattenValue, type WgslFunction, WgslDynoProgram } from "./program";
export { glslExprToWgsl, typeShape, wgslType } from "./types";
