import * as THREE from "three";

import { DynoPackedSplats, type PackedSplats } from "./PackedSplats";
import { Readback, type Rgba8Readback } from "./Readback";
import { SPLAT_TEX_WIDTH } from "./defines";
import {
  Dyno,
  type DynoBlock,
  DynoInt,
  DynoUniform,
  type DynoVal,
  add,
  dynoBlock,
  readPackedSplatRange,
  registerWgslGlobal,
  registerWgslStruct,
  splitGsplat,
  unindent,
  unindentLines,
  wgslStructTexture,
} from "./dyno";
import { getTextureSize } from "./utils";
import { gpuTextures } from "./webgpu/dyno/textures";

// The renderers RgbaArray works with: on three's WebGPURenderer the values
// come from the CPU (fromPackedSplats) or from SparkRenderer.getRgba.
type AnyRenderer = THREE.WebGLRenderer | { isWebGPURenderer?: boolean };
const isWebGPU = (renderer: AnyRenderer) =>
  !!(renderer as { isWebGPURenderer?: boolean }).isWebGPURenderer;

// An RgbaArray is a collection of ordered RGBA8 values, which can be used as a dyno
// data source, for example for recoloring Gsplats via SplatMesh.splatRgba.
// It can be instantiated from a Uint8Array of RGBA8 values, or it can be
// generated using a Rgba8Readback dyno program.

export type RgbaArrayOptions = {
  // Reserve space for at least this many RGBA values.
  capacity?: number;
  // Use the provided array of RGBA8 values as the source.
  array?: Uint8Array;
  // The number of actual RGBA8 values in the array.
  count?: number;
};

export class RgbaArray {
  capacity = 0;
  count = 0;
  array: Uint8Array | null = null;

  readback: Readback | null = null;
  source: THREE.DataArrayTexture | null = null;
  // Set to true if source array is updated to have it upload to GPU
  needsUpdate = true;
  // On WebGPU, values a kernel wrote (SparkRenderer.getRgba): `placeholder`
  // stands for `texture` in dyno uniforms (see gpuTextures).
  private gpu: {
    device: GPUDevice;
    texture: GPUTexture;
    placeholder: THREE.DataArrayTexture;
  } | null = null;

  // Use this as a TRgbaArray in a dyno graph
  dyno: DynoUniform<typeof TRgbaArray, "rgbaArray">;

  constructor(options: RgbaArrayOptions = {}) {
    this.dyno = new DynoUniform({
      key: "rgbaArray",
      type: TRgbaArray,
      globals: () => [defineRgbaArray],
      value: {
        texture: RgbaArray.getEmpty(),
        count: 0,
      },
      update: (value) => {
        value.texture = this.getTexture();
        value.count = this.count;
        return value;
      },
    });

    if (options.array) {
      // Initialize with given array
      this.array = options.array;
      const splatCount = Math.floor(this.array.length / 4);
      this.capacity = Math.ceil(splatCount / SPLAT_TEX_WIDTH) * SPLAT_TEX_WIDTH;
      if (this.capacity > splatCount) {
        this.array = new Uint8Array(this.capacity * 4);
        this.array.set(options.array);
      }
      this.count = Math.min(
        splatCount,
        options.count ?? Number.POSITIVE_INFINITY,
      );
    } else {
      this.capacity = options.capacity ?? 0;
      this.count = 0;
    }
  }

  // Free up resources
  dispose() {
    if (this.gpu) {
      this.gpu.texture.destroy();
      this.gpu.placeholder.dispose();
      this.gpu = null;
    }
    if (this.readback) {
      this.readback.dispose();
      this.readback = null;
    }
    if (this.source) {
      this.source.dispose();
      this.source = null;
    }
  }

  // Ensure that our array is large enough to hold capacity RGBA8 values.
  ensureCapacity(capacity: number): Uint8Array {
    if (!this.array || capacity > (this.array?.length ?? 0) / 4) {
      this.capacity = getTextureSize(capacity).maxSplats;
      const newArray = new Uint8Array(this.capacity * 4);
      if (this.array) {
        // Copy over existing data
        newArray.set(this.array);
      }
      this.array = newArray;
    }
    return this.array;
  }

  // Get the THREE.DataArrayTexture from either the readback or the source.
  getTexture(): THREE.DataArrayTexture {
    if (this.gpu) return this.gpu.placeholder;
    let texture = this.readback?.getTexture();
    if (this.source || this.array) {
      texture = this.maybeUpdateSource();
    }
    return texture ?? RgbaArray.getEmpty();
  }

  // Create or get a THREE.DataArrayTexture from the data array.
  private maybeUpdateSource(): THREE.DataArrayTexture {
    if (!this.array) {
      throw new Error("No array");
    }

    if (this.needsUpdate || !this.source) {
      this.needsUpdate = false;

      if (this.source) {
        const { width, height, depth } = this.source.image;
        if (this.capacity !== width * height * depth) {
          this.source.dispose();
          this.source = null;
        }
      }
      if (!this.source) {
        const { width, height, depth } = getTextureSize(this.capacity);
        this.source = new THREE.DataArrayTexture(
          this.array as Uint8Array<ArrayBuffer>,
          width,
          height,
          depth,
        );
        this.source.format = THREE.RGBAFormat;
        this.source.type = THREE.UnsignedByteType;
        this.source.internalFormat = "RGBA8";
        this.source.needsUpdate = true;
      } else if (this.array.buffer !== this.source.image.data.buffer) {
        this.source.image.data = new Uint8Array(this.array.buffer);
      }
      this.source.needsUpdate = true;
    }
    return this.source;
  }

  // Generate the RGBA8 values from a Rgba8Readback dyno program.
  render({
    reader,
    count,
    renderer,
  }: { reader: Rgba8Readback; count: number; renderer: AnyRenderer }) {
    if (isWebGPU(renderer)) {
      throw new Error(
        "RgbaArray.render needs a WebGLRenderer: on WebGPU use SparkRenderer.getRgba",
      );
    }
    const gl = renderer as THREE.WebGLRenderer;
    if (!this.readback) {
      this.readback = new Readback({ renderer: gl });
    }
    this.readback.render({ reader, count, renderer: gl });
    this.capacity = this.readback.capacity;
    this.count = this.readback.count;
  }

  // Extract the RGBA8 values from a PackedSplats collection.
  fromPackedSplats({
    packedSplats,
    base,
    count,
    renderer,
  }: {
    packedSplats: PackedSplats;
    base: number;
    count: number;
    renderer: AnyRenderer;
  }) {
    if (isWebGPU(renderer)) {
      this.fromPackedArray(packedSplats, base, count);
      return this;
    }
    const { dynoSplats, dynoBase, dynoCount, reader } = RgbaArray.makeDynos();
    dynoSplats.packedSplats = packedSplats;
    dynoBase.value = base;
    dynoCount.value = count;
    this.render({ reader, count, renderer });
    return this;
  }

  // The RGBA8 values decoded on the CPU, as the WebGL readback computes them.
  private fromPackedArray(
    packedSplats: PackedSplats,
    base: number,
    maxCount: number,
  ) {
    const packed = packedSplats.packedArray;
    const count = Math.max(
      0,
      Math.min(maxCount, packedSplats.numSplats - base),
    );
    const array = this.ensureCapacity(count);
    const { rgbMin, rgbMax, lodOpacity } = packedSplats.splatEncoding;
    if (packed) {
      const bytes = new Uint8Array(packed.buffer, packed.byteOffset);
      const plain = rgbMin === 0 && rgbMax === 1 && !lodOpacity;
      const toByte = (v: number) =>
        Math.round(Math.min(1, Math.max(0, v)) * 255);
      const range = (rgbMax - rgbMin) / 255;
      for (let i = 0; i < count; i++) {
        const s = (base + i) * 16;
        for (let c = 0; c < 4; c++) {
          const byte = bytes[s + c];
          array[i * 4 + c] = plain
            ? byte
            : c < 3
              ? toByte(rgbMin + byte * range)
              : toByte(((lodOpacity ? 2 : 1) * byte) / 255);
        }
      }
    }
    this.count = count;
    this.needsUpdate = true;
  }

  /**
   * @internal The GPU texture for `count` values that a WebGPU kernel will
   * write (SparkRenderer.getRgba); the array then reads from it.
   */
  gpuTexture(device: GPUDevice, count: number): GPUTexture {
    const { width, height, depth, maxSplats } = getTextureSize(count);
    let gpu = this.gpu;
    if (
      !gpu ||
      gpu.device !== device ||
      gpu.texture.width !== width ||
      gpu.texture.height !== height ||
      gpu.texture.depthOrArrayLayers !== depth
    ) {
      this.dispose();
      const texture = device.createTexture({
        label: "RgbaArray",
        size: { width, height, depthOrArrayLayers: depth },
        format: "rgba8unorm",
        usage:
          GPUTextureUsage.STORAGE_BINDING |
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.COPY_SRC,
      });
      const placeholder = new THREE.DataArrayTexture(
        null,
        width,
        height,
        depth,
      );
      placeholder.format = THREE.RGBAFormat;
      placeholder.type = THREE.UnsignedByteType;
      gpuTextures.set(placeholder, texture);
      gpu = { device, texture, placeholder };
      this.gpu = gpu;
    }
    // A new version: generates that read the array run again.
    gpu.placeholder.version += 1;
    this.capacity = maxSplats;
    this.count = count;
    this.array = null;
    return gpu.texture;
  }

  // Read back the RGBA8 values from the readback buffer.
  async read(): Promise<Uint8Array> {
    if (this.gpu) return this.readGpu(this.gpu);
    if (!this.readback) {
      throw new Error("No readback");
    }
    if (!this.array || this.array.length < this.count * 4) {
      this.array = new Uint8Array(this.capacity * 4);
    }
    const result = await this.readback.readback({ readback: this.array });
    return result.subarray(0, this.count * 4);
  }

  private async readGpu({
    device,
    texture,
  }: { device: GPUDevice; texture: GPUTexture }): Promise<Uint8Array> {
    const { width, height, depthOrArrayLayers } = texture;
    const bytes = width * height * depthOrArrayLayers * 4;
    const staging = device.createBuffer({
      label: "RgbaArray readback",
      size: bytes,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const encoder = device.createCommandEncoder({ label: "RgbaArray read" });
    // Rows of 2048 texels are 256-byte aligned: index order, tightly packed.
    encoder.copyTextureToBuffer(
      { texture },
      { buffer: staging, bytesPerRow: width * 4, rowsPerImage: height },
      { width, height, depthOrArrayLayers },
    );
    device.queue.submit([encoder.finish()]);
    try {
      await staging.mapAsync(GPUMapMode.READ);
      if (!this.array || this.array.length < bytes) {
        this.array = new Uint8Array(bytes);
      }
      this.array.set(new Uint8Array(staging.getMappedRange(), 0, bytes));
    } finally {
      staging.destroy();
    }
    return this.array.subarray(0, this.count * 4);
  }

  async getArray(): Promise<Uint8Array> {
    if (this.gpu) return this.read();
    if (this.readback) {
      return await this.read();
    }
    if (this.array) {
      return this.array;
    }
    throw new Error("No array");
  }

  private static emptySource: THREE.DataArrayTexture | null = null;

  // Can be used where you need an uninitialized THREE.DataArrayTexture like
  // a uniform you will update with the result of this.getTexture() later.
  static getEmpty(): THREE.DataArrayTexture {
    if (!RgbaArray.emptySource) {
      const emptyArray = new Uint8Array(1 * 4);
      RgbaArray.emptySource = new THREE.DataArrayTexture(emptyArray, 1, 1, 1);
      RgbaArray.emptySource.format = THREE.RGBAFormat;
      RgbaArray.emptySource.type = THREE.UnsignedByteType;
      RgbaArray.emptySource.internalFormat = "RGBA8";
      RgbaArray.emptySource.needsUpdate = true;
    }
    return RgbaArray.emptySource;
  }

  private static dynos: {
    dynoSplats: DynoPackedSplats;
    dynoBase: DynoInt<string>;
    dynoCount: DynoInt<string>;
    reader: DynoBlock<{ index: "int" }, { rgba8: "vec4" }>;
  } | null = null;

  // Create a dyno program that can extract RGBA8 values from a PackedSplats
  private static makeDynos() {
    if (!RgbaArray.dynos) {
      const dynoSplats = new DynoPackedSplats();
      const dynoBase = new DynoInt({ value: 0 });
      const dynoCount = new DynoInt({ value: 0 });
      const reader = dynoBlock(
        { index: "int" },
        { rgba8: "vec4" },
        ({ index }) => {
          if (!index) {
            throw new Error("index is undefined");
          }
          index = add(index, dynoBase);
          const gsplat = readPackedSplatRange(
            dynoSplats,
            index,
            dynoBase,
            dynoCount,
          );
          return { rgba8: splitGsplat(gsplat).outputs.rgba };
        },
      );
      RgbaArray.dynos = { dynoSplats, dynoBase, dynoCount, reader };
    }
    return RgbaArray.dynos;
  }
}

// Dyno types and definitions

export const TRgbaArray = { type: "RgbaArray" } as { type: "RgbaArray" };

export const defineRgbaArray = unindent(/* glsl */ `
  struct RgbaArray {
    sampler2DArray texture;
    int count;
  };
`);

// On WebGPU the struct's texture is a binding of its own (wgslStructTexture).
registerWgslGlobal(defineRgbaArray, "");
registerWgslStruct(TRgbaArray, { texture: "sampler2DArray", count: "int" });

export function readRgbaArray(
  rgba: DynoVal<typeof TRgbaArray>,
  index: DynoVal<"int">,
): DynoVal<"vec4"> {
  const dyno = new Dyno<
    { rgba: typeof TRgbaArray; index: "int" },
    { rgba: "vec4" }
  >({
    inTypes: { rgba: TRgbaArray, index: "int" },
    outTypes: { rgba: "vec4" },
    inputs: { rgba, index },
    globals: () => [defineRgbaArray],
    statements: ({ inputs, outputs }) =>
      unindentLines(/* glsl */ `
        if ((${inputs.index} >= 0) && (${inputs.index} < ${inputs.rgba}.count)) {
          ${outputs.rgba} = texelFetch(${inputs.rgba}.texture, splatTexCoord(${inputs.index}), 0);
        } else {
          ${outputs.rgba} = vec4(0.0, 0.0, 0.0, 0.0);
        }
      `),
    wgsl: {
      statements: ({ inputs, outputs }) => {
        const texture = wgslStructTexture(inputs.rgba as string, "texture");
        return unindentLines(/* wgsl */ `
          if ((${inputs.index} >= 0) && (${inputs.index} < ${inputs.rgba}.count)) {
            let coord = splatTexCoord(${inputs.index});
            ${outputs.rgba} = textureLoad(${texture}, coord.xy, coord.z, 0);
          } else {
            ${outputs.rgba} = vec4f(0.0);
          }
        `);
      },
    },
  });
  return dyno.outputs.rgba;
}
