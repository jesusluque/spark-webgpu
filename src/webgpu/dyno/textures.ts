// GPU copies of the three.js textures dyno sampler uniforms hold, uploaded
// on first use and again when texture.version changes (needsUpdate). Data
// textures (DataTexture, DataArrayTexture, Data3DTexture) upload their typed
// array; anything else is treated as an image and copied as rgba8unorm.
//
// Textures rather than storage buffers: they map one to one onto GLSL
// samplers (texelFetch -> textureLoad, texture -> textureSampleLevel) and
// don't count against the 8 storage buffers a stage may use.

import * as THREE from "three";

interface FormatInfo {
  format: GPUTextureFormat;
  bytes: number;
}

const FORMATS: Record<number, Record<number, FormatInfo>> = {
  [THREE.RGBAFormat]: {
    [THREE.UnsignedByteType]: { format: "rgba8unorm", bytes: 4 },
    [THREE.HalfFloatType]: { format: "rgba16float", bytes: 8 },
    [THREE.FloatType]: { format: "rgba32float", bytes: 16 },
  },
  [THREE.RGFormat]: {
    [THREE.UnsignedByteType]: { format: "rg8unorm", bytes: 2 },
    [THREE.HalfFloatType]: { format: "rg16float", bytes: 4 },
    [THREE.FloatType]: { format: "rg32float", bytes: 8 },
  },
  [THREE.RedFormat]: {
    [THREE.UnsignedByteType]: { format: "r8unorm", bytes: 1 },
    [THREE.HalfFloatType]: { format: "r16float", bytes: 2 },
    [THREE.FloatType]: { format: "r32float", bytes: 4 },
  },
  [THREE.RGBAIntegerFormat]: {
    [THREE.UnsignedByteType]: { format: "rgba8uint", bytes: 4 },
    [THREE.ByteType]: { format: "rgba8sint", bytes: 4 },
    [THREE.UnsignedShortType]: { format: "rgba16uint", bytes: 8 },
    [THREE.ShortType]: { format: "rgba16sint", bytes: 8 },
    [THREE.UnsignedIntType]: { format: "rgba32uint", bytes: 16 },
    [THREE.IntType]: { format: "rgba32sint", bytes: 16 },
  },
  [THREE.RGIntegerFormat]: {
    [THREE.UnsignedByteType]: { format: "rg8uint", bytes: 2 },
    [THREE.UnsignedShortType]: { format: "rg16uint", bytes: 4 },
    [THREE.UnsignedIntType]: { format: "rg32uint", bytes: 8 },
    [THREE.IntType]: { format: "rg32sint", bytes: 8 },
  },
  [THREE.RedIntegerFormat]: {
    [THREE.UnsignedByteType]: { format: "r8uint", bytes: 1 },
    [THREE.UnsignedShortType]: { format: "r16uint", bytes: 2 },
    [THREE.UnsignedIntType]: { format: "r32uint", bytes: 4 },
    [THREE.IntType]: { format: "r32sint", bytes: 4 },
  },
};

export function textureFormat(texture: THREE.Texture): FormatInfo {
  const info = FORMATS[texture.format]?.[texture.type];
  if (!info) {
    throw new Error(
      `dyno texture: no WebGPU format for three format ${texture.format} / type ${texture.type}`,
    );
  }
  return info;
}

interface DataImage {
  data: ArrayBufferView;
  width: number;
  height: number;
  depth?: number;
}

function isData(texture: THREE.Texture) {
  const t = texture as THREE.Texture & {
    isDataTexture?: boolean;
    isDataArrayTexture?: boolean;
    isData3DTexture?: boolean;
  };
  return t.isDataTexture || t.isDataArrayTexture || t.isData3DTexture;
}

/** Whether a float texture can be sampled with filtering (not 32-bit). */
export function isFilterable(texture: THREE.Texture): boolean {
  return !isData(texture) || !textureFormat(texture).format.endsWith("32float");
}

export class TextureCache {
  private entries = new WeakMap<
    THREE.Texture,
    { texture: GPUTexture; version: number }
  >();
  private samplers = new Map<string, GPUSampler>();

  constructor(readonly device: GPUDevice) {}

  texture(texture: THREE.Texture): GPUTexture {
    const entry = this.entries.get(texture);
    if (entry && entry.version === texture.version) return entry.texture;
    const gpu = this.upload(texture, entry?.texture);
    if (entry?.texture && entry.texture !== gpu) entry.texture.destroy();
    this.entries.set(texture, { texture: gpu, version: texture.version });
    return gpu;
  }

  view(
    texture: THREE.Texture,
    dimension: GPUTextureViewDimension,
  ): GPUTextureView {
    return this.texture(texture).createView({ dimension });
  }

  sampler(texture: THREE.Texture, filterable: boolean): GPUSampler {
    const linear = (f: number) =>
      filterable && f !== THREE.NearestFilter ? "linear" : "nearest";
    const wrap = (w: number): GPUAddressMode =>
      w === THREE.RepeatWrapping
        ? "repeat"
        : w === THREE.MirroredRepeatWrapping
          ? "mirror-repeat"
          : "clamp-to-edge";
    const desc: GPUSamplerDescriptor = {
      magFilter: linear(texture.magFilter),
      minFilter: linear(texture.minFilter),
      addressModeU: wrap(texture.wrapS),
      addressModeV: wrap(texture.wrapT),
    };
    const key = JSON.stringify(desc);
    let s = this.samplers.get(key);
    if (!s) {
      s = this.device.createSampler(desc);
      this.samplers.set(key, s);
    }
    return s;
  }

  private upload(texture: THREE.Texture, previous?: GPUTexture): GPUTexture {
    const { device } = this;
    const usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST;
    if (!isData(texture)) {
      const image = texture.image as ImageBitmap;
      const size = { width: image.width, height: image.height };
      const gpu =
        previous &&
        previous.width === size.width &&
        previous.height === size.height &&
        previous.format === "rgba8unorm"
          ? previous
          : device.createTexture({
              size,
              format: "rgba8unorm",
              usage: usage | GPUTextureUsage.RENDER_ATTACHMENT,
            });
      device.queue.copyExternalImageToTexture(
        { source: image, flipY: texture.flipY },
        { texture: gpu },
        size,
      );
      return gpu;
    }
    const image = texture.image as DataImage;
    const { format, bytes } = textureFormat(texture);
    const is3D = (texture as THREE.Data3DTexture).isData3DTexture;
    const size = {
      width: image.width,
      height: image.height,
      depthOrArrayLayers: image.depth ?? 1,
    };
    const dimension: GPUTextureDimension = is3D ? "3d" : "2d";
    const gpu =
      previous &&
      previous.width === size.width &&
      previous.height === size.height &&
      previous.depthOrArrayLayers === size.depthOrArrayLayers &&
      previous.format === format &&
      previous.dimension === dimension
        ? previous
        : device.createTexture({ size, format, usage, dimension });
    const { data } = image;
    device.queue.writeTexture(
      { texture: gpu },
      data.buffer,
      {
        offset: data.byteOffset,
        bytesPerRow: size.width * bytes,
        rowsPerImage: size.height,
      },
      size,
    );
    return gpu;
  }
}
