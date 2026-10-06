// Buffer and texture helpers shared by the WebGPU modules. GPUBufferUsage and
// GPUTextureUsage are read lazily: importing Spark must work in browsers
// without WebGPU.

/**
 * A storage buffer that can be copied to and from, rounded up to 16 bytes;
 * `extraUsage` adds flags (INDIRECT for draw arguments).
 */
export function createStorage(
  device: GPUDevice,
  bytes: number,
  label: string,
  extraUsage = 0,
) {
  return device.createBuffer({
    label,
    size: Math.max(16, Math.ceil(bytes / 16) * 16),
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_DST |
      GPUBufferUsage.COPY_SRC |
      extraUsage,
  });
}

/** A storage buffer holding `data`. */
export function upload(device: GPUDevice, data: Uint32Array, label: string) {
  const buffer = createStorage(device, data.byteLength, label);
  device.queue.writeBuffer(
    buffer,
    0,
    data.buffer,
    data.byteOffset,
    data.byteLength,
  );
  return buffer;
}

/** A uniform buffer of `bytes`, written with queue.writeBuffer. */
export function createUniform(device: GPUDevice, bytes: number, label: string) {
  return device.createBuffer({
    label,
    size: bytes,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
}

/**
 * Writes `data` at `offset`. writeBuffer takes whole words in every browser,
 * so a ragged tail is padded through a copy rather than read past the end
 * of the caller's view.
 */
export function writeWords(
  device: GPUDevice,
  buffer: GPUBuffer,
  data: ArrayBufferView,
  offset = 0,
) {
  let bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  if (bytes.byteLength % 4 !== 0) {
    const padded = new Uint8Array(Math.ceil(bytes.byteLength / 4) * 4);
    padded.set(bytes);
    bytes = padded;
  }
  device.queue.writeBuffer(buffer, offset, bytes);
}

/** A buffer to copy GPU results into and map for reading. */
export function createReadback(
  device: GPUDevice,
  bytes: number,
  label: string,
) {
  return device.createBuffer({
    label,
    size: Math.ceil(bytes / 4) * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
}

/**
 * The contents of a readback buffer once the GPU has written it, after
 * which the buffer is destroyed.
 */
export async function readAndDestroy(staging: GPUBuffer): Promise<ArrayBuffer> {
  try {
    await staging.mapAsync(GPUMapMode.READ);
    return staging.getMappedRange().slice(0);
  } finally {
    staging.destroy();
  }
}

/** Bytes per row of a texture copy: WebGPU aligns rows to 256 bytes. */
export function copyBytesPerRow(rowBytes: number) {
  return Math.ceil(rowBytes / 256) * 256;
}

/**
 * `current` when it has `desc`'s size, format, sample count, dimension and
 * usage; otherwise a new texture, and `current` is destroyed.
 */
export function reuseTexture(
  device: GPUDevice,
  current: GPUTexture | null | undefined,
  desc: GPUTextureDescriptor & { size: readonly number[] },
): GPUTexture {
  const [width, height = 1, layers = 1] = desc.size;
  if (
    current &&
    current.width === width &&
    current.height === height &&
    current.depthOrArrayLayers === layers &&
    current.format === desc.format &&
    current.sampleCount === (desc.sampleCount ?? 1) &&
    current.dimension === (desc.dimension ?? "2d") &&
    current.usage === desc.usage
  ) {
    return current;
  }
  current?.destroy();
  return device.createTexture(desc);
}
