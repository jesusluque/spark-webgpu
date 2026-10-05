// Buffer helpers shared by the WebGPU modules. GPUBufferUsage is read
// lazily: importing Spark must work in browsers without WebGPU.

export function createStorage(device: GPUDevice, bytes: number, label: string) {
  return device.createBuffer({
    label,
    size: Math.max(16, Math.ceil(bytes / 16) * 16),
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_DST |
      GPUBufferUsage.COPY_SRC,
  });
}

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
