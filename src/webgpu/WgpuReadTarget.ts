// Offscreen splat renders read back to the CPU, as SparkRenderer's `target`
// option with renderTarget / readTarget / renderReadTarget: a three scene and
// a WgpuSplatRenderer's splats into an 8-bit sRGB RenderTarget (with depth,
// so opaque objects occlude splats), optionally supersampled.
//
// Usage:
//   const offscreen = new WgpuReadTarget(splats, { width: 64, height: 64, superXY: 2 });
//   const rgba = await offscreen.renderReadTarget({ scene, camera });

import * as THREE from "three";
import type { WgpuSplatRenderer } from "./WgpuSplatRenderer";

/**
 * Bytes per row of an RGBA8 readback from readRenderTargetPixelsAsync: rows
 * come padded to 256 bytes (WebGPU's copy alignment) unless tightly packed.
 */
export function rowStride(length: number, width: number, height: number) {
  return length === width * height * 4
    ? width * 4
    : Math.ceil((width * 4) / 256) * 256;
}

/**
 * `target`'s pixels as RGBA8, averaged over superXY x superXY blocks, rows
 * bottom to top as WebGL's readPixels gives them. Reuses `out` when it has
 * the right size.
 */
export async function readTargetPixels(
  renderer: Pick<RendererLike, "readRenderTargetPixelsAsync">,
  target: THREE.RenderTarget,
  superXY: number,
  out?: Uint8Array,
): Promise<Uint8Array> {
  const { width, height } = target;
  const data = await renderer.readRenderTargetPixelsAsync(
    target,
    0,
    0,
    width,
    height,
  );
  const src = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  const stride = rowStride(src.length, width, height);
  const w = width / superXY;
  const h = height / superXY;
  const pixels =
    out && out.length === w * h * 4 ? out : new Uint8Array(w * h * 4);
  const n = superXY * superXY;
  for (let y = 0; y < h; y++) {
    const outRow = (h - 1 - y) * w * 4;
    for (let x = 0; x < w; x++) {
      for (let c = 0; c < 4; c++) {
        let sum = 0;
        for (let sy = 0; sy < superXY; sy++) {
          const row = (y * superXY + sy) * stride;
          for (let sx = 0; sx < superXY; sx++) {
            sum += src[row + (x * superXY + sx) * 4 + c];
          }
        }
        pixels[outRow + x * 4 + c] = sum / n;
      }
    }
  }
  return pixels;
}

export interface WgpuReadTargetOptions {
  width: number;
  height: number;
  /** Render at superXY times the size and average down on read (default 1). */
  superXY?: number;
}

interface RendererLike {
  getRenderTarget(): THREE.RenderTarget | null;
  setRenderTarget(target: THREE.RenderTarget | null): void;
  render(scene: THREE.Object3D, camera: THREE.Camera): void;
  readRenderTargetPixelsAsync(
    target: THREE.RenderTarget,
    x: number,
    y: number,
    width: number,
    height: number,
  ): Promise<ArrayBufferView>;
}

export class WgpuReadTarget {
  readonly target: THREE.RenderTarget;
  readonly superXY: number;
  private pixels: Uint8Array | null = null;

  constructor(
    readonly splats: WgpuSplatRenderer,
    { width, height, superXY = 1 }: WgpuReadTargetOptions,
  ) {
    this.superXY = Math.max(1, Math.floor(superXY));
    const w = width * this.superXY;
    const h = height * this.superXY;
    this.target = new THREE.RenderTarget(w, h, {
      type: THREE.UnsignedByteType,
      colorSpace: THREE.SRGBColorSpace,
      depthTexture: new THREE.DepthTexture(w, h),
    });
  }

  private get renderer(): RendererLike {
    return this.splats.renderer as unknown as RendererLike;
  }

  /** Renders `scene` (three objects) and the splats into the target. */
  render({
    scene,
    camera,
  }: { scene: THREE.Scene; camera: THREE.Camera }): THREE.RenderTarget {
    const { renderer, target } = this;
    const previous = renderer.getRenderTarget();
    try {
      renderer.setRenderTarget(target);
      renderer.render(scene, camera);
      this.splats.render(camera, target);
    } finally {
      renderer.setRenderTarget(previous);
    }
    return target;
  }

  /**
   * The last render as RGBA8, averaged over superXY x superXY blocks. Rows
   * bottom to top, as SparkRenderer.readTarget (WebGL's readPixels) gives
   * them. The array is reused by the next call.
   */
  async readTarget(): Promise<Uint8Array> {
    this.pixels = await readTargetPixels(
      this.renderer,
      this.target,
      this.superXY,
      this.pixels ?? undefined,
    );
    return this.pixels;
  }

  async renderReadTarget(options: {
    scene: THREE.Scene;
    camera: THREE.Camera;
  }): Promise<Uint8Array> {
    this.render(options);
    return this.readTarget();
  }

  dispose() {
    this.target.depthTexture?.dispose();
    this.target.dispose();
  }
}
