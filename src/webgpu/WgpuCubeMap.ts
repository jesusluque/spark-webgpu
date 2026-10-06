// Cube maps and environment maps of a WgpuSplatRenderer's splats and a three
// scene, as SparkRenderer.renderCubeMap / renderEnvMap / readCubeTargets.
//
// Each face renders into a 2D RenderTarget with a depth texture, three's
// scene first and the splats over it (so opaque objects occlude them), and is
// then copied into its layer of the cube texture: three's CubeRenderTarget
// keeps its depth buffer internal, which the splat pass can't test against.
// Faces use three's CubeCamera orientations, so the result samples as any
// cube map three rendered itself.
//
// Usage:
//   const cube = new WgpuCubeMap(splats);
//   const envMap = await cube.renderEnvMap({ scene, worldCenter, hideObjects: [duck] });
//   duck.material.envMap = envMap;

import * as THREE from "three";
import type { WgpuSplatRenderer } from "./WgpuSplatRenderer";

export interface WgpuCubeMapOptions {
  /** three objects to draw under the splats (lights, meshes...). */
  scene: THREE.Scene;
  worldCenter: THREE.Vector3;
  /** Face size in pixels (default 256). */
  size?: number;
  near?: number;
  far?: number;
  /** three objects, or splat meshes' objects, left out of the render. */
  hideObjects?: THREE.Object3D[];
  /** Mipmapped and linear, for filtering (renderEnvMap sets it). */
  filter?: boolean;
}

interface TextureBackend {
  get(resource: object): { texture?: GPUTexture } | undefined;
  generateMipmaps?(texture: THREE.Texture): void;
}

interface RendererLike {
  backend: TextureBackend;
  getRenderTarget(): THREE.RenderTarget | null;
  getActiveCubeFace(): number;
  getActiveMipmapLevel(): number;
  setRenderTarget(target: THREE.RenderTarget | null, face?: number): void;
  render(scene: THREE.Object3D, camera: THREE.Camera): void;
  initTexture(texture: THREE.Texture): void;
  readRenderTargetPixelsAsync(
    target: THREE.RenderTarget,
    x: number,
    y: number,
    width: number,
    height: number,
    textureIndex?: number,
    faceIndex?: number,
  ): Promise<ArrayBufferView>;
  coordinateSystem: THREE.CoordinateSystem;
  xr?: { enabled: boolean };
}

export class WgpuCubeMap {
  private cube: {
    target: THREE.RenderTarget;
    face: THREE.RenderTarget;
    camera: THREE.CubeCamera;
    key: string;
  } | null = null;
  private pmrem: { fromCubemap(t: THREE.Texture): THREE.RenderTarget } | null =
    null;

  /**
   * `PMREMGenerator`: three/webgpu's, for renderEnvMap, when the "three"
   * Spark resolves is not three/webgpu (a bundler resolving bare imports
   * itself, as Vite's dev server does, gives three's WebGL one).
   */
  constructor(
    readonly splats: WgpuSplatRenderer,
    private readonly classes: {
      PMREMGenerator?: new (
        renderer: never,
      ) => { fromCubemap(t: THREE.Texture): THREE.RenderTarget };
    } = {},
  ) {}

  private get renderer(): RendererLike {
    return this.splats.renderer as unknown as RendererLike;
  }

  /** The cube target the last renderCubeMap drew into. */
  get target(): THREE.RenderTarget | null {
    return this.cube?.target ?? null;
  }

  private prepare(size: number, near: number, far: number, filter: boolean) {
    const key = `${size}/${near}/${far}/${filter}`;
    if (this.cube?.key === key) return this.cube;
    this.dispose();
    // three/webgpu's CubeRenderTarget; WebGLCubeRenderTarget otherwise (the
    // WebGPU backend renders into either).
    const CubeTarget =
      (
        THREE as unknown as {
          CubeRenderTarget?: typeof THREE.WebGLCubeRenderTarget;
        }
      ).CubeRenderTarget ?? THREE.WebGLCubeRenderTarget;
    const colorSpace = filter
      ? THREE.LinearSRGBColorSpace
      : THREE.SRGBColorSpace;
    const target = new CubeTarget(size, {
      format: THREE.RGBAFormat,
      type: THREE.UnsignedByteType,
      generateMipmaps: filter,
      minFilter: filter ? THREE.LinearMipMapLinearFilter : THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      colorSpace,
    }) as unknown as THREE.RenderTarget;
    const face = new THREE.RenderTarget(size, size, {
      type: THREE.UnsignedByteType,
      colorSpace,
      depthTexture: new THREE.DepthTexture(size, size),
    });
    const camera = new THREE.CubeCamera(
      near,
      far,
      target as unknown as THREE.WebGLCubeRenderTarget,
    );
    this.cube = { target, face, camera, key };
    return this.cube;
  }

  /**
   * Renders the six faces around `worldCenter` and returns the cube texture.
   * Needs the GPU sort (each face is sorted in its own submit).
   */
  async renderCubeMap({
    scene,
    worldCenter,
    size = 256,
    near = 0.1,
    far = 1000,
    hideObjects = [],
    filter = false,
  }: WgpuCubeMapOptions): Promise<THREE.CubeTexture> {
    const { splats, renderer } = this;
    if (splats.options.sort !== "gpu") {
      throw new Error(
        "WgpuCubeMap: needs WgpuSplatRenderer's GPU sort (the CPU sort draws a frame behind)",
      );
    }
    const { target, face, camera } = this.prepare(size, near, far, filter);
    camera.position.copy(worldCenter);
    camera.updateMatrixWorld();
    const cam = camera as unknown as {
      coordinateSystem: THREE.CoordinateSystem | null;
      updateCoordinateSystem(): void;
    };
    if (cam.coordinateSystem !== renderer.coordinateSystem) {
      cam.coordinateSystem = renderer.coordinateSystem;
      cam.updateCoordinateSystem();
    }

    const hidden = hideObjects.filter((o) => o.visible);
    for (const o of hidden) o.visible = false;
    // Splat meshes whose object is hidden sit out, put back after.
    const all = [...splats.meshes];
    const kept = all.filter((m) => !hideObjects.includes(m.object));
    splats.meshes.splice(0, all.length, ...kept);

    const previous = renderer.getRenderTarget();
    const previousFace = renderer.getActiveCubeFace();
    const xr = renderer.xr?.enabled ?? false;
    if (renderer.xr) renderer.xr.enabled = false;
    try {
      renderer.initTexture(target.texture);
      const cubeTexture = renderer.backend.get(target.texture)?.texture;
      if (!cubeTexture) throw new Error("WgpuCubeMap: no cube texture");
      const device = splats.device;
      camera.children.forEach((faceCamera, i) => {
        renderer.setRenderTarget(face);
        renderer.render(scene, faceCamera as THREE.Camera);
        splats.markDirty();
        splats.render(faceCamera as THREE.Camera, face);
        const faceTexture = renderer.backend.get(face.texture)?.texture;
        if (!faceTexture) throw new Error("WgpuCubeMap: no face texture");
        const encoder = device.createCommandEncoder({ label: "cube face" });
        encoder.copyTextureToTexture(
          { texture: faceTexture },
          { texture: cubeTexture, origin: [0, 0, i] },
          [size, size, 1],
        );
        device.queue.submit([encoder.finish()]);
      });
      if (filter) renderer.backend.generateMipmaps?.(target.texture);
    } finally {
      splats.meshes.splice(0, splats.meshes.length, ...all);
      splats.markDirty();
      for (const o of hidden) o.visible = true;
      renderer.setRenderTarget(previous, previousFace);
      if (renderer.xr) renderer.xr.enabled = xr;
    }
    return target.texture as unknown as THREE.CubeTexture;
  }

  /**
   * The last cube map's faces as RGBA8 pixels, rows top to bottom, in
   * SparkRenderer.readCubeTargets' layout: three's WebGPU CubeCamera stores
   * +X / -X swapped and every face mirrored in x (it samples them so), which
   * this undoes.
   */
  async readCubeTargets(): Promise<Uint8Array[]> {
    const target = this.cube?.target;
    if (!target) throw new Error("WgpuCubeMap: no cube render");
    const { width, height } = target;
    const webgpu =
      this.renderer.coordinateSystem === THREE.WebGPUCoordinateSystem;
    const order = webgpu ? [1, 0, 2, 3, 4, 5] : [0, 1, 2, 3, 4, 5];
    const faces = await Promise.all(
      order.map((i) =>
        this.renderer.readRenderTargetPixelsAsync(
          target,
          0,
          0,
          width,
          height,
          0,
          i,
        ),
      ),
    );
    const row = width * 4;
    return faces.map((v) => {
      const src = new Uint8Array(v.buffer, v.byteOffset, v.byteLength);
      // Rows may come padded to 256 bytes.
      const stride = src.length / height;
      const out = new Uint8Array(row * height);
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const from = y * stride + 4 * (webgpu ? width - 1 - x : x);
          out.set(src.subarray(from, from + 4), y * row + 4 * x);
        }
      }
      return out;
    });
  }

  /**
   * renderCubeMap, prefiltered by three's PMREMGenerator: a texture for
   * MeshStandardMaterial.envMap.
   */
  async renderEnvMap(
    options: Omit<WgpuCubeMapOptions, "filter">,
  ): Promise<THREE.Texture> {
    const cube = await this.renderCubeMap({ ...options, filter: true });
    const Generator = (this.classes.PMREMGenerator ??
      THREE.PMREMGenerator) as unknown as new (
      renderer: unknown,
    ) => { fromCubemap(t: THREE.Texture): THREE.RenderTarget };
    this.pmrem ??= new Generator(this.splats.renderer);
    return this.pmrem.fromCubemap(cube).texture;
  }

  dispose() {
    this.cube?.target.dispose();
    this.cube?.face.depthTexture?.dispose();
    this.cube?.face.dispose();
    this.cube = null;
  }
}
