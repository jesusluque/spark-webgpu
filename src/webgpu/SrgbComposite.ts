// Splats blended in sRGB space over a linear target, as WebGL Spark blends
// them on the canvas. Hardware blending works on the stored (linear) values,
// so the splats first accumulate in a layer of their own, cleared to (0, 0,
// 0, 1): premultiplied sRGB colour L and, in alpha, the transmittance T left
// by all of them. Blending back to front over a background B gives
// L + T * B in sRGB, so a full-screen pass composites
//   linear(L + T * srgb(dst))
// over a copy of the target. srgb() and linear() are three's own transfer
// functions, so where no splat covers a pixel three's output pass returns
// exactly what it would have.

const COMPOSITE_WGSL = (multisampled: boolean) => /* wgsl */ `
@group(0) @binding(0) var dstTex: ${multisampled ? "texture_multisampled_2d" : "texture_2d"}<f32>;
@group(0) @binding(1) var layerTex: ${multisampled ? "texture_multisampled_2d" : "texture_2d"}<f32>;

@vertex
fn vertexMain(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let uv = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4f(uv * 2.0 - 1.0, 0.0, 1.0);
}

// three's sRGBTransferOETF / sRGBTransferEOTF.
fn toSrgb(c: vec3f) -> vec3f {
  return select(1.055 * pow(c, vec3f(0.41666)) - 0.055, c * 12.92, c <= vec3f(0.0031308));
}
fn toLinear(c: vec3f) -> vec3f {
  return select(pow(c * 0.9478672986 + 0.0521327014, vec3f(2.4)), c * 0.0773993808, c <= vec3f(0.04045));
}

@fragment
fn fragmentMain(@builtin(position) p: vec4f${multisampled ? ", @builtin(sample_index) s: u32" : ""}) -> @location(0) vec4f {
  let xy = vec2i(p.xy);
  let dst = textureLoad(dstTex, xy, ${multisampled ? "s" : "0"});
  let layer = textureLoad(layerTex, xy, ${multisampled ? "s" : "0"});
  let t = layer.a;
  let rgb = layer.rgb + t * toSrgb(max(dst.rgb, vec3f(0.0)));
  return vec4f(toLinear(rgb), 1.0 - t + t * dst.a);
}
`;

/** The splat layer's format: half floats keep L and T exact enough. */
export const SRGB_LAYER_FORMAT: GPUTextureFormat = "rgba16float";

/** Blend state for the layer: colour over, alpha the transmittance. */
export const SRGB_LAYER_BLEND: GPUBlendState = {
  color: { srcFactor: "one", dstFactor: "one-minus-src-alpha" },
  alpha: { srcFactor: "zero", dstFactor: "one-minus-src-alpha" },
};

interface Scratch {
  layer: GPUTexture;
  copy: GPUTexture;
}

export class SrgbComposite {
  private pipelines = new Map<string, GPURenderPipeline>();
  private scratch: Scratch | null = null;

  constructor(readonly device: GPUDevice) {}

  /** A layer for `color` (the target's attachment texture), cleared. */
  beginLayer(
    encoder: GPUCommandEncoder,
    color: GPUTexture,
    depth: GPURenderPassDepthStencilAttachment | undefined,
  ): GPURenderPassEncoder {
    const { layer } = this.ensureScratch(color);
    return encoder.beginRenderPass({
      label: "splat layer",
      colorAttachments: [
        {
          view: layer.createView(),
          loadOp: "clear",
          storeOp: "store",
          clearValue: [0, 0, 0, 1],
        },
      ],
      depthStencilAttachment: depth,
    });
  }

  /**
   * Composites the layer over `color`, through `view` (its attachment view,
   * which may resolve), within `viewport` when given.
   */
  composite(
    encoder: GPUCommandEncoder,
    color: GPUTexture,
    attachment: Pick<GPURenderPassColorAttachment, "view" | "resolveTarget">,
    viewport?: { x: number; y: number; z: number; w: number } | null,
  ) {
    const { layer, copy } = this.ensureScratch(color);
    encoder.copyTextureToTexture({ texture: color }, { texture: copy }, [
      color.width,
      color.height,
    ]);
    const pipeline = this.pipeline(color.format, color.sampleCount);
    const pass = encoder.beginRenderPass({
      label: "splat layer composite",
      colorAttachments: [{ ...attachment, loadOp: "load", storeOp: "store" }],
    });
    if (viewport) {
      pass.setViewport(viewport.x, viewport.y, viewport.z, viewport.w, 0, 1);
      pass.setScissorRect(viewport.x, viewport.y, viewport.z, viewport.w);
    }
    pass.setPipeline(pipeline);
    pass.setBindGroup(
      0,
      this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: copy.createView() },
          { binding: 1, resource: layer.createView() },
        ],
      }),
    );
    pass.draw(3);
    pass.end();
  }

  private ensureScratch(color: GPUTexture): Scratch {
    const s = this.scratch;
    if (
      s &&
      s.copy.width === color.width &&
      s.copy.height === color.height &&
      s.copy.format === color.format &&
      s.copy.sampleCount === color.sampleCount
    ) {
      return s;
    }
    s?.layer.destroy();
    s?.copy.destroy();
    const size = [color.width, color.height];
    const sampleCount = color.sampleCount;
    this.scratch = {
      layer: this.device.createTexture({
        label: "splat layer",
        size,
        format: SRGB_LAYER_FORMAT,
        sampleCount,
        usage:
          GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      }),
      copy: this.device.createTexture({
        label: "splat layer background",
        size,
        format: color.format,
        sampleCount,
        // Multisampled textures must be attachments.
        usage:
          GPUTextureUsage.COPY_DST |
          GPUTextureUsage.TEXTURE_BINDING |
          (sampleCount > 1 ? GPUTextureUsage.RENDER_ATTACHMENT : 0),
      }),
    };
    return this.scratch;
  }

  private pipeline(format: GPUTextureFormat, sampleCount: number) {
    const key = `${format}/${sampleCount}`;
    let pipeline = this.pipelines.get(key);
    if (!pipeline) {
      const module = this.device.createShaderModule({
        code: COMPOSITE_WGSL(sampleCount > 1),
      });
      pipeline = this.device.createRenderPipeline({
        label: "splat layer composite",
        layout: "auto",
        vertex: { module, entryPoint: "vertexMain" },
        fragment: { module, entryPoint: "fragmentMain", targets: [{ format }] },
        multisample: { count: sampleCount },
      });
      this.pipelines.set(key, pipeline);
    }
    return pipeline;
  }

  dispose() {
    this.scratch?.layer.destroy();
    this.scratch?.copy.destroy();
    this.scratch = null;
  }
}
