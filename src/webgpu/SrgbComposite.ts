// Splats blended in sRGB space over a linear target, as WebGL Spark blends
// them on the canvas. Hardware blending works on the stored (linear) values,
// so the splats first accumulate in a layer of their own, cleared to (0, 0,
// 0, 1): premultiplied sRGB colour L and, in alpha, the transmittance T left
// by all of them. Blending back to front over a background B gives
// L + T * B in sRGB, so a full-screen pass composites
//   linear(L + T * srgb(dst))
// over a copy of the target. srgb() and linear() are three's own transfer
// functions; pixels no splat covers are left as they were.
//
// With tone mapping, three tone-maps the target in its output pass, after
// everything is blended, where WebGL Spark blends untone-mapped splats over
// the tone-mapped picture. So the layer is composited over the picture as
// it will show, srgb(T(dst)), and written back as T's inverse
// (toneMapping.ts), which the output pass maps to the composited colour.
// Transparent objects drawn after the splats then still blend over them,
// in three's linear space. Colours T never produces (AgX and ACES desaturate
// bright colours, and AgX's white is 0.997) come out as the nearest it
// does.

import * as THREE from "three";
import { reuseTexture } from "./gpuBuffers";
import { FULLSCREEN_TRIANGLE_WGSL } from "./renderPipeline";
import { toneMappingWgsl } from "./toneMapping";

const COMPOSITE_WGSL = (
  multisampled: boolean,
  toneMapping: number,
) => /* wgsl */ `
@group(0) @binding(0) var dstTex: ${multisampled ? "texture_multisampled_2d" : "texture_2d"}<f32>;
@group(0) @binding(1) var layerTex: ${multisampled ? "texture_multisampled_2d" : "texture_2d"}<f32>;
// x: three's toneMappingExposure.
@group(0) @binding(2) var<uniform> params: vec4f;
${toneMappingWgsl(toneMapping)}
${FULLSCREEN_TRIANGLE_WGSL}
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
  if (t == 1.0 && all(layer.rgb == vec3f(0.0))) {
    return dst;
  }
  let exposure = params.x;
  let shown = toneMap(max(dst.rgb, vec3f(0.0)), exposure);
  let rgb = layer.rgb + t * toSrgb(shown);
  return vec4f(inverseToneMap(toLinear(rgb), exposure), 1.0 - t + t * dst.a);
}
`;

/**
 * The splat layer's format: 8-bit, so that each blend rounds as on WebGL's
 * 8-bit canvas (dense scenes drift a few levels from half floats).
 */
export const SRGB_LAYER_FORMAT: GPUTextureFormat = "rgba8unorm";

/** Blend state for the layer: colour over, alpha the transmittance. */
export const SRGB_LAYER_BLEND: GPUBlendState = {
  color: { srcFactor: "one", dstFactor: "one-minus-src-alpha" },
  alpha: { srcFactor: "zero", dstFactor: "one-minus-src-alpha" },
};

interface Scratch {
  layer: GPUTexture;
  copy: GPUTexture;
}

/** three's tone mapping, as its output pass will apply it. */
export interface CompositeToneMapping {
  /** A THREE.*ToneMapping in INVERTIBLE_TONE_MAPPINGS. */
  toneMapping: number;
  exposure: number;
}

export class SrgbComposite {
  private pipelines = new Map<string, GPURenderPipeline>();
  private scratch: Scratch | null = null;
  private readonly params: GPUBuffer;

  constructor(readonly device: GPUDevice) {
    this.params = device.createBuffer({
      label: "splat layer composite params",
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

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
   * which may resolve), within `viewport` when given, under three's
   * `toneMapping` when its output pass tone-maps `color`.
   */
  composite(
    encoder: GPUCommandEncoder,
    color: GPUTexture,
    attachment: Pick<GPURenderPassColorAttachment, "view" | "resolveTarget">,
    viewport?: { x: number; y: number; z: number; w: number } | null,
    toneMapping: CompositeToneMapping = {
      toneMapping: THREE.NoToneMapping,
      exposure: 1,
    },
  ) {
    const { layer, copy } = this.ensureScratch(color);
    encoder.copyTextureToTexture({ texture: color }, { texture: copy }, [
      color.width,
      color.height,
    ]);
    const pipeline = this.pipeline(
      color.format,
      color.sampleCount,
      toneMapping.toneMapping,
    );
    this.device.queue.writeBuffer(
      this.params,
      0,
      new Float32Array([toneMapping.exposure, 0, 0, 0]),
    );
    const pass = encoder.beginRenderPass({
      label: "splat layer composite",
      colorAttachments: [{ ...attachment, loadOp: "load", storeOp: "store" }],
    });
    if (viewport) {
      pass.setViewport(viewport.x, viewport.y, viewport.z, viewport.w, 0, 1);
      // Clamped: the viewport may reach past the target, a scissor can't.
      const x0 = Math.max(0, Math.floor(viewport.x));
      const y0 = Math.max(0, Math.floor(viewport.y));
      const x1 = Math.min(color.width, Math.ceil(viewport.x + viewport.z));
      const y1 = Math.min(color.height, Math.ceil(viewport.y + viewport.w));
      pass.setScissorRect(x0, y0, Math.max(0, x1 - x0), Math.max(0, y1 - y0));
    }
    pass.setPipeline(pipeline);
    pass.setBindGroup(
      0,
      this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: copy.createView() },
          { binding: 1, resource: layer.createView() },
          { binding: 2, resource: { buffer: this.params } },
        ],
      }),
    );
    pass.draw(3);
    pass.end();
  }

  private ensureScratch(color: GPUTexture): Scratch {
    const size = [color.width, color.height];
    const sampleCount = color.sampleCount;
    this.scratch = {
      layer: reuseTexture(this.device, this.scratch?.layer, {
        label: "splat layer",
        size,
        format: SRGB_LAYER_FORMAT,
        sampleCount,
        usage:
          GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      }),
      copy: reuseTexture(this.device, this.scratch?.copy, {
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

  private pipeline(
    format: GPUTextureFormat,
    sampleCount: number,
    toneMapping: number,
  ) {
    const key = `${format}/${sampleCount}/${toneMapping}`;
    let pipeline = this.pipelines.get(key);
    if (!pipeline) {
      const module = this.device.createShaderModule({
        code: COMPOSITE_WGSL(sampleCount > 1, toneMapping),
      });
      pipeline = this.device.createRenderPipeline({
        label: "splat layer composite",
        layout: "auto",
        vertex: { module, entryPoint: "fullscreenVertex" },
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
    this.params.destroy();
  }
}
