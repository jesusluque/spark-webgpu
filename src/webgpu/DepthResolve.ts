// A single-sample copy of a multisampled depth texture (its sample 0).
//
// With antialias: true three renders the scene multisampled. Splats drawn on
// the single-sample canvas after three's output pass (SparkWebGPU) can't
// attach that depth, and WebGPU can't resolve depth textures, so a
// full-screen pass copies it into a depth32float texture they test against.

const SHADER = /* wgsl */ `
@group(0) @binding(0) var depthMs: texture_depth_multisampled_2d;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) pos: vec4f) -> @builtin(frag_depth) f32 {
  return textureLoad(depthMs, vec2i(pos.xy), 0);
}
`;

export const RESOLVED_DEPTH_FORMAT: GPUTextureFormat = "depth32float";

export class DepthResolve {
  private pipeline?: GPURenderPipeline;
  private texture?: GPUTexture;

  constructor(readonly device: GPUDevice) {}

  /** Records the copy of `source` into `encoder`; returns the copy. */
  resolve(encoder: GPUCommandEncoder, source: GPUTexture): GPUTexture {
    const { device } = this;
    if (!this.pipeline) {
      const module = device.createShaderModule({
        label: "depth resolve",
        code: SHADER,
      });
      this.pipeline = device.createRenderPipeline({
        label: "depth resolve",
        layout: "auto",
        vertex: { module, entryPoint: "vs" },
        fragment: { module, entryPoint: "fs", targets: [] },
        depthStencil: {
          format: RESOLVED_DEPTH_FORMAT,
          depthWriteEnabled: true,
          depthCompare: "always",
        },
      });
    }
    let texture = this.texture;
    if (
      !texture ||
      texture.width !== source.width ||
      texture.height !== source.height
    ) {
      texture?.destroy();
      texture = device.createTexture({
        label: "resolved depth",
        size: [source.width, source.height],
        format: RESOLVED_DEPTH_FORMAT,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
      });
      this.texture = texture;
    }
    const bindGroup = device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: source.createView({ aspect: "depth-only" }),
        },
      ],
    });
    const pass = encoder.beginRenderPass({
      label: "depth resolve",
      colorAttachments: [],
      depthStencilAttachment: {
        view: texture.createView(),
        depthLoadOp: "clear",
        depthClearValue: 0,
        depthStoreOp: "store",
      },
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(3);
    pass.end();
    return texture;
  }

  dispose() {
    this.texture?.destroy();
    this.texture = undefined;
  }
}
