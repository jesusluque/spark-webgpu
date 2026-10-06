// The extra attributes of a paged .rad, resident on the GPU like
// WgpuSplatPager's core and SH pools: one interleaved pool (the layout
// slang/core/attrib.slang reads) over all pages, page p's chunk written at
// splats [p * 65536, ...). The CPU AttribPool it exposes holds the schema
// only, with gpuBuffer set, so SplatAttributes binds the pool as it is.

import { columnValues } from "./lodAttributes";
import {
  AttribPool,
  type AttributeSpec,
  type LodMerge,
  type PoolLayout,
  attribWords,
  packColumn,
  poolHeader,
  poolLayout,
  poolWords,
} from "./schema";

/** An attribute as RadMeta.attributes lists it. */
export interface RadAttributeMeta {
  name: string;
  format: string;
  components: number;
  lodMerge?: string;
}

/**
 * Specs from a .rad's attribute list, flagged as attribsFromResult flags a
 * decoded file's: drawable, and renormalized 3-vectors turn with the object.
 */
export function specsFromRadMeta(
  attributes: readonly RadAttributeMeta[],
): AttributeSpec[] {
  return attributes.map(({ name, format, components, lodMerge }) => ({
    name,
    format: format as AttributeSpec["format"],
    components,
    lodMerge: lodMerge as LodMerge | undefined,
    direction: lodMerge === "normalizeMean" && components === 3,
    toDraw: true,
  }));
}

export class PagedAttribPool {
  readonly layout: PoolLayout;
  /** The schema for renderers; its data lives in `buffer`. */
  readonly pool: AttribPool;
  readonly buffer: GPUBuffer;

  constructor(
    readonly device: GPUDevice,
    specs: AttributeSpec[],
    readonly maxSplats: number,
  ) {
    this.layout = poolLayout(specs);
    this.buffer = device.createBuffer({
      label: "attribute pages",
      size: Math.ceil((poolWords(this.layout, maxSplats) * 4) / 16) * 16,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
    });
    const header = poolHeader(this.layout, maxSplats);
    device.queue.writeBuffer(this.buffer, 0, header);
    this.pool = new AttribPool(maxSplats);
    this.pool.columns = specs.map((spec) => ({
      spec,
      words: new Uint32Array(0),
    }));
    this.pool.gpuBuffer = this.buffer;
  }

  /** Bytes the pool needs for `specs` over `maxSplats` splats. */
  static bytes(specs: AttributeSpec[], maxSplats: number) {
    return poolWords(poolLayout(specs), maxSplats) * 4;
  }

  /**
   * Writes `count` splats from `base` (a page) from a chunk's attributes,
   * matched by name and converted to the pool's formats; attributes the
   * chunk lacks are zero.
   */
  uploadPage(base: number, count: number, chunk: AttribPool | null) {
    const { layout } = this;
    const stride = layout.strideWords;
    if (stride === 0) return;
    const rows = new Uint32Array(count * stride);
    layout.specs.forEach((spec, k) => {
      const col = chunk?.column(spec.name);
      if (!col || col.spec.components !== spec.components) return;
      const words =
        col.spec.format === spec.format
          ? col.words
          : packColumn(spec, columnValues(col, count), count);
      const per = attribWords(spec);
      for (let i = 0; i < count; i++) {
        rows.set(
          words.subarray(i * per, (i + 1) * per),
          i * stride + layout.offsets[k],
        );
      }
    });
    this.device.queue.writeBuffer(
      this.buffer,
      (layout.headerWords + base * stride) * 4,
      rows,
    );
    this.pool.version += 1;
  }

  destroy() {
    this.buffer.destroy();
  }
}
