// Attributes through the WASM LOD builders (rust/spark-lib/src/attrib.rs):
// handed over as decoded values before the build, which merges and reorders
// them with the splats, and read back into an AttribPool for the LOD splats.

import {
  AttribPool,
  type AttributeColumn,
  columnBits,
  decodeComponent,
} from "./schema";

/** The WASM GsplatArray/CsplatArray methods used here (older builds lack them). */
export interface LodAttribTarget {
  len(): number;
  set_attribs?(specs: unknown, columns: unknown[]): void;
  get_attribs?(): Float64Array[];
}

/** A column's decoded values, components per splat. */
export function columnValues(
  col: AttributeColumn,
  count: number,
): Float64Array {
  const n = col.spec.components;
  const out = new Float64Array(count * n);
  for (let i = 0; i < count; i++) {
    for (let c = 0; c < n; c++) {
      out[i * n + c] = decodeComponent(col.spec.format, columnBits(col, i, c));
    }
  }
  return out;
}

/** Gives `attribs` to the LOD builder; returns them if it took them. */
export function setLodAttribs(
  target: LodAttribTarget,
  attribs: AttribPool | null,
): AttribPool | null {
  if (!attribs || !target.set_attribs || attribs.count !== target.len()) {
    return null;
  }
  target.set_attribs(
    attribs.schema,
    attribs.columns.map((col) => columnValues(col, attribs.count)),
  );
  return attribs;
}

/** The merged attributes for the built LOD splats, as extra.attribs. */
export function withLodAttribs<
  R extends { numSplats: number; extra: Record<string, unknown> },
>(result: R, target: LodAttribTarget, attribs: AttribPool | null): R {
  const columns = attribs && target.get_attribs?.();
  if (!attribs || !columns) return result;
  const pool = new AttribPool(result.numSplats);
  attribs.schema.forEach(({ name, format, components, ...options }, k) => {
    pool.setAttribute(name, columns[k], format, components, options);
  });
  result.extra.attribs = pool;
  return result;
}
