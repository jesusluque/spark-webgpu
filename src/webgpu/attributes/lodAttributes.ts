// Attributes through the WASM LOD builders (rust/spark-lib/src/attrib.rs).
// The WASM decoders read a PLY's extra properties and a .rad's attribute
// chunks themselves, and results carry them (attribsFromResult). With a WASM
// build that does not, the PLY attributes PlyAttributeReader found are handed
// over before the build, which merges and reorders them with the splats, and
// read back for the LOD splats.

import {
  AttribPool,
  type AttributeColumn,
  type AttributeSpec,
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
  // Already decoded with the splats (spark-lib ply.rs): merged in any case.
  if (target.get_attribs?.().length) return null;
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

/**
 * Attributes a WASM splat result carries (decoded from a .rad, or merged by
 * a LOD build): attribSpecs and one Float64Array per attribute.
 */
export function attribsFromResult(result: {
  numSplats: number;
  attribSpecs?: AttributeSpec[];
  attribColumns?: Float64Array[];
}): AttribPool | undefined {
  const { attribSpecs: specs, attribColumns: columns } = result;
  if (!specs?.length || !columns) return undefined;
  const pool = new AttribPool(result.numSplats);
  specs.forEach(({ name, format, components, lodMerge }, k) => {
    // The file keeps no flags: a renormalized 3-vector is a direction. An
    // all-zero one is a 3DGS trainer's placeholder normal.
    const direction = lodMerge === "normalizeMean" && components === 3;
    if (direction && columns[k].every((v) => v === 0)) return;
    pool.setAttribute(name, columns[k], format, components, {
      lodMerge,
      direction,
      toDraw: true,
    });
  });
  return pool.columns.length ? pool : undefined;
}
