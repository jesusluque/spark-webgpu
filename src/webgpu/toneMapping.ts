// three's tone mapping operators (r180, nodes/display/ToneMappingFunctions)
// in WGSL, each with a right inverse: toneMap(inverseToneMap(y)) = y for a
// display colour y the operator can produce. SrgbComposite uses them to
// blend splats over three's tone-mapped picture in display space, as WebGL
// Spark does, and write the result back into three's linear HDR target as
// the value its output pass tone-maps to that colour.
//
// Linear, Reinhard, Cineon and ACES Filmic invert in closed form per
// channel (between ACES's colour matrices); AgX's contrast polynomial,
// monotonic on [0, 1], by bisection; Neutral by undoing its peak
// compression, desaturation and toe in turn. Custom tone mapping, a node of
// the app's, has no inverse here.

import * as THREE from "three";

type Mat3 = readonly number[];

// Column-major, as WGSL's mat3x3f takes them. TSL's mat3 of vec3s takes
// columns too, but of nine numbers rows: ACES's, transposed here.
const ACES_INPUT: Mat3 = transpose([
  0.59719, 0.35458, 0.04823, 0.076, 0.90834, 0.01566, 0.0284, 0.13383, 0.83777,
]);
const ACES_OUTPUT: Mat3 = transpose([
  1.60475, -0.53108, -0.07367, -0.10208, 1.10813, -0.00605, -0.00327, -0.07276,
  1.07602,
]);
const REC2020_TO_SRGB: Mat3 = [
  1.6605, -0.1246, -0.0182, -0.5876, 1.1329, -0.1006, -0.0728, -0.0083, 1.1187,
];
const SRGB_TO_REC2020: Mat3 = [
  0.6274, 0.0691, 0.0164, 0.3293, 0.9195, 0.088, 0.0433, 0.0113, 0.8956,
];
const AGX_INSET: Mat3 = [
  0.856627153315983, 0.137318972929847, 0.11189821299995, 0.0951212405381588,
  0.761241990602591, 0.0767994186031903, 0.0482516061458583, 0.101439036467562,
  0.811302368396859,
];
const AGX_OUTSET: Mat3 = [
  1.1271005818144368, -0.1413297634984383, -0.14132976349843826,
  -0.11060664309660323, 1.157823702216272, -0.11060664309660294,
  -0.016493938717834573, -0.016493938717834257, 1.2519364065950405,
];

function transpose(m: Mat3): number[] {
  return [0, 3, 6, 1, 4, 7, 2, 5, 8].map((i) => m[i]);
}

// Inverse of a 3x3 matrix (any consistent layout).
function invert(m: Mat3): number[] {
  const [a, b, c, d, e, f, g, h, i] = m;
  const A = e * i - f * h;
  const B = -(d * i - f * g);
  const C = d * h - e * g;
  const det = a * A + b * B + c * C;
  return [
    A / det,
    -(b * i - c * h) / det,
    (b * f - c * e) / det,
    B / det,
    (a * i - c * g) / det,
    -(a * f - c * d) / det,
    C / det,
    -(a * h - b * g) / det,
    (a * e - b * d) / det,
  ];
}

const mat = (m: Mat3) =>
  `mat3x3f(${m.map((v) => v.toPrecision(17)).join(", ")})`;

/** The operators with an inverse here (THREE.*ToneMapping). */
export const INVERTIBLE_TONE_MAPPINGS: readonly number[] = [
  THREE.NoToneMapping,
  THREE.LinearToneMapping,
  THREE.ReinhardToneMapping,
  THREE.CineonToneMapping,
  THREE.ACESFilmicToneMapping,
  THREE.AgXToneMapping,
  THREE.NeutralToneMapping,
];

// Largest value written back: half floats (three's default frame buffer
// type) end at 65504. Display values the operator only reaches at infinity
// come back as this.
const MAX_HDR = 65000;

/**
 * WGSL for `toneMap(c, exposure)` and `inverseToneMap(y, exposure)` of one
 * of INVERTIBLE_TONE_MAPPINGS.
 */
export function toneMappingWgsl(toneMapping: number): string {
  const body = OPERATORS[toneMapping];
  if (body === undefined) {
    throw new Error(`No inverse for tone mapping ${toneMapping}`);
  }
  return /* wgsl */ `
const MAX_HDR = ${MAX_HDR}.0;
${body}
`;
}

const OPERATORS: Record<number, string> = {
  [THREE.NoToneMapping]: /* wgsl */ `
fn toneMap(c: vec3f, exposure: f32) -> vec3f { return c; }
fn inverseToneMap(y: vec3f, exposure: f32) -> vec3f { return y; }
`,
  [THREE.LinearToneMapping]: /* wgsl */ `
fn toneMap(c: vec3f, exposure: f32) -> vec3f { return saturate(c * exposure); }
fn inverseToneMap(y: vec3f, exposure: f32) -> vec3f {
  return min(saturate(y) / exposure, vec3f(MAX_HDR));
}
`,
  [THREE.ReinhardToneMapping]: /* wgsl */ `
fn toneMap(c: vec3f, exposure: f32) -> vec3f {
  let e = c * exposure;
  return saturate(e / (e + 1.0));
}
fn inverseToneMap(y: vec3f, exposure: f32) -> vec3f {
  let v = saturate(y);
  return min(v / max(1.0 - v, vec3f(1e-6)) / exposure, vec3f(MAX_HDR));
}
`,
  [THREE.CineonToneMapping]: /* wgsl */ `
fn toneMap(c: vec3f, exposure: f32) -> vec3f {
  let e = max(c * exposure - 0.004, vec3f(0.0));
  let a = e * (e * 6.2 + 0.5);
  let b = e * (e * 6.2 + 1.7) + 0.06;
  return pow(a / b, vec3f(2.2));
}
// a / b = r is 6.2 (r - 1) e^2 + (1.7 r - 0.5) e + 0.06 r = 0: its root >= 0.
fn inverseToneMap(y: vec3f, exposure: f32) -> vec3f {
  let r = min(pow(saturate(y), vec3f(1.0 / 2.2)), vec3f(0.999999));
  let qa = 6.2 * (r - 1.0);
  let qb = 1.7 * r - 0.5;
  let qc = 0.06 * r;
  let e = (-qb - sqrt(max(qb * qb - 4.0 * qa * qc, vec3f(0.0)))) / (2.0 * qa);
  let c = select(e + 0.004, vec3f(0.0), e <= vec3f(0.0));
  return min(c / exposure, vec3f(MAX_HDR));
}
`,
  [THREE.ACESFilmicToneMapping]: /* wgsl */ `
const ACES_INPUT = ${mat(ACES_INPUT)};
const ACES_OUTPUT = ${mat(ACES_OUTPUT)};
const ACES_INPUT_INV = ${mat(invert(ACES_INPUT))};
const ACES_OUTPUT_INV = ${mat(invert(ACES_OUTPUT))};
fn rrtAndOdtFit(v: vec3f) -> vec3f {
  let a = v * (v + 0.0245786) - 0.000090537;
  let b = v * (v + 0.4329510) * 0.983729 + 0.238081;
  return a / b;
}
fn toneMap(c: vec3f, exposure: f32) -> vec3f {
  return saturate(ACES_OUTPUT * rrtAndOdtFit(ACES_INPUT * (c * exposure / 0.6)));
}
// rrtAndOdtFit(u) = v is a quadratic in u: its larger root. The fit tends
// to 1 / 0.983729 at infinity.
fn inverseToneMap(y: vec3f, exposure: f32) -> vec3f {
  let v = clamp(ACES_OUTPUT_INV * saturate(y), vec3f(-0.00038), vec3f(1.0165));
  let qa = 1.0 - 0.983729 * v;
  let qb = 0.0245786 - 0.4329510 * 0.983729 * v;
  let qc = -(0.000090537 + 0.238081 * v);
  let u = (-qb + sqrt(max(qb * qb - 4.0 * qa * qc, vec3f(0.0)))) / (2.0 * qa);
  let c = ACES_INPUT_INV * u * 0.6 / exposure;
  return clamp(c, vec3f(0.0), vec3f(MAX_HDR));
}
`,
  [THREE.AgXToneMapping]: /* wgsl */ `
const AGX_SRGB_TO_REC2020 = ${mat(SRGB_TO_REC2020)};
const AGX_REC2020_TO_SRGB = ${mat(REC2020_TO_SRGB)};
const AGX_INSET = ${mat(AGX_INSET)};
const AGX_OUTSET = ${mat(AGX_OUTSET)};
const AGX_IN_INV = ${mat(invert(mul(AGX_INSET, SRGB_TO_REC2020)))};
const AGX_OUTSET_INV = ${mat(invert(AGX_OUTSET))};
const AGX_REC2020_TO_SRGB_INV = ${mat(invert(REC2020_TO_SRGB))};
const AGX_MIN_EV = -12.47393;
const AGX_MAX_EV = 4.026069;
fn agxContrast(x: vec3f) -> vec3f {
  let x2 = x * x;
  let x4 = x2 * x2;
  return 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232;
}
fn toneMap(c: vec3f, exposure: f32) -> vec3f {
  var t = AGX_INSET * (AGX_SRGB_TO_REC2020 * (c * exposure));
  t = log2(max(t, vec3f(1e-10)));
  t = saturate((t - AGX_MIN_EV) / (AGX_MAX_EV - AGX_MIN_EV));
  t = AGX_OUTSET * agxContrast(t);
  t = pow(max(vec3f(0.0), t), vec3f(2.2));
  return saturate(AGX_REC2020_TO_SRGB * t);
}
// agxContrast is increasing on [0, 1]: bisection.
fn inverseToneMap(y: vec3f, exposure: f32) -> vec3f {
  let p = pow(max(AGX_REC2020_TO_SRGB_INV * saturate(y), vec3f(0.0)), vec3f(1.0 / 2.2));
  let q = clamp(AGX_OUTSET_INV * p, vec3f(-0.00232), vec3f(0.99858));
  var lo = vec3f(0.0);
  var hi = vec3f(1.0);
  for (var i = 0; i < 24; i++) {
    let mid = 0.5 * (lo + hi);
    let below = agxContrast(mid) < q;
    lo = select(lo, mid, below);
    hi = select(mid, hi, below);
  }
  let t = exp2(0.5 * (lo + hi) * (AGX_MAX_EV - AGX_MIN_EV) + AGX_MIN_EV);
  return clamp(AGX_IN_INV * t / exposure, vec3f(0.0), vec3f(MAX_HDR));
}
`,
  [THREE.NeutralToneMapping]: /* wgsl */ `
const NEUTRAL_START = 0.76;
const NEUTRAL_DESATURATION = 0.15;
fn toneMap(c: vec3f, exposure: f32) -> vec3f {
  var color = c * exposure;
  let x = min(color.r, min(color.g, color.b));
  color -= select(0.04, x - 6.25 * x * x, x < 0.08);
  let peak = max(color.r, max(color.g, color.b));
  if (peak < NEUTRAL_START) {
    return color;
  }
  let d = 1.0 - NEUTRAL_START;
  let newPeak = 1.0 - d * d / (peak + d - NEUTRAL_START);
  color *= newPeak / peak;
  let g = 1.0 - 1.0 / (NEUTRAL_DESATURATION * (peak - newPeak) + 1.0);
  return mix(color, vec3f(newPeak), g);
}
// The desaturation keeps the peak, which gives the compression's input,
// then the toe: x - 6.25 x^2 off the minimum below 0.08, else 0.04.
fn inverseToneMap(y: vec3f, exposure: f32) -> vec3f {
  var color = saturate(y);
  let newPeak = max(color.r, max(color.g, color.b));
  if (newPeak >= NEUTRAL_START) {
    let d = 1.0 - NEUTRAL_START;
    let peak = d * d / max(1.0 - newPeak, 1e-6) - d + NEUTRAL_START;
    let g = 1.0 - 1.0 / (NEUTRAL_DESATURATION * (peak - newPeak) + 1.0);
    color = (color - g * newPeak) / (1.0 - g) * (peak / newPeak);
  }
  let m = min(color.r, min(color.g, color.b));
  let x = select(m + 0.04, sqrt(max(m, 0.0) / 6.25), m < 0.04);
  return clamp((color + (x - m)) / exposure, vec3f(0.0), vec3f(MAX_HDR));
}
`,
};

// a * b of column-major 3x3 matrices.
function mul(a: Mat3, b: Mat3): number[] {
  const out: number[] = [];
  for (let col = 0; col < 3; col++) {
    for (let row = 0; row < 3; row++) {
      let s = 0;
      for (let k = 0; k < 3; k++) s += a[3 * k + row] * b[3 * col + k];
      out.push(s);
    }
  }
  return out;
}
