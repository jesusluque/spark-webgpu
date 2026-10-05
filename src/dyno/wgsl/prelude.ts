// WGSL counterparts of what dyno's GLSL relies on: the splatDefines.glsl
// constants and helpers its ops and users call, and the Gsplat/CovSplat
// structs. Always in front of the generated code, as splatDefines is in GLSL.

import { defineCovSplat, defineGsplat, defineGsplatNormal } from "../splats";

export const WGSL_PRELUDE = /* wgsl */ `
const PI: f32 = 3.1415926535897932384626433832795;
const LN_SCALE_MIN: f32 = -12.0;
const LN_SCALE_MAX: f32 = 9.0;
const SPLAT_TEX_WIDTH_BITS: u32 = 11u;
const SPLAT_TEX_HEIGHT_BITS: u32 = 11u;
const SPLAT_TEX_LAYER_BITS: u32 = 22u;
const SPLAT_TEX_WIDTH_MASK: u32 = 2047u;
const SPLAT_TEX_HEIGHT_MASK: u32 = 2047u;
const GSPLAT_FLAG_ACTIVE: u32 = 1u;

struct Gsplat {
    center: vec3f,
    flags: u32,
    scales: vec3f,
    index: i32,
    quaternion: vec4f,
    rgba: vec4f,
}

struct CovSplat {
    center: vec3f,
    flags: u32,
    rgba: vec4f,
    xxyyzz: vec3f,
    index: i32,
    xyxzyz: vec3f,
}

// A runtime value, so it isn't a const-expression (where inf is an error).
fn dyno_inf() -> f32 {
    let bits = 0x7f800000u;
    return bitcast<f32>(bits);
}

fn isGsplatActive(flags: u32) -> bool {
    return (flags & GSPLAT_FLAG_ACTIVE) != 0u;
}

fn isCovSplatActive(flags: u32) -> bool {
    return (flags & GSPLAT_FLAG_ACTIVE) != 0u;
}

fn sqr(x: f32) -> f32 {
    return x * x;
}

fn pow4(x: f32) -> f32 {
    let x2 = x * x;
    return x2 * x2;
}

fn pow8(x: f32) -> f32 {
    let x4 = pow4(x);
    return x4 * x4;
}

fn srgbToLinear(rgb: vec3f) -> vec3f {
    return pow(rgb, vec3f(2.2));
}

fn linearToSrgb(rgb: vec3f) -> vec3f {
    return pow(rgb, vec3f(1.0 / 2.2));
}

fn quatVec(q: vec4f, v: vec3f) -> vec3f {
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

fn quatQuat(q1: vec4f, q2: vec4f) -> vec4f {
    return vec4f(
        q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y,
        q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x,
        q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w,
        q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z
    );
}

fn quaternionToMatrix(q: vec4f) -> mat3x3f {
    return mat3x3f(
        (1.0 - 2.0 * (q.y * q.y + q.z * q.z)),
        (2.0 * (q.x * q.y + q.w * q.z)),
        (2.0 * (q.x * q.z - q.w * q.y)),
        (2.0 * (q.x * q.y - q.w * q.z)),
        (1.0 - 2.0 * (q.x * q.x + q.z * q.z)),
        (2.0 * (q.y * q.z + q.w * q.x)),
        (2.0 * (q.x * q.z + q.w * q.y)),
        (2.0 * (q.y * q.z - q.w * q.x)),
        (1.0 - 2.0 * (q.x * q.x + q.y * q.y))
    );
}

fn scaleQuaternionToMatrix(s: vec3f, q: vec4f) -> mat3x3f {
    let m = quaternionToMatrix(q);
    return mat3x3f(s.x * m[0], s.y * m[1], s.z * m[2]);
}

fn slerp(q1: vec4f, q2_in: vec4f, t: f32) -> vec4f {
    var q2 = q2_in;
    var cosHalfTheta = dot(q1, q2);
    if (abs(cosHalfTheta) >= 0.999) {
        return q1;
    }
    if (cosHalfTheta < 0.0) {
        q2 = -q2;
        cosHalfTheta = -cosHalfTheta;
    }
    let halfTheta = acos(cosHalfTheta);
    let sinHalfTheta = sqrt(1.0 - cosHalfTheta * cosHalfTheta);
    let ratioA = sin((1.0 - t) * halfTheta) / sinHalfTheta;
    let ratioB = sin(t * halfTheta) / sinHalfTheta;
    return q1 * ratioA + q2 * ratioB;
}

fn splatTexCoord(index: i32) -> vec3i {
    let u = u32(index);
    let x = u & SPLAT_TEX_WIDTH_MASK;
    let y = (u >> SPLAT_TEX_WIDTH_BITS) & SPLAT_TEX_HEIGHT_MASK;
    let z = u >> SPLAT_TEX_LAYER_BITS;
    return vec3i(vec3u(x, y, z));
}

fn pagedSplatTexCoord(index: i32) -> vec3i {
    return vec3i(index & 255, (index >> 8u) & 255, index >> 16u);
}

fn uintToVec4(u: u32) -> vec4f {
    return unpack4x8unorm(u);
}

fn floatToVec4(f: f32) -> vec4f {
    return unpack4x8unorm(bitcast<u32>(f));
}

fn debugColorHue(i: u32) -> vec3f {
    let hue = fract(f32(i) * 0.61803398875);
    let h = hue * 6.0 + vec3f(0.0, 4.0, 2.0);
    let rgb = clamp(abs(h - 6.0 * floor(h / 6.0) - 3.0) - 1.0, vec3f(0.0), vec3f(1.0));
    return mix(vec3f(1.0), rgb, 0.85);
}
`;

const WGSL_GSPLAT_NORMAL = /* wgsl */ `
fn gsplatNormal(scales: vec3f, quaternion: vec4f) -> vec3f {
    let minScale = min(scales.x, min(scales.y, scales.z));
    var normal: vec3f;
    if (scales.z == minScale) {
        normal = vec3f(0.0, 0.0, 1.0);
    } else if (scales.y == minScale) {
        normal = vec3f(0.0, 1.0, 0.0);
    } else {
        normal = vec3f(1.0, 0.0, 0.0);
    }
    return quatVec(quaternion, normal);
}
`;

/**
 * WGSL for GLSL globals that dynos share by value (exported constants). An
 * empty string means the prelude already has it.
 */
export const wgslGlobals = new Map<string, string>([
  [defineGsplat, ""],
  [defineCovSplat, ""],
  [defineGsplatNormal, WGSL_GSPLAT_NORMAL],
]);

/** Gives a shared GLSL global its WGSL version, for dynos that list it. */
export function registerWgslGlobal(glsl: string, wgsl: string) {
  wgslGlobals.set(glsl, wgsl);
}
