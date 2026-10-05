// Constants and math shared by the splat kernels.
// Ported from src/shaders/splatDefines.glsl.
//
// Matrices follow Slang's convention, mul(M, v) with M row-major. A GLSL
// mat3 built from columns c0, c1, c2 is the transpose of the same list here,
// so the constructors below list the GLSL columns as rows and transpose.

module math;

public static const float LN_SCALE_MIN = -12.0;
public static const float LN_SCALE_MAX = 9.0;
public static const uint F16_INF = 0x7c00u;
public static const float PI = 3.1415926535897932384626433832795;
public static const float INFINITY = 1.0 / 0.0;
public static const float NEG_INFINITY = -1.0 / 0.0;

public float sqr(float x) { return x * x; }
public float pow4(float x) { float x2 = x * x; return x2 * x2; }
public float pow8(float x) { float x4 = pow4(x); return x4 * x4; }

public float3 srgbToLinear(float3 rgb) { return pow(rgb, float3(2.2)); }
public float3 linearToSrgb(float3 rgb) { return pow(rgb, float3(1.0 / 2.2)); }

// packHalf2x16 is a Slang built-in with the GLSL meaning (x in the low 16
// bits); this is its inverse.
public float2 unpackHalf2x16(uint u) {
    return float2(f16tof32(u & 0xffffu), f16tof32(u >> 16u));
}

// Rotate vector v by quaternion q (x, y, z, w).
public float3 quatVec(float4 q, float3 v) {
    float3 t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

// Apply quaternion q1 after quaternion q2.
public float4 quatQuat(float4 q1, float4 q2) {
    return float4(
        q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y,
        q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x,
        q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w,
        q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z);
}

public float3x3 quaternionToMatrix(float4 q) {
    return scaleQuaternionToMatrix(float3(1.0), q);
}

// Scale by s, then rotate by q.
public float3x3 scaleQuaternionToMatrix(float3 s, float4 q) {
    float3x3 columns = float3x3(
        s.x * (1.0 - 2.0 * (q.y * q.y + q.z * q.z)),
        s.x * (2.0 * (q.x * q.y + q.w * q.z)),
        s.x * (2.0 * (q.x * q.z - q.w * q.y)),
        s.y * (2.0 * (q.x * q.y - q.w * q.z)),
        s.y * (1.0 - 2.0 * (q.x * q.x + q.z * q.z)),
        s.y * (2.0 * (q.y * q.z + q.w * q.x)),
        s.z * (2.0 * (q.x * q.z + q.w * q.y)),
        s.z * (2.0 * (q.y * q.z - q.w * q.x)),
        s.z * (1.0 - 2.0 * (q.x * q.x + q.y * q.y)));
    return transpose(columns);
}

// Spherical lerp between two quaternions.
public float4 slerp(float4 q1, float4 q2, float t) {
    float cosHalfTheta = dot(q1, q2);
    if (abs(cosHalfTheta) >= 0.999) {
        return q1;
    }
    if (cosHalfTheta < 0.0) {
        q2 = -q2;
        cosHalfTheta = -cosHalfTheta;
    }
    float halfTheta = acos(cosHalfTheta);
    float sinHalfTheta = sqrt(1.0 - cosHalfTheta * cosHalfTheta);
    float ratioA = sin((1.0 - t) * halfTheta) / sinHalfTheta;
    float ratioB = sin(t * halfTheta) / sinHalfTheta;
    return q1 * ratioA + q2 * ratioB;
}

public float4 uintToFloat4(uint u32) {
    return float4(u32 & 0xffu, (u32 >> 8u) & 0xffu, (u32 >> 16u) & 0xffu, u32 >> 24u) / 255.0;
}

public float3 debugColorHue(uint i) {
    float hue = frac(float(i) * 0.61803398875);
    float3 k = hue * 6.0 + float3(0.0, 4.0, 2.0);
    float3 rgb = saturate(abs(k - 6.0 * floor(k / 6.0) - 3.0) - 1.0);
    return lerp(float3(1.0), rgb, 0.85);
}
