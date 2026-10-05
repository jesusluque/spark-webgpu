// The 16-byte packed splat (PackedSplats): one uint4 per splat.
//   word0  rgba8
//   word1  center.xy as half
//   word2  center.z half | quat byte 0 | quat byte 1
//   word3  3 x u8 log scale (0 = zero scale) | quat byte 2
// Ported from src/shaders/splatDefines.glsl. `encoding` is
// (rgbMin, rgbMax, lnScaleMin, lnScaleMax), the PackedSplats encoding range.

module packed;

import core.math;
import core.quat_codec;
import core.splat_types;

public static const float4 DEFAULT_ENCODING = float4(0.0, 1.0, LN_SCALE_MIN, LN_SCALE_MAX);

uint4 rgbaBytes(float4 rgba, float4 encoding) {
    float3 encRgb = (rgba.rgb - encoding.x) / (encoding.y - encoding.x);
    return uint4(round(clamp(float4(encRgb, rgba.a) * 255.0, 0.0, 255.0)));
}

float4 bytesRgba(uint word0, float4 encoding) {
    float4 rgba = float4(word0 & 0xffu, (word0 >> 8u) & 0xffu, (word0 >> 16u) & 0xffu, word0 >> 24u) / 255.0;
    rgba.rgb = rgba.rgb * (encoding.y - encoding.x) + encoding.x;
    return rgba;
}

uint encodeLnScale(float scale, float lnScaleMin, float lnScaleScale) {
    return (scale == 0.0) ? 0u : uint(round(clamp((log(scale) - lnScaleMin) * lnScaleScale, 0.0, 254.0))) + 1u;
}

float decodeLnScale(uint u, float lnScaleMin, float lnScaleStep) {
    return (u == 0u) ? 0.0 : exp(lnScaleMin + float(u - 1u) * lnScaleStep);
}

public uint4 packSplat(float3 center, float3 scales, float4 quaternion, float4 rgba, float4 encoding = DEFAULT_ENCODING) {
    uint4 uRgba = rgbaBytes(rgba, encoding);
    uint uQuat = encodeQuatOctXy88R8(quaternion);
    float lnScaleScale = 254.0 / (encoding.w - encoding.z);
    uint3 uScales = uint3(
        encodeLnScale(scales.x, encoding.z, lnScaleScale),
        encodeLnScale(scales.y, encoding.z, lnScaleScale),
        encodeLnScale(scales.z, encoding.z, lnScaleScale));

    uint word0 = uRgba.r | (uRgba.g << 8u) | (uRgba.b << 16u) | (uRgba.a << 24u);
    uint word1 = packHalf2x16(center.xy);
    uint word2 = packHalf2x16(float2(center.z, 0.0)) | ((uQuat & 0xffffu) << 16u);
    uint word3 = uScales.x | (uScales.y << 8u) | (uScales.z << 16u) | ((uQuat >> 16u) << 24u);
    return uint4(word0, word1, word2, word3);
}

public void unpackSplat(uint4 packed, out float3 center, out float3 scales, out float4 quaternion, out float4 rgba,
                        float4 encoding = DEFAULT_ENCODING) {
    rgba = bytesRgba(packed.x, encoding);
    center = float3(unpackHalf2x16(packed.y), unpackHalf2x16(packed.z & 0xffffu).x);
    float lnScaleStep = (encoding.w - encoding.z) / 254.0;
    scales = float3(
        decodeLnScale(packed.w & 0xffu, encoding.z, lnScaleStep),
        decodeLnScale((packed.w >> 8u) & 0xffu, encoding.z, lnScaleStep),
        decodeLnScale((packed.w >> 16u) & 0xffu, encoding.z, lnScaleStep));
    uint uQuat = ((packed.z >> 16u) & 0xffffu) | ((packed.w >> 8u) & 0xff0000u);
    quaternion = decodeQuatOctXy88R8(uQuat);
}

public Gsplat readPackedSplat(uint4 packed, int index, float4 encoding = DEFAULT_ENCODING) {
    Gsplat s;
    unpackSplat(packed, s.center, s.scales, s.quaternion, s.rgba, encoding);
    s.index = index;
    s.flags = all(s.scales == float3(0.0)) ? 0u : GSPLAT_FLAG_ACTIVE;
    return s;
}

public uint4 packSplatCov(float3 center, float4 rgba, float3 xxyyzz, float3 xyxzyz, float4 encoding = DEFAULT_ENCODING) {
    uint4 uRgba = rgbaBytes(rgba, encoding);
    float diagScale = 255.0 / (2.0 * (encoding.w - encoding.z));
    uint3 uDiag = uint3(round(clamp((log(xxyyzz) - 2.0 * encoding.z) * diagScale, 0.0, 255.0)));
    float3 cor = float3(
        clamp(xyxzyz.x / sqrt(xxyyzz.x * xxyyzz.y), -1.0, 1.0),
        clamp(xyxzyz.y / sqrt(xxyyzz.x * xxyyzz.z), -1.0, 1.0),
        clamp(xyxzyz.z / sqrt(xxyyzz.y * xxyyzz.z), -1.0, 1.0));
    int3 iCor = int3(round(cor * 127.0));

    uint word0 = uRgba.r | (uRgba.g << 8u) | (uRgba.b << 16u) | (uRgba.a << 24u);
    uint word1 = packHalf2x16(center.xy);
    uint word2 = packHalf2x16(float2(center.z, 0.0)) | ((uint(iCor.y) & 0xffu) << 16u) | ((uint(iCor.z) & 0xffu) << 24u);
    uint word3 = uDiag.x | (uDiag.y << 8u) | (uDiag.z << 16u) | ((uint(iCor.x) & 0xffu) << 24u);
    return uint4(word0, word1, word2, word3);
}

public void unpackSplatCov(uint4 packed, out float3 center, out float4 rgba, out float3 xxyyzz, out float3 xyxzyz,
                           float4 encoding = DEFAULT_ENCODING) {
    rgba = bytesRgba(packed.x, encoding);
    center = float3(unpackHalf2x16(packed.y), unpackHalf2x16(packed.z & 0xffffu).x);
    uint3 uDiag = uint3(packed.w & 0xffu, (packed.w >> 8u) & 0xffu, (packed.w >> 16u) & 0xffu);
    int3 iCor = int3(int(packed.w) >> 24, int(packed.z << 8u) >> 24, int(packed.z) >> 24);
    float diagScale = 2.0 * (encoding.w - encoding.z) / 255.0;
    xxyyzz = exp(2.0 * encoding.z + float3(uDiag) * diagScale);
    xyxzyz = float3(iCor) / 127.0 * float3(
        sqrt(xxyyzz.x * xxyyzz.y),
        sqrt(xxyyzz.x * xxyyzz.z),
        sqrt(xxyyzz.y * xxyyzz.z));
}
