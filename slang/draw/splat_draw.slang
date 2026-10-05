// Draws the accumulator's splats as instanced quads, back to front in the
// sorted order. Ported from src/shaders/splatVertex.glsl and
// splatFragment.glsl, for WebGPU clip space (depth in [0, 1]).
//
// One instance per ordering slot, four vertices as a triangle strip. Splats
// are read from the accumulator in its packed (1 x uint4) or ext (2 x uint4)
// layout, the one kernels/generate.slang writes.
//
// Matrices come as float4 columns: no layout rules to get wrong between
// Slang, WGSL and the host. mul(M, v) below means the GLSL M * v.

import core.math;
import core.splat_types;
import core.packed;
import core.ext;

static const uint DRAW_EXT = 1u;            // accumulator is ext (else packed)
static const uint DRAW_COV = 2u;            // covariance splats
static const uint DRAW_2DGS = 4u;           // flat splats when a scale is 0
static const uint DRAW_LOD_INFLATE = 8u;    // trade LOD opacity for size
static const uint DRAW_ORTHOGRAPHIC = 16u;
static const uint DRAW_ENCODE_LINEAR = 32u; // sRGB colours to linear output
static const uint DRAW_PREMULTIPLIED = 64u; // write premultiplied alpha

struct DrawParams {
    float4 proj0;   // projection matrix columns
    float4 proj1;
    float4 proj2;
    float4 proj3;
    float4 renderToViewQuat;
    float4 renderToViewPos;   // xyz
    float4 renderToViewBasis0; // columns, for covariance splats
    float4 renderToViewBasis1;
    float4 renderToViewBasis2;

    float renderWidth;
    float renderHeight;
    float maxStdDev;
    float minPixelRadius;

    float maxPixelRadius;
    float minAlpha;
    float blurAmount;
    float preBlurAmount;

    float focalDistance;
    float apertureAngle;
    float clipXY;
    float focalAdjustment;

    float falloff;
    uint flags;
    uint pad0;
    uint pad1;
};

StructuredBuffer<uint> ordering;
StructuredBuffer<uint4> splats;
ConstantBuffer<DrawParams> params;

bool hasFlag(uint f) { return (params.flags & f) != 0u; }

float4 project(float3 v) {
    return params.proj0 * v.x + params.proj1 * v.y + params.proj2 * v.z + params.proj3;
}

float3x3 fromColumns(float3 c0, float3 c1, float3 c2) {
    return transpose(float3x3(c0, c1, c2));
}

struct VertexOut {
    float4 position : SV_Position;
    float4 rgba : COLOR0;
    float2 splatUv : TEXCOORD0;
    nointerpolation float adjustedStdDev : TEXCOORD1;
    nointerpolation uint splatIndex : TEXCOORD2;
};

static const float2 CORNERS[4] = { float2(-1.0, -1.0), float2(1.0, -1.0), float2(-1.0, 1.0), float2(1.0, 1.0) };

[shader("vertex")]
VertexOut splatVertex(uint vertexId: SV_VertexID, uint instanceId: SV_InstanceID) {
    VertexOut o;
    // Outside the clip volume, so the quad is dropped if we return early.
    o.position = float4(0.0, 0.0, 2.0, 1.0);
    o.rgba = float4(0.0);
    o.splatUv = float2(0.0);
    o.adjustedStdDev = 0.0;
    o.splatIndex = 0u;

    uint splatIndex = ordering[instanceId];
    if (splatIndex == 0xffffffffu) {
        return o;
    }
    float2 corner = CORNERS[vertexId & 3u];

    float3 center, scales, xxyyzz, xyxzyz;
    float4 quaternion, rgba;
    bool3 zeroScales = bool3(false);
    bool cov = hasFlag(DRAW_COV);

    if (hasFlag(DRAW_EXT)) {
        uint4 a = splats[2u * splatIndex];
        float alpha = unpackSplatExtCenterAlpha(a).w;
        if (alpha == 0.0 || alpha < params.minAlpha) {
            return o;
        }
        uint4 b = splats[2u * splatIndex + 1u];
        if (!cov) {
            unpackSplatExt(a, b, center, scales, quaternion, rgba);
            zeroScales = scales == float3(0.0);
            if (all(zeroScales)) {
                return o;
            }
        } else {
            unpackSplatExtCov(a, b, center, rgba, xxyyzz, xyxzyz);
            if (all(xxyyzz == float3(0.0)) && all(xyxzyz == float3(0.0))) {
                return o;
            }
        }
    } else {
        uint4 p = splats[splatIndex];
        if (!cov) {
            unpackSplat(p, center, scales, quaternion, rgba);
            zeroScales = scales == float3(0.0);
            if (all(zeroScales)) {
                return o;
            }
        } else {
            unpackSplatCov(p, center, rgba, xxyyzz, xyxzyz);
            if (all(xxyyzz == float3(0.0)) && all(xyxzyz == float3(0.0))) {
                return o;
            }
        }
        // The packed accumulator stores alpha / 2.
        rgba.a *= 2.0;
        if (rgba.a == 0.0 || rgba.a < params.minAlpha) {
            return o;
        }
    }

    // Like the reference 3DGS rasterizer, clamp the SH colour to positive.
    rgba.rgb = max(rgba.rgb, float3(0.0));

    float adjustedStdDev = params.maxStdDev;
    if (rgba.a > 1.0) {
        // LOD opacity: stretch 1..2 to 1..5.
        rgba.a = min(rgba.a * 4.0 - 3.0, 5.0);
        if (hasFlag(DRAW_LOD_INFLATE)) {
            float opacity = exp((rgba.a * rgba.a - 1.0) / 2.718281828459045);
            scales *= pow(opacity, 1.0 / 3.0);
            rgba.a = 1.0;
        }
        adjustedStdDev = params.maxStdDev + 0.7 * (rgba.a - 1.0);
    }

    float3x3 basis = fromColumns(params.renderToViewBasis0.xyz, params.renderToViewBasis1.xyz, params.renderToViewBasis2.xyz);
    float3 viewCenter = (cov ? mul(basis, center) : quatVec(params.renderToViewQuat, center)) + params.renderToViewPos.xyz;
    if (viewCenter.z >= 0.0) {
        return o;
    }
    float4 clipCenter = project(viewCenter);
    if (clipCenter.z < 0.0 || clipCenter.z >= clipCenter.w) {
        return o;
    }
    float clip = params.clipXY * clipCenter.w;
    if (abs(clipCenter.x) > clip || abs(clipCenter.y) > clip) {
        return o;
    }

    o.rgba = rgba;
    o.splatUv = corner * adjustedStdDev;
    o.adjustedStdDev = adjustedStdDev;
    o.splatIndex = splatIndex;

    float3x3 cov3D;
    if (!cov) {
        float4 viewQuaternion = quatQuat(params.renderToViewQuat, quaternion);
        if (hasFlag(DRAW_2DGS) && any(zeroScales)) {
            float3 offset;
            if (zeroScales.z) {
                offset = float3(o.splatUv * scales.xy, 0.0);
            } else if (zeroScales.y) {
                offset = float3(o.splatUv.x * scales.x, 0.0, o.splatUv.y * scales.z);
            } else {
                offset = float3(0.0, o.splatUv * scales.yz);
            }
            o.position = project(viewCenter + quatVec(viewQuaternion, offset));
            return o;
        }
        float3x3 RS = scaleQuaternionToMatrix(scales, viewQuaternion);
        cov3D = mul(RS, transpose(RS));
    } else {
        float3x3 c = float3x3(
            xxyyzz.x, xyxzyz.x, xyxzyz.y,
            xyxzyz.x, xxyyzz.y, xyxzyz.z,
            xyxzyz.y, xyxzyz.z, xxyyzz.z);
        cov3D = mul(basis, mul(c, transpose(basis)));
    }

    // Jacobian of the projection at the centre. The GLSL builds it from
    // columns (J1.x, 0, J2.x), (0, J1.y, J2.y), 0; this is the same matrix.
    float2 scaledRenderSize = float2(params.renderWidth, params.renderHeight) * params.focalAdjustment;
    float2 focal = 0.5 * scaledRenderSize * float2(params.proj0.x, params.proj1.y);
    float3x3 J;
    if (hasFlag(DRAW_ORTHOGRAPHIC)) {
        J = float3x3(focal.x, 0.0, 0.0, 0.0, focal.y, 0.0, 0.0, 0.0, 0.0);
    } else {
        float invZ = 1.0 / viewCenter.z;
        float2 J1 = focal * invZ;
        float2 J2 = -(J1 * viewCenter.xy) * invZ;
        J = float3x3(J1.x, 0.0, 0.0, 0.0, J1.y, 0.0, J2.x, J2.y, 0.0);
    }
    float3x3 cov2D = mul(transpose(J), mul(cov3D, J));
    float a = cov2D[0][0] + params.preBlurAmount;
    float d = cov2D[1][1] + params.preBlurAmount;
    float b = cov2D[1][0];

    float fullBlurAmount = params.blurAmount;
    if (params.focalDistance > 0.0 && params.apertureAngle > 0.0) {
        float focusBlur = abs((-viewCenter.z - params.focalDistance) / viewCenter.z);
        float focusRadius = focusBlur * focal.x * tan(0.5 * params.apertureAngle);
        fullBlurAmount = clamp(sqr(focusRadius), params.blurAmount, sqr(params.maxPixelRadius));
    }

    // Convolve with a ~0.5-pixel Gaussian for anti-aliasing, keeping energy.
    float detOrig = a * d - b * b;
    a += fullBlurAmount;
    d += fullBlurAmount;
    float det = a * d - b * b;
    rgba.a *= sqrt(max(0.0, detOrig / det));
    if (rgba.a < params.minAlpha) {
        return o;
    }
    o.rgba.a = rgba.a;

    float eigenAvg = 0.5 * (a + d);
    float eigenDelta = sqrt(max(0.0, eigenAvg * eigenAvg - det));
    float eigen1 = eigenAvg + eigenDelta;
    float eigen2 = eigenAvg - eigenDelta;
    float2 eigenVec1 = (abs(b) > 0.001) ? normalize(float2(b, eigen1 - a)) : ((a >= d) ? float2(1.0, 0.0) : float2(0.0, 1.0));
    float2 eigenVec2 = float2(eigenVec1.y, -eigenVec1.x);
    float scale1 = min(params.maxPixelRadius, adjustedStdDev * sqrt(eigen1));
    float scale2 = min(params.maxPixelRadius, adjustedStdDev * sqrt(eigen2));
    if (scale1 < params.minPixelRadius && scale2 < params.minPixelRadius) {
        return o;
    }

    float2 pixelOffset = corner.x * eigenVec1 * scale1 + corner.y * eigenVec2 * scale2;
    float2 ndcOffset = (2.0 / scaledRenderSize) * pixelOffset;
    float2 ndc = clipCenter.xy / clipCenter.w + ndcOffset;
    o.position = float4(ndc * clipCenter.w, clipCenter.zw);
    return o;
}

[shader("fragment")]
float4 splatFragment(VertexOut v) : SV_Target {
    float4 rgba = v.rgba;
    float z2 = dot(v.splatUv, v.splatUv);
    if (z2 > v.adjustedStdDev * v.adjustedStdDev) {
        discard;
    }
    if (rgba.a <= 1.0) {
        rgba.a = lerp(rgba.a, rgba.a * exp(-0.5 * z2), params.falloff);
    } else {
        float a = exp((rgba.a * rgba.a - 1.0) / 2.718281828459045);
        float alpha = 1.0 - pow(1.0 - exp(-0.5 * z2), a);
        rgba.a = lerp(1.0, alpha, params.falloff);
    }
    if (rgba.a < params.minAlpha) {
        discard;
    }
    if (hasFlag(DRAW_ENCODE_LINEAR)) {
        rgba.rgb = srgbToLinear(rgba.rgb);
    }
    return hasFlag(DRAW_PREMULTIPLIED) ? float4(rgba.rgb * rgba.a, rgba.a) : rgba;
}
