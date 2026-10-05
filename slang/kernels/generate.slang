// One SplatMesh's splats into the frame's accumulator: the compute pass that
// replaces SplatAccumulator.generate's fragment-shader "compute".
//
// Per output splat: remap the index through the LOD list, read the source
// splat (packed or ext), add its SH colour for the view direction, transform
// it from object to world space, recolour, and write it in the accumulator's
// format together with its sort metric. Mirrors SplatMesh.constructGenerator
// (src/SplatMesh.ts) and the accumulator outputs (src/SplatAccumulator.ts).
//
// Buffer layout, interleaved per splat so a stage needs only five storage
// buffers (WebGPU guarantees eight):
//   src  packed: 1 x uint4      ext: 2 x uint4 (a, b)
//   sh   packed: 3 x uint4 (sh1.xy, sh2, sh3)   ext: 4 x uint4 (sh1, sh2, sh3a, sh3b)
//   out  as src, in the accumulator's format

import core.math;
import core.splat_types;
import core.packed;
import core.ext;
import core.sh;
import core.dyno_hooks;

static const uint GEN_SRC_EXT = 1u;      // source is ext (else packed)
static const uint GEN_OUT_EXT = 2u;      // accumulator is ext (else packed)
static const uint GEN_USE_LOD = 4u;      // remap through lodIndices
static const uint GEN_LOD_OPACITY = 8u;  // packed source stores alpha / 2
static const uint GEN_SORT_RADIAL = 16u; // radial metric (else along viewDir)
static const uint GEN_DYNO_SOURCE = 256u; // dynoSource generates the splats

struct GenerateParams {
    uint numSplats;   // output splats for this mesh (the LOD count when LOD is on)
    uint outBase;     // first accumulator slot
    uint flags;
    int numSh;        // SH degrees to evaluate, 0-3

    uint srcCount;    // splats in the source
    uint pad0;
    uint pad1;
    uint pad2;

    float4 rotate;          // object to world quaternion
    float4 translateScale;  // xyz translation, w uniform scale
    float4 recolor;         // rgba multiplier
    float4 encoding;        // packed source (rgbMin, rgbMax, lnScaleMin, lnScaleMax)
    float4 shMax;           // packed SH ranges, xyz
    float4 viewObject;      // camera position in object space (SH direction)
    float4 viewCenter;      // camera position in world space (sort metric)
    float4 viewDir;         // camera forward in world space (z sort)
    float4 outOrigin;       // subtracted from centers before writing (packed)
};

StructuredBuffer<uint4> src;
StructuredBuffer<uint4> sh;
StructuredBuffer<uint> lodIndices;
RWStructuredBuffer<uint4> outSplats;
RWStructuredBuffer<float> sortMetric;
ConstantBuffer<GenerateParams> params;

bool hasFlag(uint f) { return (params.flags & f) != 0u; }

Gsplat readSource(uint index) {
    if (hasFlag(GEN_SRC_EXT)) {
        return readExtSplat(src[2u * index], src[2u * index + 1u], int(index));
    }
    Gsplat s = readPackedSplat(src[index], int(index), params.encoding);
    if (hasFlag(GEN_LOD_OPACITY)) {
        s.rgba.a *= 2.0;
    }
    return s;
}

float3 shColor(uint index, float3 center) {
    if (params.numSh <= 0) {
        return float3(0.0);
    }
    float3 dir = normalize(center - params.viewObject.xyz);
    if (hasFlag(GEN_SRC_EXT)) {
        uint b = 4u * index;
        return evaluateExtSH(params.numSh, dir, sh[b], sh[b + 1u], sh[b + 2u], sh[b + 3u]);
    }
    uint b = 3u * index;
    return evaluatePackedSH(params.numSh, dir, params.shMax.xyz, sh[b].xy, sh[b + 1u], sh[b + 2u]);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void generate(uint3 tid: SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= params.numSplats) {
        return;
    }
    uint index = hasFlag(GEN_USE_LOD) ? lodIndices[i] : i;

    Gsplat s;
    s.flags = 0u;
    if (index < params.srcCount) {
        s = hasFlag(GEN_DYNO_SOURCE) ? dynoSource(int(index)) : readSource(index);
    }

    float metric = INFINITY;
    if (isActive(s.flags)) {
        s.rgba.rgb += shColor(index, s.center);
        s = dynoObjectModifier(s);

        float scale = params.translateScale.w;
        s.center = quatVec(params.rotate, s.center * scale) + params.translateScale.xyz;
        s.scales *= scale;
        s.quaternion = quatQuat(params.rotate, s.quaternion);
        s.rgba *= params.recolor;
        s = dynoWorldModifier(s);

        float3 rel = s.center - params.viewCenter.xyz;
        metric = hasFlag(GEN_SORT_RADIAL) ? length(rel) : dot(rel, params.viewDir.xyz) + 100.0;
        if (!isActive(s.flags)) metric = INFINITY; // a modifier dropped it
    }

    uint o = params.outBase + i;
    sortMetric[o] = metric;
    float3 center = s.center - params.outOrigin.xyz;
    if (hasFlag(GEN_OUT_EXT)) {
        uint4 a = uint4(0u);
        uint4 b = uint4(0u);
        if (isActive(s.flags)) {
            packSplatExt(a, b, center, s.scales, s.quaternion, s.rgba);
        }
        outSplats[2u * o] = a;
        outSplats[2u * o + 1u] = b;
    } else {
        // The packed accumulator keeps alpha / 2 so LOD opacity can reach 2.
        float4 rgba = float4(s.rgba.rgb, 0.5 * s.rgba.a);
        outSplats[o] = isActive(s.flags) ? packSplat(center, s.scales, s.quaternion, rgba) : uint4(0u);
    }
}
