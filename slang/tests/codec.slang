// Round-trips the splat codecs on the GPU: for each input splat, packs and
// unpacks it in every encoding and writes the decoded values, so a test can
// compare them with the CPU encoders in src/utils.ts.

import core.math;
import core.splat_types;
import core.packed;
import core.ext;
import core.sh;

struct CodecParams {
    uint count;
    uint pad0;
    uint pad1;
    uint pad2;
};

// Per splat: center.xyz + pad, scales.xyz + pad, quaternion, rgba.
StructuredBuffer<float4> input;
// Per splat: packed (16 B) then ext (32 B) words.
RWStructuredBuffer<uint4> encoded;
// Per splat and encoding (packed, ext): center, scales, quaternion, rgba.
RWStructuredBuffer<float4> decoded;
ConstantBuffer<CodecParams> params;

// SH: per case, view direction + numSh, shMax, packed sh1 (xy), sh2, sh3,
// then ext sh1, sh2, sh3a, sh3b. Output: packed rgb, ext rgb.
StructuredBuffer<float4> shDirs;
StructuredBuffer<uint4> shWords;
RWStructuredBuffer<float4> shRgb;

[shader("compute")]
[numthreads(64, 1, 1)]
void shEvaluate(uint3 tid: SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    float4 dirN = shDirs[2 * i];
    float3 shMax = shDirs[2 * i + 1].xyz;
    int numSh = int(dirN.w);
    uint4 w[7];
    [ForceUnroll] for (uint k = 0; k < 7; k++) w[k] = shWords[7 * i + k];
    shRgb[2 * i] = float4(evaluatePackedSH(numSh, dirN.xyz, shMax, w[0].xy, w[1], w[2]), 0.0);
    shRgb[2 * i + 1] = float4(evaluateExtSH(numSh, dirN.xyz, w[3], w[4], w[5], w[6]), 0.0);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void codecRoundTrip(uint3 tid: SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    float3 center = input[4 * i + 0].xyz;
    float3 scales = input[4 * i + 1].xyz;
    float4 quat = input[4 * i + 2];
    float4 rgba = input[4 * i + 3];

    uint4 p = packSplat(center, scales, quat, rgba);
    encoded[3 * i + 0] = p;
    Gsplat gp = readPackedSplat(p, int(i));
    decoded[8 * i + 0] = float4(gp.center, float(gp.flags));
    decoded[8 * i + 1] = float4(gp.scales, 0.0);
    decoded[8 * i + 2] = gp.quaternion;
    decoded[8 * i + 3] = gp.rgba;

    uint4 a, b;
    packSplatExt(a, b, center, scales, quat, rgba);
    encoded[3 * i + 1] = a;
    encoded[3 * i + 2] = b;
    Gsplat ge = readExtSplat(a, b, int(i));
    decoded[8 * i + 4] = float4(ge.center, float(ge.flags));
    decoded[8 * i + 5] = float4(ge.scales, 0.0);
    decoded[8 * i + 6] = ge.quaternion;
    decoded[8 * i + 7] = ge.rgba;
}
