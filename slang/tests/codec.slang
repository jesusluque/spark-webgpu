// Round-trips the splat codecs on the GPU: for each input splat, packs and
// unpacks it in every encoding and writes the decoded values, so a test can
// compare them with the CPU encoders in src/utils.ts.

import core.math;
import core.splat_types;
import core.packed;
import core.ext;

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
