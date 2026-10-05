// Round-trips core/attrib.slang against the TS packer
// (src/webgpu/attributes/schema.ts): decodes every attribute of every splat,
// reads every component's raw bits, and re-encodes each group into a copy.

import core.attrib;

struct AttribTestParams {
    uint count;
    uint numAttribs;
    uint maxGroups;
    uint maxComps;
};

StructuredBuffer<uint> pool;
// Same header as pool; attribEncode fills the data.
RWStructuredBuffer<uint> rewritten;
// [splat][attrib][group]
RWStructuredBuffer<float4> decoded;
// [splat][attrib][component]
RWStructuredBuffer<uint> raw;
ConstantBuffer<AttribTestParams> params;

[shader("compute")]
[numthreads(64, 1, 1)]
void attribDecode(uint3 tid: SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    for (uint a = 0u; a < params.numAttribs; a++) {
        for (uint g = 0u; g < params.maxGroups; g++) {
            decoded[(i * params.numAttribs + a) * params.maxGroups + g] = readAttrib(pool, a, i, g);
        }
        for (uint c = 0u; c < params.maxComps; c++) {
            raw[(i * params.numAttribs + a) * params.maxComps + c] = readAttribU(pool, a, i, c);
        }
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void attribEncode(uint3 tid: SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    for (uint a = 0u; a < params.numAttribs; a++) {
        AttribDesc d = attribDesc(pool, a);
        for (uint g = 0u; 4u * g < d.components; g++) {
            writeAttrib(rewritten, a, i, g, readAttrib(pool, a, i, g));
        }
    }
}
