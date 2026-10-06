// Copies a per-splat pool to a wider stride (WgpuSplatPager's SH pool when a
// page with more SH degrees arrives): splat i's `oldStride` uint4s move from
// src[i * oldStride] to dst[i * newStride], and the new tail is zeroed.

struct RestrideParams {
    uint count;     // splats
    uint oldStride; // uint4s per splat in src
    uint newStride; // uint4s per splat in dst (>= oldStride)
    uint pad0;
};

StructuredBuffer<uint4> src;
RWStructuredBuffer<uint4> dst;
ConstantBuffer<RestrideParams> params;

[shader("compute")]
[numthreads(256, 1, 1)]
void restride(uint3 tid: SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    for (uint k = 0u; k < params.newStride; k++) {
        dst[i * params.newStride + k] = k < params.oldStride ? src[i * params.oldStride + k] : uint4(0u);
    }
}
