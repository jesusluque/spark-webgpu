// Between a WebGPU texture and the picture layout the aofx kernels use.
//
// A picture is a flat float4 buffer: premultiplied, linear, y up, `stride`
// pixels a row. A texture is y down in whatever format three gave it. The host
// copies the texture into a buffer of texels (copyTextureToBuffer, rows padded
// to 256 bytes), `unpackTexels` turns that into a picture, and `packTexels`
// does the reverse for copyBufferToTexture. Going through buffer copies rather
// than texture bindings keeps every kernel here a plain aofx kernel -- buffers
// and one constant block -- and works for any copyable colour format,
// bgra8unorm included, without asking for bgra8unorm-storage.
//
// `restoreChannels` is the host's R/G/B/A switches: every aofx node gets them,
// and the channels nobody asked to change come back from the pass-through
// input.

struct BridgeParams {
    uint width;        // picture and texture size, in pixels
    uint height;
    uint stride;       // picture row, in pixels (the one written)
    uint rowWords;     // texel row, in 32-bit words (bytesPerRow / 4)
    uint format;       // kRgba16Float, kRgba8Unorm, kBgra8Unorm, kRgba32Float
    uint srgb;         // texels are sRGB-encoded (an -srgb format, or by intent)
    uint flipY;        // texture row 0 is the picture's top row
    uint unpremultiplied; // texels are straight alpha; the picture never is
    // packTexels and restoreChannels: where the picture read sits relative
    // to the one written (or the texture), and, for restoreChannels, which
    // channels the effect was allowed to change.
    int  srcOffsetX;
    int  srcOffsetY;
    uint srcWidth;
    uint srcHeight;
    uint srcStride;
    uint channelMask;  // bit 0 R ... bit 3 A
    uint pad0;
    uint pad1;
};

static const uint kRgba16Float = 0u;
static const uint kRgba8Unorm = 1u;
static const uint kBgra8Unorm = 2u;
static const uint kRgba32Float = 3u;

StructuredBuffer<uint>       texels;
RWStructuredBuffer<uint>     texelsOut;
StructuredBuffer<float4>     src;
RWStructuredBuffer<float4>   picture;
ConstantBuffer<BridgeParams> params;

float toLinear(float c) {
    return c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4);
}

float toSrgb(float c) {
    c = max(c, 0.0);
    return c <= 0.0031308 ? c * 12.92 : 1.055 * pow(c, 1.0 / 2.4) - 0.055;
}

float2 halves(uint word) {
    return float2(f16tof32(word & 0xffffu), f16tof32(word >> 16u));
}

uint packHalves(float2 v) {
    return f32tof16(v.x) | (f32tof16(v.y) << 16u);
}

float4 unorm4(uint word) {
    return float4(float(word & 0xffu), float((word >> 8u) & 0xffu),
                  float((word >> 16u) & 0xffu), float(word >> 24u)) / 255.0;
}

uint packUnorm4(float4 v) {
    const uint4 q = uint4(round(saturate(v) * 255.0));
    return q.x | (q.y << 8u) | (q.z << 16u) | (q.w << 24u);
}

uint texelRow(uint y) {
    return params.flipY != 0u ? params.height - 1u - y : y;
}

[numthreads(16, 16, 1)]
void unpackTexels(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.width || tid.y >= params.height) {
        return;
    }
    const uint row = texelRow(tid.y) * params.rowWords;
    float4 c;
    if (params.format == kRgba16Float) {
        const uint at = row + tid.x * 2u;
        c = float4(halves(texels[at]), halves(texels[at + 1u]));
    } else if (params.format == kRgba32Float) {
        const uint at = row + tid.x * 4u;
        c = float4(asfloat(texels[at]), asfloat(texels[at + 1u]),
                   asfloat(texels[at + 2u]), asfloat(texels[at + 3u]));
    } else {
        c = unorm4(texels[row + tid.x]);
        if (params.format == kBgra8Unorm) {
            c = c.zyxw;
        }
    }
    if (params.srgb != 0u) {
        c = float4(toLinear(c.x), toLinear(c.y), toLinear(c.z), c.w);
    }
    if (params.unpremultiplied != 0u) {
        c = float4(c.xyz * c.w, c.w);
    }
    picture[tid.y * params.stride + tid.x] = c;
}

[numthreads(16, 16, 1)]
void packTexels(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.width || tid.y >= params.height) {
        return;
    }
    // The picture need not cover the texture: a cut crop or a transform
    // leaves it smaller or elsewhere, and outside it is transparent.
    const int sx = int(tid.x) + params.srcOffsetX;
    const int sy = int(tid.y) + params.srcOffsetY;
    float4 c = float4(0.0, 0.0, 0.0, 0.0);
    if (sx >= 0 && sy >= 0 && sx < int(params.srcWidth) &&
        sy < int(params.srcHeight)) {
        c = src[uint(sy) * params.srcStride + uint(sx)];
    }
    if (params.unpremultiplied != 0u) {
        c = c.w > 0.0 ? float4(c.xyz / c.w, c.w) : float4(0.0, 0.0, 0.0, 0.0);
    }
    if (params.srgb != 0u) {
        c = float4(toSrgb(c.x), toSrgb(c.y), toSrgb(c.z), c.w);
    }
    const uint row = texelRow(tid.y) * params.rowWords;
    if (params.format == kRgba16Float) {
        const uint at = row + tid.x * 2u;
        texelsOut[at] = packHalves(c.xy);
        texelsOut[at + 1u] = packHalves(c.zw);
    } else if (params.format == kRgba32Float) {
        const uint at = row + tid.x * 4u;
        texelsOut[at] = asuint(c.x);
        texelsOut[at + 1u] = asuint(c.y);
        texelsOut[at + 2u] = asuint(c.z);
        texelsOut[at + 3u] = asuint(c.w);
    } else {
        texelsOut[row + tid.x] =
            packUnorm4(params.format == kBgra8Unorm ? c.zyxw : c);
    }
}

[numthreads(16, 16, 1)]
void restoreChannels(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.width || tid.y >= params.height) {
        return;
    }
    const int sx = int(tid.x) + params.srcOffsetX;
    const int sy = int(tid.y) + params.srcOffsetY;
    float4 before = float4(0.0, 0.0, 0.0, 0.0);
    if (sx >= 0 && sy >= 0 && sx < int(params.srcWidth) &&
        sy < int(params.srcHeight)) {
        before = src[uint(sy) * params.srcStride + uint(sx)];
    }
    const uint at = tid.y * params.stride + tid.x;
    const float4 after = picture[at];
    const uint m = params.channelMask;
    picture[at] = float4((m & 1u) != 0u ? after.x : before.x,
                         (m & 2u) != 0u ? after.y : before.y,
                         (m & 4u) != 0u ? after.z : before.z,
                         (m & 8u) != 0u ? after.w : before.w);
}
