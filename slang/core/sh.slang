// Spherical-harmonics colour for degrees 1-3, in Spark's two storage forms.
// Ported from the evaluatePackedSH* (src/PackedSplats.ts) and evaluateExtSH*
// (src/ExtSplats.ts) GLSL. Both share the basis below; they differ only in
// how the 3 + 5 + 7 RGB coefficients are stored.

module sh;

import core.ext;

// Real SH basis, in Spark's coefficient order, for a unit view direction.
public void shBasis1(float3 d, out float b[3]) {
    b[0] = -0.4886025 * d.y;
    b[1] = 0.4886025 * d.z;
    b[2] = -0.4886025 * d.x;
}

public void shBasis2(float3 d, out float b[5]) {
    b[0] = 1.0925484 * d.x * d.y;
    b[1] = -1.0925484 * d.y * d.z;
    b[2] = 0.3153915 * (2.0 * d.z * d.z - d.x * d.x - d.y * d.y);
    b[3] = -1.0925484 * d.x * d.z;
    b[4] = 0.5462742 * (d.x * d.x - d.y * d.y);
}

public void shBasis3(float3 d, out float b[7]) {
    float xx = d.x * d.x;
    float yy = d.y * d.y;
    float zz = d.z * d.z;
    b[0] = -0.5900436 * d.y * (3.0 * xx - yy);
    b[1] = 2.8906114 * d.x * d.y * d.z;
    b[2] = -0.4570458 * d.y * (4.0 * zz - xx - yy);
    b[3] = 0.3731763 * d.z * (2.0 * zz - 3.0 * xx - 3.0 * yy);
    b[4] = -0.4570458 * d.x * (4.0 * zz - xx - yy);
    b[5] = 1.4453057 * d.z * (xx - yy);
    b[6] = -0.5900436 * d.x * (xx - 3.0 * yy);
}

// Signed integer of `width` bits starting at bit `offset` of the 128-bit
// little-endian stream in `words` (a field may straddle two words).
public int readSignedBits(uint4 words, uint offset, uint width) {
    uint wi = offset >> 5u;
    uint bo = offset & 31u;
    uint v = words[wi] >> bo;
    if (bo + width > 32u) {
        v |= words[wi + 1u] << (32u - bo);
    }
    return int(v << (32u - width)) >> int(32u - width);
}

// One RGB coefficient: three consecutive signed fields.
float3 packedCoeff(uint4 words, uint k, uint width) {
    uint o = 3u * k * width;
    return float3(
        readSignedBits(words, o, width),
        readSignedBits(words, o + width, width),
        readSignedBits(words, o + 2u * width, width));
}

// PackedSplats: sh1 is 9 x sint7 in a uint2, sh2 15 x sint8 and sh3
// 21 x sint6 in a uint4 each, scaled by shMax / (2^(bits-1) - 1).
public float3 evaluatePackedSH(int numSh, float3 viewDir, float3 shMax, uint2 sh1, uint4 sh2, uint4 sh3) {
    float3 rgb = float3(0.0);
    if (numSh < 1) {
        return rgb;
    }
    float b1[3];
    shBasis1(viewDir, b1);
    uint4 w1 = uint4(sh1, 0u, 0u);
    float3 acc = float3(0.0);
    [ForceUnroll] for (uint k = 0; k < 3; k++) acc += packedCoeff(w1, k, 7u) * b1[k];
    rgb += acc * (shMax.x / 63.0);
    if (numSh < 2) {
        return rgb;
    }
    float b2[5];
    shBasis2(viewDir, b2);
    acc = float3(0.0);
    [ForceUnroll] for (uint k = 0; k < 5; k++) acc += packedCoeff(sh2, k, 8u) * b2[k];
    rgb += acc * (shMax.y / 127.0);
    if (numSh < 3) {
        return rgb;
    }
    float b3[7];
    shBasis3(viewDir, b3);
    acc = float3(0.0);
    [ForceUnroll] for (uint k = 0; k < 7; k++) acc += packedCoeff(sh3, k, 6u) * b3[k];
    rgb += acc * (shMax.z / 31.0);
    return rgb;
}

// ExtSplats: each coefficient is an ExtRgb word. sh1.xyz holds degree 1,
// sh1.w and sh2.xyzw degree 2, sh3a.xyzw and sh3b.xyz degree 3.
public float3 evaluateExtSH(int numSh, float3 viewDir, uint4 sh1, uint4 sh2, uint4 sh3a, uint4 sh3b) {
    float3 rgb = float3(0.0);
    if (numSh < 1) {
        return rgb;
    }
    float b1[3];
    shBasis1(viewDir, b1);
    [ForceUnroll] for (uint k = 0; k < 3; k++) rgb += decodeExtRgb(sh1[k]) * b1[k];
    if (numSh < 2) {
        return rgb;
    }
    float b2[5];
    shBasis2(viewDir, b2);
    uint c2[5] = { sh1.w, sh2.x, sh2.y, sh2.z, sh2.w };
    [ForceUnroll] for (uint k = 0; k < 5; k++) rgb += decodeExtRgb(c2[k]) * b2[k];
    if (numSh < 3) {
        return rgb;
    }
    float b3[7];
    shBasis3(viewDir, b3);
    uint c3[7] = { sh3a.x, sh3a.y, sh3a.z, sh3a.w, sh3b.x, sh3b.y, sh3b.z };
    [ForceUnroll] for (uint k = 0; k < 7; k++) rgb += decodeExtRgb(c3[k]) * b3[k];
    return rgb;
}
