// Folded-octahedral quaternion encodings: an axis on the octahedron plus the
// rotation angle. Ported from src/shaders/splatDefines.glsl.

module quat_codec;

import core.math;

// Axis of a quaternion folded onto [0,1]^2, and its angle in [0, pi].
float3 octAxisAngle(float4 q) {
    if (q.w < 0.0) {
        q = -q;
    }
    float halfTheta = acos(q.w);
    float s = sin(halfTheta);
    float3 axis = (abs(s) < 1e-6) ? float3(1.0, 0.0, 0.0) : q.xyz / s;
    float sum = abs(axis.x) + abs(axis.y) + abs(axis.z);
    float2 p = axis.xy / sum;
    if (axis.z < 0.0) {
        float oldPx = p.x;
        p.x = (1.0 - abs(p.y)) * (p.x >= 0.0 ? 1.0 : -1.0);
        p.y = (1.0 - abs(oldPx)) * (p.y >= 0.0 ? 1.0 : -1.0);
    }
    return float3(p * 0.5 + 0.5, 2.0 * halfTheta);
}

float4 octAxisAngleToQuat(float2 uv, float theta) {
    float2 f = uv * 2.0 - 1.0;
    float3 axis = float3(f, 1.0 - abs(f.x) - abs(f.y));
    float t = max(-axis.z, 0.0);
    axis.x += (axis.x >= 0.0) ? -t : t;
    axis.y += (axis.y >= 0.0) ? -t : t;
    axis = normalize(axis);
    float halfTheta = theta * 0.5;
    return float4(axis * sin(halfTheta), cos(halfTheta));
}

// 24 bits: u in [0,8), v in [8,16), angle in [16,24).
public uint encodeQuatOctXy88R8(float4 q) {
    float3 uvt = octAxisAngle(q);
    uint u = uint(clamp(round(uvt.x * 255.0), 0.0, 255.0));
    uint v = uint(clamp(round(uvt.y * 255.0), 0.0, 255.0));
    uint a = uint(clamp(round((uvt.z / 3.14159265359) * 255.0), 0.0, 255.0));
    return (a << 16u) | (v << 8u) | u;
}

public float4 decodeQuatOctXy88R8(uint encoded) {
    float2 uv = float2(encoded & 0xffu, (encoded >> 8u) & 0xffu) / 255.0;
    float theta = (float(encoded >> 16u) / 255.0) * 3.14159265359;
    return octAxisAngleToQuat(uv, theta);
}

// 32 bits: u in [0,10), v in [10,20), angle in [20,32).
public uint encodeQuatOctXy1010R12(float4 q) {
    float3 uvt = octAxisAngle(q);
    uint u = uint(clamp(round(uvt.x * 1023.0), 0.0, 1023.0));
    uint v = uint(clamp(round(uvt.y * 1023.0), 0.0, 1023.0));
    uint a = uint(clamp(round((uvt.z / PI) * 4095.0), 0.0, 4095.0));
    return (a << 20u) | (v << 10u) | u;
}

public float4 decodeQuatOctXy1010R12(uint encoded) {
    float2 uv = float2(encoded & 0x3ffu, (encoded >> 10u) & 0x3ffu) / 1023.0;
    float theta = (float(encoded >> 20u) / 4095.0) * PI;
    return octAxisAngleToQuat(uv, theta);
}
