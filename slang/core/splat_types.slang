// A splat as the kernels pass it around, matching the dyno Gsplat struct
// (src/dyno/splats.ts) so generators port one to one.

module splat_types;

public static const uint GSPLAT_FLAG_ACTIVE = 1u;

public struct Gsplat {
    public float3 center;
    public uint flags;
    public float3 scales;
    public int index;
    public float4 quaternion;
    public float4 rgba;
};

// A splat with its covariance instead of scales and rotation.
public struct CovSplat {
    public float3 center;
    public uint flags;
    public float4 rgba;
    public float3 xxyyzz;
    public int index;
    public float3 xyxzyz;
};

public bool isActive(uint flags) {
    return (flags & GSPLAT_FLAG_ACTIVE) != 0u;
}
