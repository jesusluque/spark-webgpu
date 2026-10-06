// Covariance splats (CovSplat): a splat's shape as its 3D covariance, which
// survives any linear transform (non-uniform scale, shear, linear-blend
// skinning) where scales + quaternion can't. As gsplatToCovSplat and
// CovSplatTransformer in src/SplatGenerator.ts.

module cov;

import core.math;
import core.splat_types;

public float3x3 covMatrix(float3 xxyyzz, float3 xyxzyz) {
    return float3x3(
        xxyyzz.x, xyxzyz.x, xyxzyz.y,
        xyxzyz.x, xxyyzz.y, xyxzyz.z,
        xyxzyz.y, xyxzyz.z, xxyyzz.z);
}

public CovSplat gsplatToCovSplat(Gsplat s) {
    CovSplat c;
    c.center = s.center;
    c.flags = s.flags;
    c.rgba = s.rgba;
    c.index = s.index;
    float3x3 RS = scaleQuaternionToMatrix(s.scales, s.quaternion);
    float3x3 m = mul(RS, transpose(RS));
    c.xxyyzz = float3(m[0][0], m[1][1], m[2][2]);
    c.xyxzyz = float3(m[0][1], m[0][2], m[1][2]);
    return c;
}

// center -> basis * center + offset, covariance -> basis * cov * basis^T.
public CovSplat transformCovSplat(CovSplat c, float3x3 basis, float3 offset) {
    c.center = mul(basis, c.center) + offset;
    float3x3 m = mul(basis, mul(covMatrix(c.xxyyzz, c.xyxzyz), transpose(basis)));
    c.xxyyzz = float3(m[0][0], m[1][1], m[2][2]);
    c.xyxzyz = float3(m[0][1], m[0][2], m[1][2]);
    return c;
}
