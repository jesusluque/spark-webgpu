//! Generic per-Gaussian attributes carried through LOD (the schema is
//! src/webgpu/attributes/schema.ts): one column of decoded values per
//! attribute, kept aligned with the splats through merges, permutations and
//! filtering, and merged by each attribute's `LodMerge` rule with the same
//! normalized weights the splat merge uses (area x opacity).
//!
//! Values are f64 so integer labels and ids up to 2^53 merge exactly.

use ahash::AHashMap;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum LodMerge {
    /// Weighted mean of each component.
    WeightedMean,
    /// Weighted mean, renormalized to unit length (normals). Children are
    /// first flipped into the heaviest child's hemisphere: a splat's normal
    /// has no inside or outside.
    NormalizeMean,
    /// Per-component maximum.
    Max,
    /// The value (whole vector) with the largest total weight (labels).
    Mode,
    /// The first child's value.
    First,
}

impl LodMerge {
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "weightedMean" => Self::WeightedMean,
            "normalizeMean" => Self::NormalizeMean,
            "max" => Self::Max,
            "mode" => Self::Mode,
            "first" => Self::First,
            _ => return None,
        })
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::WeightedMean => "weightedMean",
            Self::NormalizeMean => "normalizeMean",
            Self::Max => "max",
            Self::Mode => "mode",
            Self::First => "first",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AttribSpec {
    pub name: String,
    /// Storage format on the GPU ("f32", "u8", ...), carried for the caller.
    pub format: String,
    pub components: usize,
    #[serde(rename = "lodMerge")]
    pub lod_merge: LodMerge,
}

#[derive(Clone, Debug, Default)]
pub struct AttribArray {
    pub specs: Vec<AttribSpec>,
    /// Per attribute, `components` values per splat.
    pub columns: Vec<Vec<f64>>,
}

impl AttribArray {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.specs.is_empty()
    }

    /// Splats per column (0 without attributes).
    pub fn len(&self) -> usize {
        match self.specs.first() {
            Some(spec) => self.columns[0].len() / spec.components.max(1),
            None => 0,
        }
    }

    /// Adds an attribute; its values must cover the same splats as the others.
    pub fn add(&mut self, spec: AttribSpec, values: Vec<f64>) -> anyhow::Result<()> {
        if spec.components == 0 || values.len() % spec.components != 0 {
            anyhow::bail!("attribute {}: {} values for {} components", spec.name, values.len(), spec.components);
        }
        let count = values.len() / spec.components;
        if !self.is_empty() && count != self.len() {
            anyhow::bail!("attribute {}: {} splats, others have {}", spec.name, count, self.len());
        }
        self.specs.push(spec);
        self.columns.push(values);
        Ok(())
    }

    /// Zeroed columns of `specs` for `count` splats, for a decoder to fill.
    pub fn new_zeroed(specs: &[AttribSpec], count: usize) -> Self {
        Self {
            specs: specs.to_vec(),
            columns: specs.iter().map(|s| vec![0.0; count * s.components]).collect(),
        }
    }

    /// Writes `count` splats of attribute `attrib` from `base`.
    pub fn set_range(&mut self, attrib: usize, base: usize, count: usize, values: &[f64]) {
        let c = self.specs[attrib].components;
        self.columns[attrib][base * c..(base + count) * c].copy_from_slice(&values[..count * c]);
    }

    /// Attributes worth storing: not an all-zero normal, which 3DGS
    /// trainers write as nx = ny = nz = 0.
    pub fn meaningful(&self) -> Vec<usize> {
        (0..self.specs.len()).filter(|&k| {
            self.specs[k].lod_merge != LodMerge::NormalizeMean || self.columns[k].iter().any(|&v| v != 0.0)
        }).collect()
    }

    pub fn get(&self, attrib: usize, index: usize) -> &[f64] {
        let c = self.specs[attrib].components;
        &self.columns[attrib][index * c..(index + 1) * c]
    }

    /// Appends zeros, for a splat added without attributes.
    pub fn push_default(&mut self) {
        for (spec, col) in self.specs.iter().zip(self.columns.iter_mut()) {
            col.resize(col.len() + spec.components, 0.0);
        }
    }

    /// Appends the merge of `indices` with their normalized `weights`.
    pub fn push_merged(&mut self, indices: &[usize], weights: &[f32]) {
        for (spec, col) in self.specs.iter().zip(self.columns.iter_mut()) {
            let merged = merge(spec, col, indices, weights);
            col.extend_from_slice(&merged);
        }
    }

    pub fn retain(&mut self, keep: &[bool]) {
        for (spec, col) in self.specs.iter().zip(self.columns.iter_mut()) {
            let c = spec.components;
            let mut out = Vec::with_capacity(col.len());
            for (i, &k) in keep.iter().enumerate() {
                if k {
                    out.extend_from_slice(&col[i * c..(i + 1) * c]);
                }
            }
            *col = out;
        }
    }

    /// Applies the swaps of tsplat::compute_swaps.
    pub fn apply_swaps(&mut self, swaps: &[(usize, usize)]) {
        for (spec, col) in self.specs.iter().zip(self.columns.iter_mut()) {
            let c = spec.components;
            for &(a, b) in swaps {
                for k in 0..c {
                    col.swap(a * c + k, b * c + k);
                }
            }
        }
    }

    pub fn truncate(&mut self, count: usize) {
        for (spec, col) in self.specs.iter().zip(self.columns.iter_mut()) {
            col.truncate(count * spec.components);
        }
    }

    pub fn from_index_map(&self, index_map: &[usize]) -> Self {
        let columns = self.specs.iter().zip(self.columns.iter()).map(|(spec, col)| {
            let c = spec.components;
            index_map.iter().flat_map(|&i| col[i * c..(i + 1) * c].iter().copied()).collect()
        }).collect();
        Self { specs: self.specs.clone(), columns }
    }

    pub fn subset(&self, start: usize, count: usize) -> Self {
        let columns = self.specs.iter().zip(self.columns.iter()).map(|(spec, col)| {
            let c = spec.components;
            col[start * c..(start + count) * c].to_vec()
        }).collect();
        Self { specs: self.specs.clone(), columns }
    }
}

fn merge(spec: &AttribSpec, col: &[f64], indices: &[usize], weights: &[f32]) -> Vec<f64> {
    let c = spec.components;
    let value = |i: usize| &col[indices[i] * c..(indices[i] + 1) * c];
    let mut out = vec![0.0; c];
    match spec.lod_merge {
        LodMerge::WeightedMean => {
            for (i, &w) in weights.iter().enumerate() {
                for (o, v) in out.iter_mut().zip(value(i)) {
                    *o += w as f64 * v;
                }
            }
        }
        LodMerge::NormalizeMean => {
            let heaviest = (0..weights.len()).max_by(|&a, &b| weights[a].total_cmp(&weights[b])).unwrap_or(0);
            let reference = value(heaviest).to_vec();
            for (i, &w) in weights.iter().enumerate() {
                let v = value(i);
                let dot: f64 = v.iter().zip(&reference).map(|(a, b)| a * b).sum();
                let w = if dot < 0.0 { -(w as f64) } else { w as f64 };
                for (o, x) in out.iter_mut().zip(v) {
                    *o += w * x;
                }
            }
            let len = out.iter().map(|x| x * x).sum::<f64>().sqrt();
            if len > 0.0 {
                out.iter_mut().for_each(|x| *x /= len);
            }
        }
        LodMerge::Max => {
            out.copy_from_slice(value(0));
            for i in 1..indices.len() {
                for (o, &v) in out.iter_mut().zip(value(i)) {
                    *o = o.max(v);
                }
            }
        }
        LodMerge::Mode => {
            // Ties go to the earliest child, for determinism.
            let mut totals: AHashMap<Vec<u64>, (f64, usize)> = AHashMap::new();
            for (i, &w) in weights.iter().enumerate() {
                let key = value(i).iter().map(|v| v.to_bits()).collect();
                totals.entry(key).or_insert((0.0, i)).0 += w as f64;
            }
            let (_, best) = totals.values().copied()
                .max_by(|a, b| a.0.total_cmp(&b.0).then(b.1.cmp(&a.1)))
                .unwrap_or((0.0, 0));
            out.copy_from_slice(value(best));
        }
        LodMerge::First => out.copy_from_slice(value(0)),
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(name: &str, components: usize, lod_merge: LodMerge) -> AttribSpec {
        AttribSpec { name: name.into(), format: "f32".into(), components, lod_merge }
    }

    #[test]
    fn merges_by_rule() {
        let mut a = AttribArray::new();
        a.add(spec("mean", 2, LodMerge::WeightedMean), vec![0.0, 10.0, 1.0, 20.0, 2.0, 30.0]).unwrap();
        a.add(spec("normal", 3, LodMerge::NormalizeMean), vec![0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0]).unwrap();
        a.add(spec("max", 1, LodMerge::Max), vec![3.0, 7.0, 5.0]).unwrap();
        a.add(spec("label", 1, LodMerge::Mode), vec![4.0, 9.0, 9.0]).unwrap();
        a.add(spec("first", 1, LodMerge::First), vec![1.0, 2.0, 3.0]).unwrap();
        a.push_merged(&[0, 1, 2], &[0.5, 0.3, 0.2]);
        assert_eq!(a.len(), 4);
        let m = a.get(0, 3);
        assert!((m[0] - 0.7).abs() < 1e-6 && (m[1] - 17.0).abs() < 1e-5);
        // (0,0,1) heaviest; (0,0,-1) flipped to agree; (1,0,0) adds x.
        let n = a.get(1, 3);
        let len = (0.2f64 * 0.2 + 0.8 * 0.8).sqrt();
        assert!((n[0] - 0.2 / len).abs() < 1e-6 && n[1].abs() < 1e-9 && (n[2] - 0.8 / len).abs() < 1e-6);
        assert_eq!(a.get(2, 3), &[7.0]);
        assert_eq!(a.get(3, 3), &[9.0]); // 0.5 for 9 vs 0.5 for 4: tie to the earliest
        assert_eq!(a.get(4, 3), &[1.0]);
    }

    #[test]
    fn mode_prefers_weight() {
        let mut a = AttribArray::new();
        a.add(spec("label", 1, LodMerge::Mode), vec![4.0, 9.0, 9.0]).unwrap();
        a.push_merged(&[0, 1, 2], &[0.6, 0.2, 0.2]);
        assert_eq!(a.get(0, 3), &[4.0]);
    }

    // Through the LOD builders: a feature equal to x must merge exactly as
    // the centers do (same weights), labels to a child's label.
    fn lod_case<SA: crate::tsplat::TsplatArray + HasAttribs>(mut splats: SA, build: impl Fn(&mut SA), n: usize, tolerance: f64) {
        use crate::tsplat::Tsplat;
        let xs: Vec<f64> = (0..n).map(|i| splats.get(i).center().x as f64).collect();
        let labels: Vec<f64> = (0..n).map(|i| if xs[i] < 0.0 { 1.0 } else { 2.0 }).collect();
        let mut attribs = AttribArray::new();
        attribs.add(spec("x", 1, LodMerge::WeightedMean), xs).unwrap();
        attribs.add(spec("label", 1, LodMerge::Mode), labels).unwrap();
        attribs.add(spec("id", 1, LodMerge::First), (0..n).map(|i| i as f64).collect()).unwrap();
        *attribs_of(&mut splats) = attribs;
        build(&mut splats);
        let attribs = attribs_of(&mut splats).clone();
        assert!(splats.len() > n);
        assert_eq!(attribs.len(), splats.len());
        let mut leaves = 0;
        for i in 0..splats.len() {
            let x = splats.get(i).center().x as f64;
            assert!((attribs.get(0, i)[0] - x).abs() < tolerance * (1.0 + x.abs()), "x at {i}");
            let children = splats.get_children(i);
            if children.is_empty() {
                leaves += 1;
                let label = attribs.get(1, i)[0];
                assert_eq!(label, if x < 0.0 { 1.0 } else { 2.0 });
            } else {
                let label = attribs.get(1, i)[0];
                assert!(children.iter().any(|&c| attribs.get(1, c)[0] == label));
                assert!(children.iter().any(|&c| attribs.get(2, c)[0] == attribs.get(2, i)[0]));
            }
        }
        assert_eq!(leaves, n);
    }

    trait HasAttribs { fn attribs_mut(&mut self) -> &mut AttribArray; }
    #[cfg(feature = "gsplat")]
    impl HasAttribs for crate::gsplat::GsplatArray { fn attribs_mut(&mut self) -> &mut AttribArray { &mut self.attribs } }
    #[cfg(feature = "csplat")]
    impl HasAttribs for crate::csplat::CsplatArray { fn attribs_mut(&mut self) -> &mut AttribArray { &mut self.attribs } }
    fn attribs_of<T: HasAttribs>(t: &mut T) -> &mut AttribArray { t.attribs_mut() }

    #[cfg(feature = "gsplat")]
    fn random_gsplats(n: usize) -> crate::gsplat::GsplatArray {
        use crate::gsplat::{Gsplat, GsplatArray};
        use crate::tsplat::TsplatArray;
        use glam::{Quat, Vec3A};
        let mut s = 12345u32;
        let mut r = move || { s = s.wrapping_mul(1664525).wrapping_add(1013904223); s as f32 / u32::MAX as f32 };
        let mut arr = GsplatArray::new_capacity(n, 0);
        for _ in 0..n {
            let c = Vec3A::new(r() * 4.0 - 2.0, r() * 4.0 - 2.0, r() * 4.0 - 2.0);
            let sc = 0.01 + 0.05 * r();
            arr.push_splat(Gsplat::new(c, 0.3 + 0.6 * r(), Vec3A::splat(r()), Vec3A::splat(sc), Quat::IDENTITY), None, None, None);
        }
        arr
    }

    #[cfg(all(feature = "gsplat", feature = "tiny_lod"))]
    #[test]
    fn through_tiny_lod() {
        lod_case(random_gsplats(3000), |s| crate::tiny_lod::compute_lod_tree(s, 1.5, true, |_| {}), 3000, 1e-3);
    }

    #[cfg(all(feature = "gsplat", feature = "bhatt_lod"))]
    #[test]
    fn through_bhatt_lod() {
        lod_case(random_gsplats(1500), |s| crate::bhatt_lod::compute_lod_tree(s, 1.5, |_| {}), 1500, 1e-3);
    }

    #[cfg(all(feature = "gsplat", feature = "quick_lod"))]
    #[test]
    fn through_quick_lod() {
        lod_case(random_gsplats(3000), |s| crate::quick_lod::compute_lod_tree(s, 1.5, true, |_| {}), 3000, 1e-3);
    }

    // The packed path's arrays, with quantized centers.
    #[cfg(all(feature = "gsplat", feature = "csplat", feature = "tiny_lod"))]
    #[test]
    fn through_tiny_lod_csplat() {
        use crate::csplat::{Csplat, CsplatArray};
        use crate::tsplat::{Tsplat, TsplatArray};
        let g = random_gsplats(3000);
        let mut c = CsplatArray::new_capacity(3000, 0);
        for i in 0..3000 {
            let s = g.get(i);
            c.splats.push(Csplat::new(s.center(), s.opacity(), s.rgb(), s.scales(), s.quaternion(), &c.encoding));
        }
        lod_case(c, |s| crate::tiny_lod::compute_lod_tree(s, 1.5, true, |_| {}), 3000, 1e-2);
    }

    // Attributes round-trip through a .rad file in their own formats.
    #[cfg(all(feature = "gsplat", feature = "rad"))]
    #[test]
    fn through_rad() {
        use crate::decoder::ChunkReceiver;
        use crate::gsplat::GsplatArray;
        use crate::rad::{RadDecoder, RadEncoder};
        use crate::tsplat::TsplatArray;
        // Over one 65536-splat chunk, so the second chunk's base matters.
        let n = 70000;
        let mut splats = random_gsplats(n);
        let mut attribs = AttribArray::new();
        let fmt = |name: &str, format: &str, components: usize, lod_merge| AttribSpec {
            name: name.into(), format: format.into(), components, lod_merge,
        };
        attribs.add(fmt("label", "u8", 1, LodMerge::Mode), (0..n).map(|i| (i % 251) as f64).collect()).unwrap();
        attribs.add(fmt("id", "u32", 1, LodMerge::First), (0..n).map(|i| (i * 977) as f64).collect()).unwrap();
        attribs.add(fmt("normal", "snorm8", 3, LodMerge::NormalizeMean), (0..3 * n).map(|k| if k % 3 == 2 { -1.0 } else { 0.0 }).collect()).unwrap();
        attribs.add(fmt("feature", "f16", 5, LodMerge::WeightedMean), (0..5 * n).map(|k| (k % 1000) as f64 * 0.5).collect()).unwrap();
        attribs.add(fmt("weight", "f32", 1, LodMerge::Max), (0..n).map(|i| i as f64 / 3.0).collect()).unwrap();
        attribs.add(fmt("mask", "unorm8", 2, LodMerge::WeightedMean), (0..2 * n).map(|k| (k % 256) as f64 / 255.0).collect()).unwrap();
        splats.attribs = attribs.clone();

        let mut bytes = Vec::new();
        RadEncoder::new(splats).encode(&mut bytes).unwrap();
        let mut decoder = RadDecoder::new(GsplatArray::new());
        decoder.push(&bytes).unwrap();
        decoder.finish().unwrap();
        let out = decoder.into_splats();
        assert_eq!(out.len(), n);
        assert_eq!(out.attribs.specs, attribs.specs);
        for (k, spec) in attribs.specs.iter().enumerate() {
            for i in [0, 1, 4095, 65535, 65536, n - 1] {
                let (got, want) = (out.attribs.get(k, i), attribs.get(k, i));
                for c in 0..spec.components {
                    let tol = if spec.format == "f32" { 1e-6 * (1.0 + want[c].abs()) } else { 1e-2 * (1.0 + want[c].abs()) };
                    assert!((got[c] - want[c]).abs() <= tol, "{} [{i}][{c}]: {} vs {}", spec.name, got[c], want[c]);
                }
            }
        }
    }

    // A 3DGS PLY with extra properties: decoded into attributes, through
    // LOD, then through a .rad file.
    #[cfg(all(feature = "gsplat", feature = "ply", feature = "rad", feature = "tiny_lod"))]
    #[test]
    fn ply_to_lod_to_rad() {
        use crate::decoder::ChunkReceiver;
        use crate::gsplat::GsplatArray;
        use crate::ply::PlyDecoder;
        use crate::rad::{RadDecoder, RadEncoder};
        use crate::tsplat::{Tsplat, TsplatArray};
        let n = 4000;
        let floats = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
            "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3", "feat_0", "feat_1"];
        let mut header = format!("ply\nformat binary_little_endian 1.0\nelement vertex {n}\n");
        for p in floats {
            header += &format!("property float {p}\n");
        }
        header += "property uchar label\nproperty uint id\nproperty float zero_nx\nend_header\n";
        let mut bytes = header.into_bytes();
        let mut s = 7u32;
        let mut r = move || { s = s.wrapping_mul(1664525).wrapping_add(1013904223); s as f32 / u32::MAX as f32 };
        for i in 0..n {
            let x = if i % 2 == 0 { -2.0 } else { 2.0 } + r() - 0.5;
            let vals = [x, r() - 0.5, r() - 0.5, 0.0, 0.0, if r() < 0.5 { 1.0 } else { -1.0 }, 0.0, 0.0, 0.0, 2.0,
                -4.0, -4.0, -4.0, 1.0, 0.0, 0.0, 0.0, x, 1.0];
            for v in vals {
                bytes.extend(v.to_le_bytes());
            }
            bytes.push(if x < 0.0 { 1 } else { 2 });
            bytes.extend((100_000_000u32 + i).to_le_bytes());
            bytes.extend(0.0f32.to_le_bytes());
        }

        let mut ply = PlyDecoder::new(GsplatArray::new());
        for chunk in bytes.chunks(1000) {
            ply.push(chunk).unwrap();
        }
        ply.finish().unwrap();
        let mut splats = ply.into_splats();
        let names: Vec<_> = splats.attribs.specs.iter().map(|s| (s.name.as_str(), s.format.as_str(), s.components, s.lod_merge)).collect();
        assert_eq!(names, vec![
            ("normal", "f32", 3, LodMerge::NormalizeMean),
            ("feat", "f32", 2, LodMerge::WeightedMean),
            ("label", "u8", 1, LodMerge::Mode),
            ("id", "u32", 1, LodMerge::Mode),
            ("zero_nx", "f32", 1, LodMerge::WeightedMean),
        ]);
        assert_eq!(splats.attribs.get(3, 17), &[100_000_017.0]);
        assert_eq!(splats.attribs.get(1, 5)[0], splats.get(5).center().x as f64);

        crate::tiny_lod::compute_lod_tree(&mut splats, 1.5, true, |_| {});
        let lod_count = splats.len();
        assert!(lod_count > n as usize);

        let mut rad = Vec::new();
        RadEncoder::new(splats).encode(&mut rad).unwrap();
        let mut decoder = RadDecoder::new(GsplatArray::new());
        decoder.push(&rad).unwrap();
        decoder.finish().unwrap();
        let out = decoder.into_splats();
        assert_eq!(out.len(), lod_count);
        for i in 0..lod_count {
            let x = out.get(i).center().x as f64;
            let feat = out.attribs.get(1, i);
            assert!((feat[0] - x).abs() < 1e-3 * (1.0 + x.abs()) && (feat[1] - 1.0).abs() < 1e-5);
            let normal = out.attribs.get(0, i);
            assert!(normal[2].abs() > 0.999, "normal {normal:?}");
            let label = out.attribs.get(2, i)[0];
            assert!(label == 1.0 || label == 2.0);
            assert!(out.attribs.get(3, i)[0] >= 100_000_000.0);
        }
    }

    #[test]
    fn keeps_alignment() {
        let mut a = AttribArray::new();
        a.add(spec("v", 2, LodMerge::First), vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]).unwrap();
        a.apply_swaps(&[(0, 3)]);
        assert_eq!(a.get(0, 0), &[3.0, 3.5]);
        a.retain(&[true, false, true, true]);
        assert_eq!(a.columns[0], vec![3.0, 3.5, 2.0, 2.5, 0.0, 0.5]);
        assert_eq!(a.from_index_map(&[2, 0]).columns[0], vec![0.0, 0.5, 3.0, 3.5]);
        assert_eq!(a.subset(1, 1).columns[0], vec![2.0, 2.5]);
        a.truncate(1);
        assert_eq!(a.len(), 1);
    }
}
