//! Generic per-Gaussian attributes carried through LOD (the schema is
//! src/webgpu/attributes/schema.ts): one column of decoded values per
//! attribute, kept aligned with the splats through merges, permutations and
//! filtering, and merged by each attribute's `LodMerge` rule with the same
//! normalized weights the splat merge uses (area x opacity).
//!
//! Values are f64 so integer labels and ids up to 2^53 merge exactly.

use ahash::AHashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

#[derive(Clone, Debug, PartialEq)]
pub struct AttribSpec {
    pub name: String,
    /// Storage format on the GPU ("f32", "u8", ...), carried for the caller.
    pub format: String,
    pub components: usize,
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
