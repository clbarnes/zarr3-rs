use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::{codecs::ArrayRepr, data_type::ReflectedType, ArcArrayD, CoordVec, Ndim};

use super::AACodec;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct TransposeCodec {
    pub order: CoordVec<usize>,
}

fn validate_permutation(perm: &[usize]) -> Result<(), &'static str> {
    let max = perm.len();
    let mut elems = HashSet::with_capacity(max);
    for item in perm.iter() {
        if item >= &max {
            return Err("Permutation skips some elements");
        }
        if !elems.insert(item) {
            return Err("Permutation contains repeated elements");
        }
    }

    Ok(())
}

impl TransposeCodec {
    pub fn new_transpose(ndim: usize) -> Self {
        let order = (0..ndim).rev().collect();
        Self { order }
    }

    pub fn new(perm: CoordVec<usize>) -> Result<Self, &'static str> {
        let s = Self { order: perm };
        s.validate()?;
        Ok(s)
    }

    pub fn validate(&self) -> Result<(), &'static str> {
        validate_permutation(&self.order)
    }
}

impl AACodec for TransposeCodec {
    fn encode<T: ReflectedType>(&self, decoded: ArcArrayD<T>) -> ArcArrayD<T> {
        decoded.permuted_axes(self.order.as_slice())
    }

    fn decode<T: ReflectedType>(&self, encoded: ArcArrayD<T>) -> ArcArrayD<T> {
        encoded.permuted_axes(reverse_permutation(self.order.as_slice()).as_slice())
    }

    fn compute_encoded_representation_type<T: ReflectedType>(
        &self,
        decoded_repr: ArrayRepr<T>,
    ) -> ArrayRepr<T> {
        let shape = self
            .order
            .iter()
            .map(|idx| decoded_repr.shape[*idx])
            .collect();
        ArrayRepr {
            shape,
            fill_value: decoded_repr.fill_value,
        }
    }

    fn compute_encoded_size<T: ReflectedType>(&self, decoded_repr: ArrayRepr<T>) -> ArrayRepr<T> {
        let shape = self
            .order
            .iter()
            .map(|idx| decoded_repr.shape[*idx])
            .collect();

        ArrayRepr {
            shape,
            fill_value: decoded_repr.fill_value,
        }
    }
}

fn reverse_permutation(p: &[usize]) -> CoordVec<usize> {
    let mut pos_idx: HashMap<_, _> = p.iter().enumerate().map(|(idx, pos)| (*pos, idx)).collect();
    (0..pos_idx.len())
        .map(|pos| pos_idx.remove(&pos).unwrap())
        .collect()
}

impl Ndim for TransposeCodec {
    fn ndim(&self) -> usize {
        self.order.len()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use serde_json;
    use smallvec::smallvec;
    use ArcArrayD;

    const SHAPE: [usize; 3] = [3, 4, 5];

    #[test]
    fn roundtrip_order() {
        let to_deser = vec![r#"[0,1,2]"#];
        for s in to_deser.into_iter() {
            let c: Vec<usize> =
                serde_json::from_str(s).unwrap_or_else(|_| panic!("Could not deser {s}"));
            let s2 = serde_json::to_string(&c).unwrap_or_else(|_| panic!("Could not ser {c:?}"));
            assert_eq!(s, &s2); // might depend on spaces
        }
    }

    fn make_arr() -> ArcArrayD<u8> {
        ArcArrayD::from_shape_vec(SHAPE.to_vec(), (0..60).collect()).unwrap()
    }

    #[test]
    fn transpose_permutation() {
        let orig = make_arr();
        let perm = smallvec![2, 0, 1];
        let t = TransposeCodec::new(perm.clone()).unwrap();

        let encoded = t.encode(orig.clone());
        let expected_shape: Vec<_> = perm.iter().map(|idx| SHAPE[*idx]).collect();
        assert_eq!(encoded.shape(), expected_shape.as_slice());

        let decoded = t.decode(encoded.clone());
        assert_eq!(decoded.shape(), orig.shape());
    }

    #[test]
    fn transpose() {
        let t = TransposeCodec::new_transpose(3);
        assert_eq!(t.order.as_slice(), &[2, 1, 0])
    }
}
