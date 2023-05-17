use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::{codecs::ArrayRepr, data_type::ReflectedType, ArcArrayD, CoordVec, MaybeNdim};

use super::AACodec;

mod strings {
    use crate::named_unit_variant;
    named_unit_variant!(C);
    named_unit_variant!(F);
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum Order {
    #[serde(with = "strings::C")]
    C,
    #[serde(with = "strings::F")]
    F,
    Permutation(CoordVec<usize>),
}

impl Order {
    /// Checks that order is a valid permutation,
    /// and simplifies to C or F if possible.
    pub fn validate(self) -> Result<Self, &'static str> {
        let permutation = match self {
            Order::C => return Ok(self),
            Order::F => return Ok(self),
            Order::Permutation(p) => p,
        };

        let mut it = permutation.iter();
        let mut last = *it.next().ok_or("Empty permutations")?;

        let mut visited = HashSet::with_capacity(permutation.len());
        visited.insert(last);

        let mut is_increasing = true;
        let mut is_decreasing = true;

        for p in it {
            if is_decreasing && p > &last {
                is_decreasing = false;
            }
            if is_increasing && p < &last {
                is_increasing = false;
            }
            if !visited.insert(*p) {
                return Err("Repeated dimension index");
            }
            last = *p;
        }

        if visited.into_iter().max().unwrap() != permutation.len() - 1 {
            return Err("Skipped dimension index");
        }

        if is_increasing {
            Ok(Self::C)
        } else if is_decreasing {
            Ok(Self::F)
        } else {
            Ok(Self::Permutation(permutation))
        }
    }

    pub fn new_permutation(permutation: CoordVec<usize>) -> Result<Self, &'static str> {
        Self::Permutation(permutation).validate()
    }
}

// impl From<NDAOrder> for Order {}

// impl TryInto<NDAOrder> for Order {}

impl Default for Order {
    fn default() -> Self {
        Self::C
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TransposeCodec {
    pub order: Order,
}

impl TransposeCodec {
    pub fn new_c() -> Self {
        Self { order: Order::C }
    }

    pub fn new_f() -> Self {
        Self { order: Order::F }
    }

    pub fn new_permutation(perm: CoordVec<usize>) -> Result<Self, &'static str> {
        Ok(Self {
            order: Order::new_permutation(perm)?,
        })
    }
}

impl Default for TransposeCodec {
    fn default() -> Self {
        Self {
            order: Default::default(),
        }
    }
}

impl AACodec for TransposeCodec {
    fn encode<T: ReflectedType>(&self, decoded: ArcArrayD<T>) -> ArcArrayD<T> {
        match &self.order {
            Order::C => decoded,
            Order::F => decoded.reversed_axes(),
            Order::Permutation(p) => decoded.permuted_axes(p.as_slice()),
        }
    }

    fn decode<T: ReflectedType>(&self, encoded: ArcArrayD<T>) -> ArcArrayD<T> {
        match &self.order {
            Order::C => encoded,
            Order::F => encoded.reversed_axes(),
            Order::Permutation(p) => encoded.permuted_axes(reverse_permutation(p).as_slice()),
        }
    }

    fn compute_encoded_representation(&self, decoded_repr: ArrayRepr) -> ArrayRepr {
        let shape = match &self.order {
            Order::C => decoded_repr.shape,
            Order::F => decoded_repr.shape.iter().rev().cloned().collect(),
            Order::Permutation(p) => p.iter().map(|idx| decoded_repr.shape[*idx]).collect(),
        };
        ArrayRepr {
            shape,
            data_type: decoded_repr.data_type,
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

impl MaybeNdim for Order {
    fn maybe_ndim(&self) -> Option<usize> {
        match self {
            Self::Permutation(p) => Some(p.len()),
            _ => None,
        }
    }
}

impl MaybeNdim for TransposeCodec {
    fn maybe_ndim(&self) -> Option<usize> {
        self.order.maybe_ndim()
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
        let to_deser = vec![r#""C""#, r#""F""#, r#"[0,1,2]"#];
        for s in to_deser.into_iter() {
            let c: Order = serde_json::from_str(s).expect(&format!("Could not deser {s}"));
            let s2 = serde_json::to_string(&c).expect(&format!("Could not ser {c:?}"));
            assert_eq!(s, &s2); // might depend on spaces
        }
    }

    fn make_arr() -> ArcArrayD<u8> {
        ArcArrayD::from_shape_vec(SHAPE.to_vec(), (0..60).collect()).unwrap()
    }

    #[test]
    fn transpose_c_is_noop() {
        let orig = make_arr();
        let t = TransposeCodec::new_c();
        let encoded = t.encode(orig.clone());
        assert_eq!(encoded.shape(), orig.shape());
        let decoded = t.decode(encoded.clone());
        assert_eq!(decoded.shape(), orig.shape());
    }

    #[test]
    fn transpose_f() {
        let orig = make_arr();
        let t = TransposeCodec::new_f();
        let encoded = t.encode(orig.clone());

        let mut rev_shape = orig.shape().to_vec();
        rev_shape.reverse();
        assert_eq!(encoded.shape(), rev_shape.as_slice());

        let decoded = t.decode(encoded.clone());
        assert_eq!(decoded.shape(), orig.shape());
    }

    #[test]
    fn transpose_permutation() {
        let orig = make_arr();
        let perm = smallvec![2, 0, 1];
        let t = TransposeCodec::new_permutation(perm.clone()).unwrap();

        let encoded = t.encode(orig.clone());
        let expected_shape: Vec<_> = perm.iter().map(|idx| SHAPE[*idx]).collect();
        assert_eq!(encoded.shape(), expected_shape.as_slice());

        let decoded = t.decode(encoded.clone());
        assert_eq!(decoded.shape(), orig.shape());
    }
}
