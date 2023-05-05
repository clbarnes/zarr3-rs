use std::collections::HashSet;

use ndarray::Order as NDAOrder;
use serde::{Deserialize, Serialize};

use crate::{CoordVec, MaybeNdim};

#[derive(Clone, Serialize, Deserialize, PartialEq, Debug)]
#[serde(rename_all = "lowercase", tag = "codec", content = "configuration")]
#[enum_delegate::implement(MaybeNdim)]
pub enum AACodecType {
    Transpose(TransposeCodec),
}

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

    pub fn new(permutation: CoordVec<usize>) -> Result<Self, &'static str> {
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
    order: Order,
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

    #[test]
    fn roundtrip_order() {
        let to_deser = vec![r#""C""#, r#""F""#, r#"[0,1,2]"#];
        for s in to_deser.into_iter() {
            let c: Order = serde_json::from_str(s).expect(&format!("Could not deser {s}"));
            let s2 = serde_json::to_string(&c).expect(&format!("Could not ser {c:?}"));
            assert_eq!(s, &s2); // might depend on spaces
        }
    }

    #[test]
    fn roundtrip_aacodec_transpose() {
        let s = r#"{"codec": "transpose", "configuration": {"order": [1, 2, 0]}}"#;
        let aa: AACodecType = serde_json::from_str(s).expect("Could not deser AACodec::Transpose");
        assert_eq!(
            aa,
            AACodecType::Transpose(TransposeCodec {
                order: Order::Permutation(smallvec![1, 2, 0])
            })
        );

        let s = r#"{"codec": "transpose", "configuration": {"order": "C"}}"#;
        let aa: AACodecType = serde_json::from_str(s).expect("Could not deser AACodec::Transpose");
        assert_eq!(
            aa,
            AACodecType::Transpose(TransposeCodec { order: Order::C })
        );
    }
}
