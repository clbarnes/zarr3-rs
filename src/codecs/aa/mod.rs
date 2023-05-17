use ndarray::ArrayD;
use serde::{Deserialize, Serialize};

use crate::{data_type::ReflectedType, MaybeNdim};
mod transpose;
pub use transpose::TransposeCodec;

use super::ArrayRepr;

#[derive(Clone, Serialize, Deserialize, PartialEq, Debug)]
#[serde(rename_all = "lowercase", tag = "codec", content = "configuration")]
#[enum_delegate::implement(AACodec)]
pub enum AACodecType {
    Transpose(TransposeCodec),
}

impl MaybeNdim for AACodecType {
    fn maybe_ndim(&self) -> Option<usize> {
        match self {
            Self::Transpose(t) => t.maybe_ndim(),
        }
    }
}

// todo: methods should be able to change data type
// todo: better with GATs, somehow?
// todo: a CowArray would probably reduce copies
#[enum_delegate::register]
pub trait AACodec {
    fn encode<T: ReflectedType>(&self, decoded: ArrayD<T>) -> ArrayD<T>;

    fn decode<T: ReflectedType>(&self, encoded: ArrayD<T>) -> ArrayD<T>;

    // todo: should include shape, endianness, order?, dtype, fill value
    fn compute_encoded_representation(&self, decoded_repr: ArrayRepr) -> ArrayRepr;
}

impl AACodec for &[AACodecType] {
    fn encode<T: ReflectedType>(&self, decoded: ArrayD<T>) -> ArrayD<T> {
        let mut d = decoded;
        for c in self.iter() {
            d = c.encode(d);
        }
        d
    }

    fn decode<T: ReflectedType>(&self, encoded: ArrayD<T>) -> ArrayD<T> {
        let mut e = encoded;
        for c in self.iter().rev() {
            e = c.decode(e);
        }
        e
    }

    fn compute_encoded_representation(&self, decoded_repr: ArrayRepr) -> ArrayRepr {
        self.iter()
            .fold(decoded_repr, |d, c| c.compute_encoded_representation(d))
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use serde_json;
    use smallvec::smallvec;
    use transpose::Order;

    #[test]
    fn roundtrip_aacodec_transpose() {
        let s = r#"{"codec": "transpose", "configuration": {"order": [1, 2, 0]}}"#;
        let aa: AACodecType = serde_json::from_str(s).expect("Could not deser AACodec::Transpose");
        assert_eq!(
            aa,
            AACodecType::Transpose(TransposeCodec::new_permutation(smallvec![1, 2, 0]).unwrap())
        );

        let s = r#"{"codec": "transpose", "configuration": {"order": "C"}}"#;
        let aa: AACodecType = serde_json::from_str(s).expect("Could not deser AACodec::Transpose");
        assert_eq!(
            aa,
            AACodecType::Transpose(TransposeCodec { order: Order::C })
        );
    }
}
