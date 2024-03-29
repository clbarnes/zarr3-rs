use serde::{Deserialize, Serialize};

use crate::{data_type::ReflectedType, ArcArrayD, MaybeNdim};
mod transpose;
pub use transpose::TransposeCodec;

use super::ArrayRepr;

#[derive(Clone, Serialize, Deserialize, PartialEq, Debug)]
#[serde(rename_all = "lowercase", tag = "name", content = "configuration")]
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
// todo: generic array type to minimise copies? may need different input and output types
#[enum_delegate::register]
pub trait AACodec {
    fn encode<T: ReflectedType>(&self, decoded: ArcArrayD<T>) -> ArcArrayD<T>;

    fn decode<T: ReflectedType>(&self, encoded: ArcArrayD<T>) -> ArcArrayD<T>;

    fn compute_encoded_representation_type<T: ReflectedType>(
        &self,
        decoded_repr: ArrayRepr<T>,
    ) -> ArrayRepr<T>;

    // todo: might change output type
    fn compute_encoded_size<T: ReflectedType>(&self, decoded_repr: ArrayRepr<T>) -> ArrayRepr<T>;
}

impl AACodec for &[AACodecType] {
    fn encode<T: ReflectedType>(&self, decoded: ArcArrayD<T>) -> ArcArrayD<T> {
        let mut d = decoded;
        for c in self.iter() {
            d = c.encode(d);
        }
        d
    }

    fn decode<T: ReflectedType>(&self, encoded: ArcArrayD<T>) -> ArcArrayD<T> {
        let mut e = encoded;
        for c in self.iter().rev() {
            e = c.decode(e);
        }
        e
    }

    fn compute_encoded_representation_type<T: ReflectedType>(
        &self,
        decoded_repr: ArrayRepr<T>,
    ) -> ArrayRepr<T> {
        self.iter().fold(decoded_repr, |d, c| {
            c.compute_encoded_representation_type(d)
        })
    }

    fn compute_encoded_size<T: ReflectedType>(&self, decoded_repr: ArrayRepr<T>) -> ArrayRepr<T> {
        self.iter()
            .fold(decoded_repr, |d, c| c.compute_encoded_size(d))
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use serde_json;
    use smallvec::smallvec;

    #[test]
    fn roundtrip_aacodec_transpose() {
        let s = r#"{"name": "transpose", "configuration": {"order": [1, 2, 0]}}"#;
        let aa: AACodecType =
            serde_json::from_str(s).expect("Could not deser AACodecType::Transpose");
        assert_eq!(
            aa,
            AACodecType::Transpose(TransposeCodec::new(smallvec![1, 2, 0]).unwrap())
        );

        let s = r#"{"name": "transpose", "configuration": {"order": [2, 1, 0]}}"#;
        let aa: AACodecType =
            serde_json::from_str(s).expect("Could not deser AACodecType::Transpose");
        assert_eq!(aa, AACodecType::Transpose(TransposeCodec::new_transpose(3)));
    }
}
