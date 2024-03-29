use crate::{codecs::ArrayRepr, ArcArrayD, CoordVec, MaybeNdim};
use serde::{Deserialize, Serialize};

use std::io::{Read, Write};

use super::ABCodec;
use crate::data_type::ReflectedType;

#[derive(PartialEq, Eq, Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Endian {
    Big,
    Little,
}

#[cfg(target_endian = "big")]
pub const NATIVE_ENDIAN: Endian = Endian::Big;
#[cfg(target_endian = "little")]
pub const NATIVE_ENDIAN: Endian = Endian::Little;

pub const NETWORK_ENDIAN: Endian = Endian::Big;
pub const ZARR_ENDIAN: Endian = Endian::Little;

impl Default for Endian {
    fn default() -> Self {
        ZARR_ENDIAN
    }
}

impl MaybeNdim for BytesCodec {
    fn maybe_ndim(&self) -> Option<usize> {
        None
    }
}

#[derive(PartialEq, Eq, Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BytesCodec {
    endian: Option<Endian>,
}

impl Default for BytesCodec {
    fn default() -> Self {
        Self {
            endian: Some(ZARR_ENDIAN),
        }
    }
}

impl BytesCodec {
    pub fn new(endian: Option<Endian>) -> Self {
        Self { endian }
    }

    pub fn new_big() -> Self {
        Self::new(Some(Endian::Big))
    }

    pub fn new_little() -> Self {
        Self::new(Some(Endian::Little))
    }

    pub fn new_native() -> Self {
        Self::new(Some(NATIVE_ENDIAN))
    }

    pub fn new_single_byte() -> Self {
        Self::new(None)
    }
}

impl ABCodec for BytesCodec {
    fn encode<T: ReflectedType, W: Write>(&self, decoded: ArcArrayD<T>, w: W) {
        let endian = self.valid_endian::<T>().unwrap();
        T::write_array_to(decoded, w, endian).unwrap();
    }

    fn decode<T: ReflectedType, R: Read>(&self, r: R, decoded_repr: ArrayRepr<T>) -> ArcArrayD<T> {
        if &T::ZARR_TYPE != decoded_repr.data_type() {
            panic!("Decoded array is not of the reflected type");
        }
        let endian = self.valid_endian::<T>().unwrap();
        let shape: CoordVec<_> = decoded_repr.shape.iter().map(|s| *s as usize).collect();
        T::read_array_from(r, endian, shape.as_slice())
    }

    fn endian(&self) -> Option<Endian> {
        self.endian
    }

    fn compute_encoded_size<T: ReflectedType>(&self, decoded_repr: ArrayRepr<T>) -> Option<usize> {
        Some(decoded_repr.nbytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deser_endian() {
        let s = r#"{"endian": "little"}"#;
        let _codec: BytesCodec = serde_json::from_str(s).unwrap();
    }

    #[test]
    fn deser_endian_noconfig() {
        let s = r#"{}"#;
        let _codec: BytesCodec = serde_json::from_str(s).unwrap();
    }

    #[test]
    fn can_validate_endian() {
        let ab = BytesCodec::new_big();

        ab.valid_endian::<f32>().unwrap();
        ab.valid_endian::<u8>().unwrap();
    }

    #[test]
    fn can_invalidate_endian() {
        let ab = BytesCodec::new(None);
        ab.valid_endian::<u8>().unwrap();
        ab.valid_endian::<[u8; 4]>().unwrap();
        assert!(ab.valid_endian::<f32>().is_err());
    }
}
