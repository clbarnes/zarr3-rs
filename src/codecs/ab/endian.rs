use crate::{codecs::ArrayRepr, CoordVec, MaybeNdim};
use ndarray::ArrayD;
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

impl MaybeNdim for EndianCodec {
    fn maybe_ndim(&self) -> Option<usize> {
        None
    }
}

#[derive(PartialEq, Eq, Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub struct EndianCodec {
    endian: Endian,
}

impl EndianCodec {
    pub fn new(endian: Endian) -> Self {
        Self {endian}
    }

    pub fn new_big() -> Self {
        Self::new(Endian::Big)
    }

    pub fn new_little() -> Self {
        Self::new(Endian::Little)
    }

    pub fn new_native() -> Self {
        Self::new(NATIVE_ENDIAN)
    }
}

impl ABCodec for EndianCodec {
    fn encode<T: ReflectedType, W: Write>(&self, decoded: ArrayD<T>, w: W) {
        T::write_array_to(decoded, w, self.endian).unwrap();
    }

    fn decode<T: ReflectedType, R: Read>(&self, r: R, decoded_repr: ArrayRepr) -> ArrayD<T> {
        if T::ZARR_TYPE != decoded_repr.data_type {
            panic!("Decoded array is not of the reflected type");
        }
        let shape: CoordVec<_> = decoded_repr.shape.iter().map(|s| *s as usize).collect();
        T::read_array_from(r, self.endian, shape.as_slice())
    }
}
