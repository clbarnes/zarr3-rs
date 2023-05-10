use crate::{
    data_type::{ReadToNdArray, WriteNdArray, ArrayIo},
    CoordVec, MaybeNdim,
};
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
    pub fn new_big() -> Self {
        Self {
            endian: Endian::Big,
        }
    }

    pub fn new_little() -> Self {
        Self {
            endian: Endian::Little,
        }
    }
}

impl ABCodec for EndianCodec {
    fn encode<T: ReflectedType, W: Write>(&self, decoded: ArrayD<T>, w: W) {
        todo!()
        // ArrayIo::from(decoded).write_to(w, self.endian).unwrap()
    }

    fn decode<R: Read, T: ReflectedType>(
        &self,
        r: R,
        shape: Vec<usize>,
    ) -> ArrayD<T> {
        ArrayIo::<T>::read_from(r, self.endian, shape).unwrap().into()
        // let res: Result<N, &'static str> = N::read_from(r, self.endian, shape);
        // res.unwrap().into()
    }
}