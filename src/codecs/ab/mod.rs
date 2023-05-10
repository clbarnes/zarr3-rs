use std::io::{Read, Write};

use crate::{
    data_type::{ReadToNdArray, ReflectedType, WriteNdArray},
    variant_from_data, CoordVec, MaybeNdim,
};
use enum_delegate;
use ndarray::ArrayD;
use serde::{Deserialize, Serialize};
// pub mod sharding_indexed;
// use sharding_indexed::ShardingIndexedCodec;

pub mod endian;
use endian::EndianCodec;

pub trait ABCodec {
    fn encode<T: ReflectedType, W: Write>(&self, decoded: ArrayD<T>, w: W);

    fn decode<R: Read, T: ReflectedType>(&self, r: R, shape: Vec<usize>) -> ArrayD<T>;
}

#[derive(Clone, Serialize, Deserialize, PartialEq, Debug)]
#[serde(rename_all = "snake_case", tag = "codec", content = "configuration")]
pub enum ABCodecType {
    Endian(EndianCodec),
    // ShardingIndexed(ShardingIndexedCodec),
}

impl ABCodec for ABCodecType {
    fn encode<T: ReflectedType, W: Write>(&self, decoded: ArrayD<T>, w: W) {
        match self {
            ABCodecType::Endian(e) => e.encode(decoded, w),
        }
    }

    fn decode<R: Read, T: ReflectedType>(&self, r: R, shape: Vec<usize>) -> ArrayD<T> {
        match self {
            ABCodecType::Endian(e) => e.decode::<R, T>(r, shape),
        }
    }
}

impl MaybeNdim for ABCodecType {
    fn maybe_ndim(&self) -> Option<usize> {
        match self {
            Self::Endian(e) => e.maybe_ndim(),
        }
    }
}

// variant_from_data!(ABCodecType, Endian, EndianCodec);

impl Default for ABCodecType {
    fn default() -> Self {
        Self::Endian(EndianCodec::default())
    }
}
