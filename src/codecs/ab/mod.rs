use std::io::{Read, Write};

use crate::{data_type::ReflectedType, variant_from_data, MaybeNdim};

use ndarray::ArrayD;
use serde::{Deserialize, Serialize};
// pub mod sharding_indexed;
// use sharding_indexed::ShardingIndexedCodec;

pub mod endian;
pub mod sharding_indexed;
use endian::EndianCodec;
use sharding_indexed::ShardingIndexedCodec;

use super::ArrayRepr;

#[enum_delegate::register]
pub trait ABCodec {
    fn encode<T: ReflectedType, W: Write>(&self, decoded: ArrayD<T>, w: W);

    fn decode<T: ReflectedType, R: Read>(&self, r: R, decoded_repr: ArrayRepr) -> ArrayD<T>;
}

impl<C: ABCodec + ?Sized> ABCodec for Box<C> {
    fn encode<T: ReflectedType, W: Write>(&self, decoded: ArrayD<T>, w: W) {
        ABCodec::encode(self, decoded, w)
    }

    fn decode<T: ReflectedType, R: Read>(&self, r: R, decoded_repr: ArrayRepr) -> ArrayD<T> {
        ABCodec::decode(self, r, decoded_repr)
    }
}

#[derive(Clone, Serialize, Deserialize, PartialEq, Debug)]
#[serde(rename_all = "snake_case", tag = "codec", content = "configuration")]
#[enum_delegate::implement(ABCodec)]
pub enum ABCodecType {
    Endian(EndianCodec),
    ShardingIndexed(Box<ShardingIndexedCodec>),
}

impl MaybeNdim for ABCodecType {
    fn maybe_ndim(&self) -> Option<usize> {
        match self {
            Self::Endian(c) => c.maybe_ndim(),
            Self::ShardingIndexed(c) => c.maybe_ndim(),
        }
    }
}

// variant_from_data!(ABCodecType, Endian, EndianCodec);

impl Default for ABCodecType {
    fn default() -> Self {
        Self::Endian(EndianCodec::default())
    }
}

// variant_from_data!(ABCodecType, Endian, EndianCodec);
