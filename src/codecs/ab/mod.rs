use std::io::{Read, Write};

use crate::{data_type::ReflectedType, variant_from_data, ArcArrayD, MaybeNdim};

use bytes::Bytes;
use serde::{Deserialize, Serialize};
pub mod endian;
use endian::EndianCodec;
// pub mod sharding_indexed;
// use sharding_indexed::ShardingIndexedCodec;

use super::ArrayRepr;

// enum_delegate doesn't work here because of type annotations?
// #[enum_delegate::register]
pub trait ABCodec {
    fn encode<T: ReflectedType>(&self, decoded: ArcArrayD<T>) -> Bytes;

    fn decode<T: ReflectedType>(&self, encoded: &[u8], decoded_repr: ArrayRepr<T>) -> ArcArrayD<T>;
}

impl<C: ABCodec + ?Sized> ABCodec for Box<C> {
    fn encode<T: ReflectedType>(&self, decoded: ArcArrayD<T>) -> Bytes {
        (**self).encode(decoded)
        // ABCodec::encode::<T, W>(self, decoded, w)
    }

    fn decode<T: ReflectedType>(&self, encoded: &[u8], decoded_repr: ArrayRepr<T>) -> ArcArrayD<T> {
        (**self).decode(encoded, decoded_repr)
        // ABCodec::decode::<T, R>(self, r, decoded_repr)
    }
}

#[derive(Clone, Serialize, Deserialize, PartialEq, Debug)]
#[serde(rename_all = "snake_case", tag = "codec", content = "configuration")]
// #[enum_delegate::implement(ABCodec)]
pub enum ABCodecType {
    Endian(EndianCodec),
    // box is necessary as sharding codec contains codecs,
    // so it's a recursive enum of potentially infinite size
    // ShardingIndexed(Box<ShardingIndexedCodec>),
}

impl ABCodec for ABCodecType {
    fn encode<T: ReflectedType>(&self, decoded: ArcArrayD<T>) -> Bytes {
        match self {
            Self::Endian(c) => c.encode(decoded),
            // Self::ShardingIndexed(c) => c.encode(decoded, w),
        }
    }

    fn decode<T: ReflectedType>(&self, encoded: &[u8], decoded_repr: ArrayRepr<T>) -> ArcArrayD<T> {
        match self {
            Self::Endian(c) => c.decode(encoded, decoded_repr),
            // Self::ShardingIndexed(c) => c.decode(r, decoded_repr),
        }
    }
}

impl MaybeNdim for ABCodecType {
    fn maybe_ndim(&self) -> Option<usize> {
        match self {
            Self::Endian(c) => c.maybe_ndim(),
            // Self::ShardingIndexed(c) => c.maybe_ndim(),
        }
    }
}

impl Default for ABCodecType {
    fn default() -> Self {
        Self::Endian(EndianCodec::default())
    }
}

variant_from_data!(ABCodecType, Endian, EndianCodec);

// impl From<ShardingIndexedCodec> for ABCodecType {
//     fn from(c: ShardingIndexedCodec) -> Self {
//         Self::ShardingIndexed(Box::new(c))
//     }
// }
