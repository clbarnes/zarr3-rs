use std::io::{Read, Write};

use crate::{
    data_type::{NBytes, ReflectedType},
    variant_from_data, ArcArrayD, MaybeNdim,
};

use serde::{Deserialize, Serialize};

pub mod bytes_codec;
use bytes_codec::BytesCodec;
pub mod sharding_indexed;
use sharding_indexed::ShardingIndexedCodec;

use self::bytes_codec::Endian;

use super::ArrayRepr;

// enum_delegate doesn't work here because of type annotations?
// #[enum_delegate::register]
pub trait ABCodec {
    /// Write the given array to the given [Write]r, via the configured codecs.
    fn encode<T: ReflectedType, W: Write>(&self, decoded: ArcArrayD<T>, w: W);

    /// Read an array from the given [Read]er, via the configured codecs.
    fn decode<T: ReflectedType, R: Read>(&self, r: R, decoded_repr: ArrayRepr<T>) -> ArcArrayD<T>;

    /// The configured byte endianness for this codec.
    fn endian(&self) -> Option<Endian>;

    /// A valid endianness for this data type.
    ///
    /// Uses the given endianness if the codec's endianness is [Some], or a meaningless default if the data type does not require one (e.g. single-byte) and [None] is given, or an error if an endianness is needed but not given.
    fn valid_endian<T: ReflectedType>(&self) -> Result<Endian, &'static str> {
        T::ZARR_TYPE.valid_endian(self.endian())
    }
}

impl<C: ABCodec + ?Sized> ABCodec for Box<C> {
    fn encode<T: ReflectedType, W: Write>(&self, decoded: ArcArrayD<T>, w: W) {
        (**self).encode(decoded, w)
        // ABCodec::encode::<T, W>(self, decoded, w)
    }

    fn decode<T: ReflectedType, R: Read>(&self, r: R, decoded_repr: ArrayRepr<T>) -> ArcArrayD<T> {
        (**self).decode(r, decoded_repr)
        // ABCodec::decode::<T, R>(self, r, decoded_repr)
    }

    fn endian(&self) -> Option<Endian> {
        (**self).endian()
    }
}

#[derive(Clone, Serialize, Deserialize, PartialEq, Debug)]
#[serde(rename_all = "snake_case", tag = "name", content = "configuration")]
// #[enum_delegate::implement(ABCodec)]
pub enum ABCodecType {
    Bytes(BytesCodec),
    // box is necessary as sharding codec contains codecs,
    // so it's a recursive enum of potentially infinite size
    // ShardingIndexed(Box<ShardingIndexedCodec>),
}

impl ABCodec for ABCodecType {
    fn encode<T: ReflectedType, W: Write>(&self, decoded: ArcArrayD<T>, w: W) {
        match self {
            Self::Bytes(c) => c.encode(decoded, w),
            // Self::ShardingIndexed(c) => c.encode(decoded, w),
        }
    }

    fn decode<T: ReflectedType, R: Read>(&self, r: R, decoded_repr: ArrayRepr<T>) -> ArcArrayD<T> {
        match self {
            Self::Bytes(c) => c.decode(r, decoded_repr),
            // Self::ShardingIndexed(c) => c.decode(r, decoded_repr),
        }
    }

    fn endian(&self) -> Option<Endian> {
        match self {
            Self::Bytes(c) => c.endian(),
            // Self::ShardingIndexed(c) => c.endian(),
        }
    }
}

impl MaybeNdim for ABCodecType {
    fn maybe_ndim(&self) -> Option<usize> {
        match self {
            Self::Bytes(c) => c.maybe_ndim(),
            // Self::ShardingIndexed(c) => c.maybe_ndim(),
        }
    }
}

impl Default for ABCodecType {
    fn default() -> Self {
        Self::Bytes(BytesCodec::default())
    }
}

variant_from_data!(ABCodecType, Bytes, BytesCodec);

// impl From<ShardingIndexedCodec> for ABCodecType {
//     fn from(c: ShardingIndexedCodec) -> Self {
//         Self::ShardingIndexed(Box::new(c))
//     }
// }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn can_validate_endian() {
        let ab = ABCodecType::default();

        ab.valid_endian::<f32>().unwrap();
        ab.valid_endian::<u8>().unwrap();
    }
}
