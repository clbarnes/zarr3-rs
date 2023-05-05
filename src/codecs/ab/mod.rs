use crate::{variant_from_data, MaybeNdim};
use enum_delegate;
use serde::{Deserialize, Serialize};
pub mod sharding_indexed;
use sharding_indexed::ShardingIndexedCodec;

// pub trait ABCodec {
// fn encode(&self, raw: &[u8]) -> Vec<u8>;

// fn decode(&self, encoded: &[u8]) -> Vec<u8>;

// fn partial_decode(&self, encoded: )
// }

#[derive(Clone, Serialize, Deserialize, PartialEq, Debug)]
#[serde(rename_all = "lowercase", tag = "codec", content = "configuration")]
#[enum_delegate::implement(MaybeNdim)]
pub enum ABCodecType {
    Endian(EndianCodec),
    ShardingIndexed(ShardingIndexedCodec),
}

// variant_from_data!(ABCodecType, Endian, EndianCodec);

impl Default for ABCodecType {
    fn default() -> Self {
        Self::Endian(EndianCodec::default())
    }
}

#[derive(PartialEq, Eq, Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Endian {
    Big,
    Little,
}

impl Default for Endian {
    fn default() -> Self {
        Self::Little
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
