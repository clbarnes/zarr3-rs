use crate::{variant_from_data, MaybeNdim};
use enum_delegate;
use serde::{Deserialize, Serialize};
// pub mod sharding_indexed;
// use sharding_indexed::ShardingIndexedCodec;

pub mod endian;
use endian::EndianCodec;

// pub trait ABCodec {
//     fn encode<'a, W: Write + 'a>(&self, w: W) -> Box<dyn Write + 'a>;

//     fn decode<'a, R: Read + 'a>(&self, r: R) -> Box<dyn Read + 'a>;
// }

#[derive(Clone, Serialize, Deserialize, PartialEq, Debug)]
#[serde(rename_all = "snake_case", tag = "codec", content = "configuration")]
#[enum_delegate::implement(MaybeNdim)]
pub enum ABCodecType {
    Endian(EndianCodec),
    // ShardingIndexed(ShardingIndexedCodec),
}

// variant_from_data!(ABCodecType, Endian, EndianCodec);

impl Default for ABCodecType {
    fn default() -> Self {
        Self::Endian(EndianCodec::default())
    }
}
