use serde::{Deserialize, Serialize};
use std::io::{Read, Seek};
use thiserror::Error;

use crate::chunk_arr::{ChunkSpec, ChunkSpecConstructionError};
use crate::codecs::CodecType;
use crate::{ChunkCoord, MaybeNdim, Ndim};

#[derive(Error, Debug)]
#[error("Got coord with {coord_ndim} dimensions for array of dimension {array_ndim}")]
pub struct DimensionMismatch {
    coord_ndim: usize,
    array_ndim: usize,
}

impl DimensionMismatch {
    pub fn check_coords(coord_ndim: usize, array_ndim: usize) -> Result<(), Self> {
        if coord_ndim == array_ndim {
            Ok(())
        } else {
            Err(Self {
                coord_ndim,
                array_ndim,
            })
        }
    }
}

#[derive(Clone, Serialize, Deserialize, Debug, PartialEq)]
pub struct ShardingIndexedCodec {
    pub chunk_shape: ChunkCoord,
    pub codecs: Vec<CodecType>,
}

impl Ndim for ShardingIndexedCodec {
    fn ndim(&self) -> usize  {
        self.chunk_shape.len()
    }
}

impl ShardingIndexedCodec {
    pub fn new<C: Into<ChunkCoord>>(chunk_shape: C) -> Self {
        Self {
            chunk_shape: chunk_shape.into(),
            codecs: Vec::default(),
        }
    }

    /// Ensure that all dimensioned metadata is consistent.
    fn validate_dimensions(&self) -> Result<(), &'static str> {
        // todo: how to make sure this is called after deserialisation?
        for c in self.codecs.iter() {
            self.union_ndim(c)?;
        }

        Ok(())
    }

    pub fn add_codec(&mut self, codec: CodecType) -> Result<&mut Self, &'static str> {
        self.union_ndim(&codec)?;
        self.codecs.push(codec);
        Ok(self)
    }
}

#[derive(Error, Debug)]
pub enum ChunkReadError {
    #[error("Index dimension does not match array dimension")]
    DimensionMismatch(#[from] DimensionMismatch),
    #[error("Could not read or seek")]
    Io(#[from] std::io::Error),
}

#[derive(Clone, Debug)]
pub enum ReadChunk {
    Empty,
    OutOfBounds,
    Contents(Vec<u8>),
}

impl From<ReadChunk> for Option<Vec<u8>> {
    fn from(value: ReadChunk) -> Self {
        match value {
            ReadChunk::Contents(v) => Some(v),
            _ => None,
        }
    }
}

pub struct Shard {
    config: ShardingIndexedCodec,
    chunk_spec: ChunkSpec,
}

impl Shard {
    pub fn new(config: ShardingIndexedCodec, chunk_spec: ChunkSpec) -> Self {
        Self { config, chunk_spec }
    }

    pub fn from_shard<R: Read + Seek>(
        config: ShardingIndexedCodec,
        r: &mut R,
    ) -> Result<Self, ChunkSpecConstructionError> {
        let chunk_spec = ChunkSpec::from_shard(r, config.chunk_shape.clone())?;
        Ok(Self::new(config, chunk_spec))
    }

    // pub fn read_chunk_content<R: Read + Seek>(
    //     &self,
    //     idx: &ChunkCoord,
    //     r: &mut R,
    // ) -> Result<ReadChunk, ChunkReadError> {
    // let out = match self.chunk_spec.get_idx(idx)? {
    //     Some(chunk_idx) => {
    //         if chunk_idx.is_empty() {
    //             ReadChunk::Empty
    //         } else {
    //             let encoded = chunk_idx.read_range(r)?;
    //             let decoded = self.config.codecs.as_slice().decode(&encoded);
    //             ReadChunk::Contents(decoded)
    //         }
    //     }
    //     None => ReadChunk::OutOfBounds,
    // };
    // Ok(out)
    // }
}
