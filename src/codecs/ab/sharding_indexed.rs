use serde::{Deserialize, Serialize};
use std::io::{Read, Seek};
use thiserror::Error;

use crate::chunk_arr::{ChunkSpec, ChunkSpecConstructionError};
use crate::codecs::CodecType;
use crate::{ChunkCoord, MaybeNdim, Ndim};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::{Read, Seek, SeekFrom, Write};
use thiserror::Error;

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
    fn ndim(&self) -> usize {
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

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ChunkIndex {
    // todo: replace with Option<usize>?
    pub offset: u64,
    pub nbytes: u64,
}

// todo: replace with Option<ChunkIndex>?
impl ChunkIndex {
    pub fn is_empty(&self) -> bool {
        self.offset == u64::MAX && self.nbytes == u64::MAX
    }

    pub fn empty() -> Self {
        Self {
            offset: u64::MAX,
            nbytes: u64::MAX,
        }
    }

    pub fn from_reader<R: Read>(r: &mut R) -> Result<Self, std::io::Error> {
        let offset = r.read_u64::<LittleEndian>()?;
        let nbytes = r.read_u64::<LittleEndian>()?;
        Ok(Self { offset, nbytes })
    }

    pub fn write<W: Write>(&self, w: &mut W) -> Result<(), std::io::Error> {
        w.write_u64::<LittleEndian>(self.offset)?;
        w.write_u64::<LittleEndian>(self.nbytes)?;
        Ok(())
    }

    pub fn read_range<R: Read + Seek>(&self, r: &mut R) -> Result<Vec<u8>, std::io::Error> {
        let mut buf = vec![0; self.nbytes as usize];
        r.seek(SeekFrom::Start(self.nbytes))?;
        r.read_exact(&mut buf)?;
        Ok(buf)
    }

    pub fn end_offset(&self) -> Option<u64> {
        if self.is_empty() {
            None
        } else {
            Some(self.offset + self.nbytes)
        }
    }
}

impl PartialOrd for ChunkIndex {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ChunkIndex {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let cmp = self.offset.cmp(&other.offset);
        if cmp.is_eq() {
            self.nbytes.cmp(&other.nbytes)
        } else {
            cmp
        }
    }
}

#[derive(Error, Debug)]
pub enum ChunkSpecError {
    #[error("Scalar datasets (i.e. empty chunk shape array) cannot be sharded")]
    EmptyChunkShape,
    #[error("Chunk shape contains a zero")]
    ZeroChunkDimension,
    #[error("Product of chunk shape array ({0}) does not match number of chunks ({1})")]
    MismatchedChunkNumber(usize, usize),
}

impl ChunkSpecError {
    fn check_data(data_len: usize, shape: &ChunkCoord) -> Result<(), Self> {
        if shape.is_empty() {
            return Err(Self::EmptyChunkShape);
        }
        let mut prod: usize = 1;
        for s in shape.iter() {
            if *s == 0 {
                return Err(Self::ZeroChunkDimension);
            }
            prod *= *s as usize;
        }
        if data_len != prod {
            return Err(Self::MismatchedChunkNumber(prod, data_len));
        }
        Ok(())
    }
}

#[derive(Error, Debug)]
pub enum ChunkSpecConstructionError {
    #[error("Chunk spec is malformed")]
    MalformedSpec(#[from] ChunkSpecError),
    #[error("Could not read chunk index")]
    IoError(#[from] std::io::Error),
}

/// C order
fn to_linear_idx(
    coord: &ChunkCoord,
    shape: &ChunkCoord,
) -> Result<Option<usize>, DimensionMismatch> {
    DimensionMismatch::check_coords(coord.len(), shape.len())?;

    let mut total = 0;
    let mut prev_s: usize = 1;
    for (s, i) in shape.iter().rev().zip(coord.iter().rev()) {
        if i >= s {
            return Ok(None);
        }
        total += *i as usize * prev_s;
        prev_s = *s as usize;
    }
    Ok(Some(total))
}

#[derive(Error, Debug)]
pub enum ChunkSpecModificationError {
    #[error("Index {coord:?} is out of bounds of shape {shape:?}")]
    OutOfBounds {
        coord: ChunkCoord,
        shape: ChunkCoord,
    },
    #[error("Dimension mismatch")]
    DimensionMismatch(#[from] DimensionMismatch),
}

pub struct ChunkSpec {
    chunk_idxs: Vec<ChunkIndex>,
    shape: ChunkCoord,
}

impl ChunkSpec {
    /// Checks that all axes have nonzero length, and that the given shape matches the data length.
    pub fn new(chunk_idxs: Vec<ChunkIndex>, shape: ChunkCoord) -> Result<Self, ChunkSpecError> {
        ChunkSpecError::check_data(chunk_idxs.len(), &shape)?;
        Ok(Self::new_unchecked(chunk_idxs, shape))
    }

    pub fn from_shard<R: Read + Seek>(
        r: &mut R,
        shape: ChunkCoord,
    ) -> Result<Self, ChunkSpecConstructionError> {
        let offset: i64 = shape.iter().fold(-1, |acc, x| acc * *x as i64);
        r.seek(SeekFrom::End(offset))?;
        Self::from_reader(r, shape)
    }

    pub fn from_reader<R: Read + Seek>(
        r: &mut R,
        shape: ChunkCoord,
    ) -> Result<Self, ChunkSpecConstructionError> {
        let len: usize = shape.iter().fold(1, |acc, x| acc * *x as usize);
        let mut data = Vec::with_capacity(len);
        for _ in 0..len {
            let c = ChunkIndex::from_reader(r)?;
            data.push(c);
        }

        Self::new(data, shape).map_err(|e| e.into())
    }

    pub fn write<W: Write>(&self, w: &mut W) -> Result<(), std::io::Error> {
        for c in self.chunk_idxs.iter() {
            c.write(w)?;
        }
        Ok(())
    }

    /// Offset from end of shard, in bytes
    pub fn offset(&self) -> isize {
        -16 * self.chunk_idxs.len() as isize
    }

    /// Skips checks.
    pub fn new_unchecked(chunk_idxs: Vec<ChunkIndex>, shape: ChunkCoord) -> Self {
        Self { chunk_idxs, shape }
    }

    pub fn get_idx(&self, idx: &ChunkCoord) -> Result<Option<&ChunkIndex>, DimensionMismatch> {
        Ok(to_linear_idx(idx, &self.shape)?.and_then(|t| self.chunk_idxs.get(t)))
    }

    pub fn set_idx(
        &mut self,
        idx: &ChunkCoord,
        chunk_idx: ChunkIndex,
    ) -> Result<ChunkIndex, ChunkSpecModificationError> {
        let lin_idx = to_linear_idx(idx, &self.shape)?.ok_or_else(|| {
            ChunkSpecModificationError::OutOfBounds {
                coord: idx.clone(),
                shape: self.shape.clone(),
            }
        })?;
        Ok(std::mem::replace(&mut self.chunk_idxs[lin_idx], chunk_idx))
    }

    pub fn get_first_gap(&self, min_size: usize) -> usize {
        let mut idxs: Vec<_> = self
            .chunk_idxs
            .iter()
            .filter_map(|c| Some((c.offset as usize, c.end_offset()? as usize)))
            .collect();
        if idxs.is_empty() {
            return 0;
        }

        idxs.sort_unstable_by_key(|p| p.0);

        for w in idxs.windows(2) {
            let (_, l_end) = w[0];
            let (r_start, _) = w[1];
            if r_start - l_end > min_size {
                return l_end;
            }
        }

        idxs.last().unwrap().1
    }
}
