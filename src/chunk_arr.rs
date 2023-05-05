use crate::ChunkCoord;
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::{Read, Seek, SeekFrom, Write};
use thiserror::Error;

use crate::codecs::ab::sharding_indexed::DimensionMismatch;

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
