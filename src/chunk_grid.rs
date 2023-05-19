use ndarray::{IxDyn, SliceInfo, SliceInfoElem};
use serde::{Deserialize, Serialize};

use crate::{
    chunk_arr::{CIter, PartialChunkIter},
    codecs::ab::sharding_indexed::DimensionMismatch,
    CoordVec, GridCoord, MaybeNdim, Ndim,
};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ArraySlice {
    pub offset: u64,
    pub shape: u64,
}

impl ArraySlice {
    pub fn new(offset: u64, shape: u64) -> Self {
        Self { offset, shape }
    }

    pub fn from_max(offset: u64, max_shape: u64) -> Option<Self> {
        if offset > max_shape {
            None
        } else {
            Some(Self::new(offset, max_shape - offset))
        }
    }

    pub fn end(&self) -> u64 {
        self.offset + self.shape
    }

    pub fn limit_extent(&self, max: u64) -> Option<Self> {
        Self::from_max(self.offset, max.min(self.end()))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArrayRegion(CoordVec<ArraySlice>);

impl FromIterator<ArraySlice> for ArrayRegion {
    fn from_iter<T: IntoIterator<Item = ArraySlice>>(iter: T) -> Self {
        Self(iter.into_iter().collect())
    }
}

impl Ndim for ArrayRegion {
    fn ndim(&self) -> usize {
        self.0.len()
    }
}

impl ArrayRegion {
    pub fn at_origin(&self) -> Self {
        self.0
            .iter()
            .map(|sl| ArraySlice::new(0, sl.shape))
            .collect()
    }

    pub fn is_whole_unchecked(&self, shape: &[u64]) -> bool {
        self.0
            .iter()
            .zip(shape.iter())
            .all(|(sl, sh)| sl.offset == 0 && &sl.shape == sh)
    }

    pub fn is_whole(&self, shape: &[u64]) -> bool {
        DimensionMismatch::check_coords(shape.len(), self.ndim()).unwrap();
        self.is_whole_unchecked(shape)
    }

    pub fn from_offset_shape_unchecked(offset: &[u64], shape: &[u64]) -> Self {
        offset
            .iter()
            .zip(shape.iter())
            .map(|(o, s)| ArraySlice::new(*o, *s))
            .collect()
    }

    pub fn from_offset_shape(offset: &[u64], shape: &[u64]) -> Self {
        DimensionMismatch::check_coords(offset.len(), shape.len()).unwrap();
        Self::from_offset_shape_unchecked(offset, shape)
    }

    /// Panics if dimensions are inconsistent.
    pub fn from_max(offset: &[u64], max: &[u64]) -> Option<Self> {
        DimensionMismatch::check_coords(offset.len(), max.len()).unwrap();
        let mut slices: CoordVec<ArraySlice> = CoordVec::with_capacity(offset.len());
        for (o, m) in offset.iter().zip(max.iter()) {
            slices.push(ArraySlice::from_max(*o, *m)?);
        }
        Some(Self(slices))
    }

    pub fn offset(&self) -> GridCoord {
        self.0.iter().map(|s| s.offset).collect()
    }

    pub fn shape(&self) -> GridCoord {
        self.0.iter().map(|s| s.shape).collect()
    }

    pub fn end(&self) -> GridCoord {
        self.0.iter().map(|s| s.end()).collect()
    }

    pub fn numel(&self) -> Option<u64> {
        self.0.iter().map(|s| s.shape).reduce(|a, b| a * b)
    }

    /// Panics if max has incorrect dimensionality.
    pub fn limit_extent(&self, max: &[u64]) -> Option<Self> {
        DimensionMismatch::check_coords(max.len(), self.ndim()).unwrap();
        self.limit_extent_unchecked(max)
    }

    pub fn limit_extent_unchecked(&self, max: &[u64]) -> Option<Self> {
        let mut slices: CoordVec<ArraySlice> = CoordVec::with_capacity(self.ndim());
        for (sl, mx) in self.0.iter().zip(max.iter()) {
            slices.push(sl.limit_extent(*mx)?);
        }
        Some(Self(slices))
    }

    pub fn slice_info(&self) -> SliceInfo<Vec<SliceInfoElem>, IxDyn, IxDyn> {
        let indices: Vec<_> = self
            .0
            .iter()
            .map(|sl| SliceInfoElem::Slice {
                start: sl.offset as isize,
                end: Some(sl.end() as isize),
                step: 1,
            })
            .collect();
        SliceInfo::try_from(indices).expect("Bad index size")
    }
}

pub struct PartialChunk {
    pub chunk_idx: GridCoord,
    pub chunk_region: ArrayRegion,
    pub out_region: ArrayRegion,
}

impl PartialChunk {
    pub fn new(chunk_idx: GridCoord, chunk_region: ArrayRegion, out_region: ArrayRegion) -> Self {
        DimensionMismatch::check_many(chunk_idx.len(), &[chunk_region.ndim(), out_region.ndim()])
            .unwrap();
        Self::new_unchecked(chunk_idx, chunk_region, out_region)
    }

    pub fn new_unchecked(
        chunk_idx: GridCoord,
        chunk_region: ArrayRegion,
        out_region: ArrayRegion,
    ) -> Self {
        Self {
            chunk_idx,
            chunk_region,
            out_region,
        }
    }
}

pub trait ChunkGrid: MaybeNdim {
    /// Calculate the chunk index where the voxel exists, and its offset within that chunk.
    ///
    /// Panics if dimensions mismatch.
    fn voxel_chunk(&self, idx: &[u64]) -> (GridCoord, GridCoord) {
        if let Some(d) = self.maybe_ndim() {
            DimensionMismatch::check_coords(idx.len(), d).unwrap()
        }
        self.voxel_chunk_unchecked(idx)
    }

    fn voxel_chunk_unchecked(&self, idx: &[u64]) -> (GridCoord, GridCoord);

    /// Calculate the shape of a given chunk.
    ///
    /// Panics if dimensions mismatch.
    fn chunk_shape(&self, idx: &[u64]) -> GridCoord {
        if let Some(d) = self.maybe_ndim() {
            DimensionMismatch::check_coords(idx.len(), d).unwrap();
        }
        self.chunk_shape_unchecked(idx)
    }

    fn chunk_shape_unchecked(&self, idx: &[u64]) -> GridCoord;

    /// Calculate how regions of chunks map into a given array region.
    ///
    /// Panics if dimensions mismatch.
    fn chunks_in_region(&self, region: &ArrayRegion) -> PartialChunkIter {
        if let Some(d) = self.maybe_ndim() {
            DimensionMismatch::check_coords(region.ndim(), d).unwrap();
        }
        self.chunks_in_region_unchecked(region)
    }

    fn chunks_in_region_unchecked(&self, region: &ArrayRegion) -> PartialChunkIter;
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RegularChunkGrid {
    chunk_shape: GridCoord,
}

impl RegularChunkGrid {
    fn new<T: Into<GridCoord>>(chunk_shape: T) -> Self {
        let chunk_shape = chunk_shape.into();
        Self { chunk_shape }
    }
}

impl Ndim for RegularChunkGrid {
    fn ndim(&self) -> usize {
        self.chunk_shape.len()
    }
}

impl ChunkGrid for RegularChunkGrid {
    fn chunk_shape_unchecked(&self, _idx: &[u64]) -> GridCoord {
        self.chunk_shape.clone()
    }

    fn voxel_chunk_unchecked(&self, idx: &[u64]) -> (GridCoord, GridCoord) {
        let mut chunk_idx = GridCoord::with_capacity(self.ndim());
        let mut offset = GridCoord::with_capacity(self.ndim());

        for (vx, cs) in idx.iter().zip(self.chunk_shape.iter()) {
            chunk_idx.push(vx / cs);
            offset.push(vx % cs);
        }
        (chunk_idx, offset)
    }

    fn chunks_in_region_unchecked(&self, region: &ArrayRegion) -> PartialChunkIter {
        let (min_chunk, min_offset) = self.voxel_chunk(region.offset().as_slice());
        let (max_chunk, max_offset) = self.voxel_chunk(region.end().as_slice());

        PartialChunkIter::new(
            min_chunk,
            min_offset,
            max_chunk,
            max_offset,
            self.chunk_shape.clone(),
        )
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(tag = "name", content = "configuration", rename_all = "lowercase")]
#[enum_delegate::implement(MaybeNdim)]
pub enum ChunkGridType {
    Regular(RegularChunkGrid),
}

impl ChunkGrid for ChunkGridType {
    fn voxel_chunk_unchecked(&self, idx: &[u64]) -> (GridCoord, GridCoord) {
        match self {
            Self::Regular(g) => g.voxel_chunk_unchecked(idx),
        }
    }

    fn chunk_shape_unchecked(&self, idx: &[u64]) -> GridCoord {
        match self {
            Self::Regular(g) => g.chunk_shape_unchecked(idx),
        }
    }

    fn chunks_in_region_unchecked(&self, region: &ArrayRegion) -> PartialChunkIter {
        match self {
            Self::Regular(g) => g.chunks_in_region_unchecked(region),
        }
    }
}

impl From<&[u64]> for ChunkGridType {
    fn from(value: &[u64]) -> Self {
        let cs: GridCoord = value.iter().cloned().collect();
        Self::Regular(RegularChunkGrid::new(cs))
    }
}

impl ChunkGridType {
    pub fn chunk_shape(&self, _idx: &GridCoord) -> GridCoord {
        match self {
            Self::Regular(g) => g.chunk_shape.clone(),
        }
    }
}
