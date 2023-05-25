use std::{
    fmt::Display,
    io::SeekFrom,
    ops::{Add, Range},
};

use ndarray::{ArcArray, IxDyn};
use smallvec::SmallVec;

mod chunk_arr;
mod chunk_grid;
pub mod chunk_key_encoding;
pub mod codecs;
mod data_type;
pub mod node;
pub mod prelude;
pub mod store;
mod util;

const COORD_SMALLVEC_SIZE: usize = 6;
pub const ZARR_FORMAT: usize = 3;

pub type CoordVec<T> = SmallVec<[T; COORD_SMALLVEC_SIZE]>;
// todo: split into VoxelCoord, ChunkCoord, both usize?
pub type GridCoord = CoordVec<u64>;
pub type ArcArrayD<T> = ArcArray<T, IxDyn>;

fn to_usize(coord: &[u64]) -> CoordVec<usize> {
    coord.iter().map(|n| *n as usize).collect()
}

#[enum_delegate::register]
pub trait Ndim {
    fn ndim(&self) -> usize;

    fn same_ndim<T: Ndim>(&self, other: &T) -> Result<usize, &'static str> {
        let n = self.ndim();
        if n == other.ndim() {
            Ok(n)
        } else {
            Err("Inconsistent dimensionalities")
        }
    }
}

#[enum_delegate::register]
pub trait MaybeNdim {
    fn maybe_ndim(&self) -> Option<usize>;

    fn union_ndim<T: MaybeNdim>(&self, other: &T) -> Result<Option<usize>, &'static str> {
        if let Some(n1) = self.maybe_ndim() {
            if let Some(n2) = other.maybe_ndim() {
                if n1 == n2 {
                    return Ok(Some(n1));
                } else {
                    return Err("Inconsistent dimensionalities");
                }
            } else {
                return Ok(Some(n1));
            }
        } else {
            return Ok(other.maybe_ndim());
        }
    }

    fn validate_ndim(&self) -> Result<(), &'static str> {
        Ok(())
    }
}

impl<T: Ndim> MaybeNdim for T {
    fn maybe_ndim(&self) -> Option<usize> {
        Some(self.ndim())
    }
}

// could be generic <T: PartialOrd + Add>
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RangeRequest {
    Range { offset: usize, size: Option<usize> },
    Suffix(usize),
}

impl RangeRequest {
    pub fn new_range(offset: usize, size: Option<usize>) -> Self {
        Self::Range { offset, size }
    }

    pub fn start(&self, len: Option<usize>) -> Option<usize> {
        match self {
            Self::Range { offset, size } => Some(*offset),
            Self::Suffix(s) => len.map(|l| l - s),
        }
    }

    pub fn end(&self, len: Option<usize>) -> Option<usize> {
        match self {
            Self::Range { offset, size } => size.map(|s| offset + s),
            Self::Suffix(s) => len,
        }
    }

    pub fn slice<'a, T>(&self, sl: &'a [T]) -> &'a [T] {
        &sl[self.to_range(sl.len())]
    }

    fn to_range(&self, len: usize) -> Range<usize> {
        let end = self.end(Some(len)).unwrap();
        match self {
            Self::Range { offset, size } => *offset..end,
            Self::Suffix(s) => {
                if &len < s {
                    0..end
                } else {
                    (len - s)..end
                }
            }
        }
    }

    pub fn slice_mut<'a, T>(&self, sl: &'a mut [T]) -> &'a mut [T] {
        // todo: what if start, stop are out of bounds?
        let len = sl.len();
        &mut sl[self.to_range(len)]
    }
}

impl Default for RangeRequest {
    fn default() -> Self {
        Self::Range {
            offset: 0,
            size: None,
        }
    }
}

impl Display for RangeRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RangeRequest::Range { offset, size } => {
                if let Some(s) = size {
                    f.write_fmt(format_args!("{}-{}", offset, offset + s))
                } else {
                    f.write_fmt(format_args!("{}-", offset))
                }
            }
            RangeRequest::Suffix(s) => f.write_fmt(format_args!("-{}", s)),
        }
    }
}
