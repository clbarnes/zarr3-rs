use std::io::SeekFrom;

use ndarray::{ArcArray, IxDyn};
use smallvec::SmallVec;

mod chunk_arr;
pub mod chunk_key_encoding;
pub mod codecs;
mod data_type;
pub mod metadata;
pub mod store;
mod util;

const COORD_SMALLVEC_SIZE: usize = 6;
pub const ZARR_FORMAT: usize = 3;

pub type CoordVec<T> = SmallVec<[T; COORD_SMALLVEC_SIZE]>;
pub type GridCoord = CoordVec<u64>;
pub type ArcArrayD<T> = ArcArray<T, IxDyn>;

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

pub enum Offset {
    Start(usize),
    End(usize),
}

impl Default for Offset {
    fn default() -> Self {
        Self::Start(0)
    }
}

impl Into<SeekFrom> for &Offset {
    fn into(self) -> SeekFrom {
        match self {
            Offset::Start(o) => SeekFrom::Start(*o as u64),
            Offset::End(o) => SeekFrom::End(-(*o as i64)),
        }
    }
}

pub struct ByteRange {
    offset: Offset,
    nbytes: Option<usize>,
}

impl ByteRange {
    pub fn new(offset: isize, nbytes: Option<usize>) -> Self {
        let o = if offset < 0 {
            Offset::End((offset.abs()) as usize)
        } else {
            Offset::Start(offset as usize)
        };
        Self { offset: o, nbytes }
    }

    pub fn slice<'a, T>(&self, sl: &'a [T]) -> &'a [T] {
        // todo: what if start, stop are out of bounds?
        let start = match &self.offset {
            Offset::Start(o) => *o,
            Offset::End(o) => sl.len() - o,
        };
        if let Some(n) = self.nbytes.as_ref() {
            &sl[start..start + n]
        } else {
            &sl[start..]
        }
    }

    pub fn slice_mut<'a, T>(&self, sl: &'a mut [T]) -> &'a mut [T] {
        // todo: what if start, stop are out of bounds?
        let start = match &self.offset {
            Offset::Start(o) => *o,
            Offset::End(o) => sl.len() - o,
        };
        if let Some(n) = self.nbytes.as_ref() {
            &mut sl[start..start + n]
        } else {
            &mut sl[start..]
        }
    }
}

impl Default for ByteRange {
    fn default() -> Self {
        Self {
            offset: Offset::default(),
            nbytes: None,
        }
    }
}
