use smallvec::SmallVec;

mod chunk_arr;
pub mod chunk_key_encoding;
pub mod codecs;
mod data_type;
pub mod metadata;
mod util;

const COORD_SMALLVEC_SIZE: usize = 6;
pub const ZARR_FORMAT: usize = 3;

pub type CoordVec<T> = SmallVec<[T; COORD_SMALLVEC_SIZE]>;
pub type GridCoord = CoordVec<u64>;

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
