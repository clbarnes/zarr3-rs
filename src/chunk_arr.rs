use crate::{
    chunk_grid::{ArrayRegion, ArraySlice, PartialChunk},
    CoordVec, GridCoord, Ndim,
};
use ndarray::{IxDyn, SliceInfo, SliceInfoElem};
use smallvec::smallvec;

/// Iterate N dimensional indices in C order.
pub(crate) struct CIter {
    shape: GridCoord,
    next: Option<GridCoord>,
    total_size: usize,
    count: usize,
}

impl CIter {
    pub fn new(shape: GridCoord) -> Self {
        let next = if shape.is_empty() || shape.iter().any(|s| 0 == *s) {
            None
        } else {
            Some(smallvec![0; shape.len()])
        };
        let total_size = shape.iter().product::<u64>() as usize;
        Self {
            shape,
            next,
            total_size,
            count: 0,
        }
    }
}

impl Ndim for CIter {
    fn ndim(&self) -> usize {
        self.shape.len()
    }
}

impl Iterator for CIter {
    type Item = GridCoord;

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.total_size - self.count;
        (remaining, Some(remaining))
    }

    fn next(&mut self) -> Option<Self::Item> {
        self.next.as_ref()?;
        self.count += 1;

        let curr = self.next.clone();
        let mut finished = false;

        // this scope ends the mutable borrow of self.next
        {
            let c = self.next.as_mut().unwrap();

            for idx in (0..c.len()).rev() {
                if c[idx] + 1 == self.shape[idx] {
                    if idx == 0 {
                        finished = true;
                        break;
                    }
                    c[idx] = 0;
                } else {
                    c[idx] += 1;
                    break;
                }
            }
        }
        if finished {
            self.next = None;
        }

        curr
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ChunkIterOutput {
    pub chunk_idx: GridCoord,
    pub offset: GridCoord,
    pub shape: GridCoord,
}

pub(crate) struct ChunkIter {
    arr_shape: GridCoord,
    chunk_shape: GridCoord,
    c_iter: CIter,
}

impl Ndim for ChunkIter {
    fn ndim(&self) -> usize {
        self.c_iter.ndim()
    }
}

impl Iterator for ChunkIter {
    type Item = ChunkIterOutput;

    fn next(&mut self) -> Option<Self::Item> {
        self.c_iter.next().map(|c| self.idx_to_output(c))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.c_iter.size_hint()
    }
}

impl ChunkIter {
    pub fn new(chunk_shape: GridCoord, arr_shape: GridCoord) -> Result<Self, &'static str> {
        if chunk_shape.len() != arr_shape.len() {
            Err("Mismatching dimensionality")
        } else {
            let n_chunks = arr_shape
                .iter()
                .zip(chunk_shape.iter())
                .map(|(a, c)| {
                    let rem = a % c;
                    let div = a / c;
                    if rem == 0 {
                        div
                    } else {
                        div + 1
                    }
                })
                .collect();
            Ok(Self {
                arr_shape,
                chunk_shape,
                c_iter: CIter::new(n_chunks),
            })
        }
    }

    /// Checks that chunks exactly subdivide the array.
    pub fn new_strict(chunk_shape: GridCoord, arr_shape: GridCoord) -> Result<Self, &'static str> {
        if arr_shape
            .iter()
            .zip(chunk_shape.iter())
            .any(|(a, c)| a % c != 0)
        {
            return Err("Array is not an integer number of chunks");
        }
        Self::new(chunk_shape, arr_shape)
    }

    fn idx_to_output(&self, chunk_idx: GridCoord) -> ChunkIterOutput {
        let offset: GridCoord = chunk_idx
            .iter()
            .zip(self.chunk_shape.iter())
            .map(|(i, cs)| i * cs)
            .collect();

        let shape = offset
            .iter()
            .zip(self.chunk_shape.iter())
            .zip(self.arr_shape.iter())
            .map(
                |((o, c_s), a_s)| {
                    if o + c_s > *a_s {
                        a_s - o
                    } else {
                        *c_s
                    }
                },
            )
            .collect();

        ChunkIterOutput {
            chunk_idx,
            offset,
            shape,
        }
    }
}

pub struct PartialChunkIter {
    min_chunk: GridCoord,
    min_chunk_offset: GridCoord,
    max_chunk: GridCoord,
    max_chunk_offset: GridCoord,
    chunk_shape: GridCoord,
    c_iter: CIter,
}

impl PartialChunkIter {
    pub fn new(
        min_chunk: GridCoord,
        min_chunk_offset: GridCoord,
        max_chunk: GridCoord,
        max_chunk_offset: GridCoord,
        chunk_shape: GridCoord,
    ) -> Self {
        let shape: GridCoord = min_chunk
            .iter()
            .zip(max_chunk.iter())
            .map(|(mi, ma)| ma - mi + 1)
            .collect();
        let c_iter = CIter::new(shape);

        Self {
            min_chunk,
            min_chunk_offset,
            max_chunk,
            max_chunk_offset,
            chunk_shape,
            c_iter,
        }
    }
}

impl Ndim for PartialChunkIter {
    fn ndim(&self) -> usize {
        self.c_iter.ndim()
    }
}

impl Iterator for PartialChunkIter {
    type Item = PartialChunk;

    fn next(&mut self) -> Option<Self::Item> {
        let local_chunk_idx = self.c_iter.next()?;
        let mut chunk_idx = GridCoord::with_capacity(self.ndim());
        let mut chunk_slices = CoordVec::with_capacity(self.ndim());
        let mut out_slices = CoordVec::with_capacity(self.ndim());

        for d in 0..self.ndim() {
            chunk_idx.push(self.min_chunk[d] + local_chunk_idx[d]);

            let (chunk_offset, out_offset) = if local_chunk_idx[d] == 0 {
                (self.min_chunk_offset[d], 0)
            } else {
                (
                    0,
                    self.chunk_shape[d] - self.min_chunk_offset[d]
                        + self.chunk_shape[d] * (local_chunk_idx[d] - 1),
                )
            };

            let chunk_shape = if chunk_idx[d] == self.max_chunk[d] {
                self.max_chunk_offset[d] - chunk_offset
            } else {
                self.chunk_shape[d] - chunk_offset
            };

            chunk_slices.push(ArraySlice::new(chunk_offset, chunk_shape));
            out_slices.push(ArraySlice::new(out_offset, chunk_shape));
        }

        Some(PartialChunk::new_unchecked(
            chunk_idx,
            ArrayRegion::from_iter(chunk_slices),
            ArrayRegion::from_iter(out_slices),
        ))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.c_iter.size_hint()
    }
}

pub fn offset_shape_to_slice_info(
    offset: &[u64],
    shape: &[u64],
) -> SliceInfo<Vec<SliceInfoElem>, IxDyn, IxDyn> {
    let indices: Vec<_> = offset
        .iter()
        .zip(shape.iter())
        .map(|(o, s)| SliceInfoElem::Slice {
            start: *o as isize,
            end: Some((o + s) as isize),
            step: 1,
        })
        .collect();
    SliceInfo::try_from(indices).expect("Bad index size")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn c_iter() {
        let shape = smallvec![2, 3];
        let v: Vec<_> = CIter::new(shape).collect();
        let expected: Vec<GridCoord> = vec![
            smallvec![0, 0],
            smallvec![0, 1],
            smallvec![0, 2],
            smallvec![1, 0],
            smallvec![1, 1],
            smallvec![1, 2],
        ];
        assert_eq!(v, expected)
    }

    #[test]
    fn chunk_iter() {
        let c_shape = smallvec![2, 3];
        let a_shape = smallvec![6, 6];
        let v: Vec<_> = ChunkIter::new_strict(c_shape.clone(), a_shape.clone())
            .unwrap()
            .collect();

        let expected: Vec<ChunkIterOutput> = vec![
            ChunkIterOutput {
                chunk_idx: smallvec![0, 0],
                offset: smallvec![0, 0],
                shape: smallvec![2, 3],
            },
            ChunkIterOutput {
                chunk_idx: smallvec![0, 1],
                offset: smallvec![0, 3],
                shape: smallvec![2, 3],
            },
            ChunkIterOutput {
                chunk_idx: smallvec![1, 0],
                offset: smallvec![2, 0],
                shape: smallvec![2, 3],
            },
            ChunkIterOutput {
                chunk_idx: smallvec![1, 1],
                offset: smallvec![2, 3],
                shape: smallvec![2, 3],
            },
            ChunkIterOutput {
                chunk_idx: smallvec![2, 0],
                offset: smallvec![4, 0],
                shape: smallvec![2, 3],
            },
            ChunkIterOutput {
                chunk_idx: smallvec![2, 1],
                offset: smallvec![4, 3],
                shape: smallvec![2, 3],
            },
        ];
        assert_eq!(v, expected)
    }
}
