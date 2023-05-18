use crate::GridCoord;
use ndarray::{IxDyn, SliceInfo, SliceInfoElem};
use smallvec::smallvec;

/// Iterate N dimensional indices in C order.
pub(crate) struct CIter {
    shape: GridCoord,
    next: Option<GridCoord>,
}

impl CIter {
    pub fn new(shape: GridCoord) -> Self {
        let next = if shape.len() == 0 || shape.iter().any(|s| 0 == *s) {
            None
        } else {
            Some(smallvec![0; shape.len()])
        };
        Self { shape, next }
    }
}

impl Iterator for CIter {
    type Item = GridCoord;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next.is_none() {
            return None;
        }
        let curr = self.next.clone();
        let mut finished = false;

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

impl Iterator for ChunkIter {
    type Item = ChunkIterOutput;

    fn next(&mut self) -> Option<Self::Item> {
        self.c_iter.next().map(|c| self.idx_to_output(c))
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
    SliceInfo::try_from(indices).expect("Bad index size size")
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
