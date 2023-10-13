use crc32c::crc32c;
use serde::{Deserialize, Serialize};

use std::io::{BufWriter, Cursor, Read, Seek};
use thiserror::Error;

use crate::chunk_arr::{offset_shape_to_slice_info, ChunkIter};
use crate::codecs::aa::AACodecType;
use crate::codecs::bb::BBCodecType;
use crate::codecs::{ArrayRepr, CodecChain};
use crate::data_type::ReflectedType;
use crate::{ArcArrayD, GridCoord, MaybeNdim, Ndim};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::{SeekFrom, Write};

use super::{ABCodec, ABCodecType};

#[derive(Clone, Serialize, Deserialize, Debug, PartialEq)]
pub struct ShardingIndexedCodec {
    pub chunk_shape: GridCoord,
    pub codecs: CodecChain,
}

impl Ndim for ShardingIndexedCodec {
    fn ndim(&self) -> usize {
        self.chunk_shape.len()
    }
}

impl ShardingIndexedCodec {
    pub fn new<C: Into<GridCoord>>(chunk_shape: C) -> Self {
        Self {
            chunk_shape: chunk_shape.into(),
            codecs: CodecChain::default(),
        }
    }

    pub fn n_chunks(&self, shard_shape: &[u64]) -> Result<Vec<u64>, &'static str> {
        self.chunk_shape
            .iter()
            .zip(shard_shape.iter())
            .map(|(c, s)| {
                if s % c != 0 {
                    return Err("Shard shape does not match sub-chunks");
                }
                Ok(s / c)
            })
            .collect()
    }

    /// Set the array->bytes codec.
    ///
    /// By default, uses a little-[crate::codecs::ab::endian::EndianCodec].
    ///
    /// Replaces an existing AB codec.
    /// Fails if the dimensions are not compatible with the array's shape.
    pub fn ab_codec<T: Into<ABCodecType>>(mut self, codec: T) -> Result<Self, &'static str> {
        let c = codec.into();
        self.union_ndim(&c)?;
        self.codecs.replace_ab_codec(Some(c));
        Ok(self)
    }

    /// Append an array->array codec.
    ///
    /// This will be the last AA encoder, or first AA decoder.
    ///
    /// Fails if the dimensions are not compatible with the array's shape.
    pub fn push_aa_codec<T: Into<AACodecType>>(mut self, codec: T) -> Result<Self, &'static str> {
        let c = codec.into();
        self.union_ndim(&c)?;
        self.codecs.aa_codecs_mut().push(c);
        Ok(self)
    }

    /// Append a bytes->bytes codec.
    ///
    /// This will be the last BB encoder, or first BB decoder.
    pub fn push_bb_codec<T: Into<BBCodecType>>(mut self, codec: T) -> Self {
        let c = codec.into();
        // todo: check blosc type size
        self.codecs.bb_codecs_mut().push(c);
        self
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

impl ABCodec for ShardingIndexedCodec {
    fn encode<T: ReflectedType, W: Write>(&self, decoded: ArcArrayD<T>, w: W) {
        let mut bw = BufWriter::new(w);
        let mut curs = Cursor::new(Vec::default());

        let dec_shape: GridCoord = decoded.shape().iter().map(|s| *s as u64).collect();
        let mut offset: u64 = 0;

        let mut addrs = Vec::default();
        for c_info in ChunkIter::new_strict(self.chunk_shape.clone(), dec_shape).unwrap() {
            let sl = offset_shape_to_slice_info(&c_info.offset, &c_info.shape);
            // todo: is this a clone which can be avoided?
            let sub_arr = decoded.slice(sl).to_shared();
            self.codecs.encode(sub_arr, &mut curs);
            let nbytes = curs.position();
            bw.write(&curs.get_ref()[..(nbytes as usize)])
                .expect("Could not write sub-chunk");
            addrs.push(ChunkAddress {
                offset: offset.clone(),
                nbytes: nbytes.clone(),
            });
            offset += nbytes;
            curs.set_position(0);
        }

        for addr in addrs {
            addr.write_to(&mut bw).expect("Could not write chunk addr");
        }

        bw.flush()
            .expect("Could not write shard to underlying buffer");
    }

    fn decode<T: ReflectedType, R: Read>(
        &self,
        mut r: R,
        decoded_repr: ArrayRepr<T>,
    ) -> ArcArrayD<T> {
        let shape: Vec<_> = decoded_repr.shape.iter().map(|s| *s as usize).collect();
        let mut arr = decoded_repr.empty_array();
        let mut chunk_buf = Vec::default();
        r.read_to_end(&mut chunk_buf).expect("Could not read");
        let chunk_len = chunk_buf.len();
        let mut curs = Cursor::new(chunk_buf);

        let n_chunks = shape
            .iter()
            .zip(self.chunk_shape.iter())
            .map(|(a_s, c_s)| *a_s as u64 / c_s)
            .collect();
        let cspec =
            ChunkSpec::from_shard(&mut curs, n_chunks).expect("Could not construct chunk spec");

        let total_chunks = cspec.n_subchunks();

        let mut subchunk_buf: Vec<u8> = Vec::default();

        for c_info in ChunkIter::new_strict(self.chunk_shape.clone(), decoded_repr.shape)
            .expect("Could not iterate shard chunks")
        {
            let addr = cspec.get_idx(&c_info.chunk_idx).unwrap().unwrap();

            if addr.is_empty() {
                continue;
            }

            // this prevents a bad chunk address trying to allocate all our RAM
            let nbytes = (addr.nbytes as usize).min(
                chunk_len
                    - total_chunks * std::mem::size_of::<ChunkAddress>()
                    - addr.offset as usize,
            );

            if subchunk_buf.len() < nbytes {
                // safety factor of 2 to reduce repeated resizes.
                // Resize is usually fast but might have to re-allocate
                subchunk_buf.resize(nbytes * 2, 0);
            }
            curs.seek(SeekFrom::Start(addr.offset))
                .expect("Could not seek");
            curs.read_exact(&mut subchunk_buf[..nbytes])
                .expect("Could not read sub-chunk");

            let sub_arr = self.codecs.decode::<T, _>(
                &subchunk_buf[..nbytes],
                ArrayRepr {
                    shape: c_info.shape.clone(),
                    fill_value: decoded_repr.fill_value,
                },
            );

            let sl = offset_shape_to_slice_info(&c_info.offset, &c_info.shape);
            let mut view = arr.slice_mut(sl);
            view.assign(&sub_arr);
        }
        arr
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ChunkAddress {
    pub offset: u64,
    pub nbytes: u64,
}

// todo: replace with Option<ChunkIndex>?
impl ChunkAddress {
    pub fn is_empty(&self) -> bool {
        self.offset == u64::MAX && self.nbytes == u64::MAX
    }

    pub fn nbytes() -> usize {
        std::mem::size_of::<u64>() + std::mem::size_of::<u64>()
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

    pub fn write_to<W: Write>(&self, w: &mut W) -> Result<(), std::io::Error> {
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

impl PartialOrd for ChunkAddress {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ChunkAddress {
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
    fn check_data(data_len: usize, shape: &GridCoord) -> Result<(), Self> {
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
    #[error("Chunk spec does not match checksum")]
    ChecksumFailure,
}

/// C order
fn to_linear_idx(coord: &GridCoord, shape: &GridCoord) -> Result<Option<usize>, DimensionMismatch> {
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
    OutOfBounds { coord: GridCoord, shape: GridCoord },
    #[error("Dimension mismatch")]
    DimensionMismatch(#[from] DimensionMismatch),
}

pub struct ChunkSpec {
    chunk_idxs: Vec<ChunkAddress>,
    shape: GridCoord,
}

impl ChunkSpec {
    /// Checks that all axes have nonzero length, and that the given shape matches the data length.
    pub fn new(chunk_idxs: Vec<ChunkAddress>, shape: GridCoord) -> Result<Self, ChunkSpecError> {
        ChunkSpecError::check_data(chunk_idxs.len(), &shape)?;
        Ok(Self::new_unchecked(chunk_idxs, shape))
    }

    /// From a [Seek]able [Read]er representing a whole shard.
    pub fn from_shard<R: Read + Seek>(
        r: &mut R,
        shape: GridCoord,
    ) -> Result<Self, ChunkSpecConstructionError> {
        let prod = shape.iter().fold(1, |a, b| a * b);
        if prod == 0 {
            Ok(Self::new_unchecked(vec![], shape))
        } else {
            let chksum_len = std::mem::size_of::<u32>() as i64;
            let offset = -(prod as i64) * std::mem::size_of::<ChunkAddress>() as i64 - chksum_len;
            r.seek(SeekFrom::End(offset))?;
            Self::from_reader(r, shape)
        }
    }

    /// From a [Read]er representing the footer at the end of a shard.
    pub fn from_reader<R: Read>(
        r: &mut R,
        shape: GridCoord,
    ) -> Result<Self, ChunkSpecConstructionError> {
        let n_c_addrs: usize = shape.iter().fold(1, |acc, x| acc * *x as usize);
        let chksum_len = std::mem::size_of::<u32>();
        let buf_len = n_c_addrs * std::mem::size_of::<ChunkAddress>() + chksum_len;
        let mut buf = vec![u8::MAX; buf_len];
        r.read_exact(&mut buf)?;
        let chksum_offset = buf.len() - chksum_len;
        let chksum_calc = crc32c(&buf[..chksum_offset]);

        let mut c_idxs = Vec::with_capacity(n_c_addrs);
        let mut curs = Cursor::new(buf);
        for _ in 0..n_c_addrs {
            let c = ChunkAddress::from_reader(&mut curs)?;
            c_idxs.push(c);
        }

        let chksum_read = curs.read_u32::<LittleEndian>()?;
        if chksum_calc != chksum_read {
            Self::new(c_idxs, shape).map_err(|e| e.into())
        } else {
            Err(ChunkSpecConstructionError::ChecksumFailure)
        }
    }

    pub fn write_to<W: Write>(&self, w: &mut W) -> Result<(), std::io::Error> {
        let mut curs = Cursor::new(Vec::default());
        for c in self.chunk_idxs.iter() {
            c.write_to(&mut curs)?;
        }
        let chksum = crc32c(&curs.get_ref()[..]);
        curs.write_u32::<LittleEndian>(chksum)?;
        w.write_all(&curs.into_inner()[..])?;

        Ok(())
    }

    /// Offset from end of shard, in bytes
    pub fn offset(&self) -> isize {
        -16 * self.chunk_idxs.len() as isize
    }

    /// Skips checks.
    pub fn new_unchecked(chunk_idxs: Vec<ChunkAddress>, shape: GridCoord) -> Self {
        Self { chunk_idxs, shape }
    }

    pub fn get_idx(&self, idx: &GridCoord) -> Result<Option<&ChunkAddress>, DimensionMismatch> {
        Ok(to_linear_idx(idx, &self.shape)?.and_then(|t| self.chunk_idxs.get(t)))
    }

    pub fn set_idx(
        &mut self,
        idx: &GridCoord,
        chunk_idx: ChunkAddress,
    ) -> Result<ChunkAddress, ChunkSpecModificationError> {
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

    pub fn n_subchunks(&self) -> usize {
        self.chunk_idxs.len()
    }
}

#[cfg(test)]
mod tests {
    use crate::codecs::{aa::TransposeCodec, ab::endian::EndianCodec};

    use super::*;
    use smallvec::smallvec;

    fn make_arr() -> ArcArrayD<i32> {
        ArcArrayD::from_shape_vec(vec![50, 60], (0..50 * 60).collect()).unwrap()
    }

    #[test]
    fn roundtrip_shard_simple() {
        let codec = ShardingIndexedCodec::new(smallvec![10, 20]);
        let arr = make_arr();
        let arr1 = arr.clone();
        let mut buf = Cursor::new(Vec::<u8>::default());
        codec.encode(arr, &mut buf);

        buf.set_position(0);
        let arr2 = codec.decode::<i32, _>(&mut buf, ArrayRepr::new(vec![50, 60].as_slice(), 0i32));

        assert_eq!(arr1, arr2);
    }

    #[cfg(feature = "gzip")]
    #[test]
    fn roundtrip_shard_complex() {
        use crate::codecs::bb::gzip_codec::GzipCodec;

        let codec = ShardingIndexedCodec::new(smallvec![10, 20])
            .push_aa_codec(TransposeCodec::new_f())
            .unwrap()
            .ab_codec(EndianCodec::new_big())
            .unwrap()
            .push_bb_codec(GzipCodec::default());

        let arr = make_arr();
        let arr1 = arr.clone();
        let mut buf = Cursor::new(Vec::<u8>::default());
        codec.encode(arr, &mut buf);

        buf.set_position(0);
        let arr2 = codec.decode::<i32, _>(&mut buf, ArrayRepr::new(vec![50, 60].as_slice(), 0i32));

        assert_eq!(arr1, arr2);
    }
}
