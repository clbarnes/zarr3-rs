use std::io::{ErrorKind, Read, Seek, SeekFrom, Result, Cursor};

use serde::{Deserialize, Serialize};

use crate::MaybeNdim;
use crate::codecs::Interval;

#[cfg(feature = "blosc")]
mod blosc_codec;
#[cfg(feature = "gzip")]
mod gzip_codec;

const BUF_SIZE: usize = 8*1024;

fn discard_bytes<R: Read>(rdr: &mut R, n: usize) -> Result<usize> {
    let buf_len = n.min(BUF_SIZE);
    let mut buf = vec![0; buf_len];
    let mut total_read = 0;
    while n > total_read - buf_len {
        total_read += rdr.read(&mut buf)?;
    }
    total_read += rdr.read(&mut buf[..(total_read - buf_len)])?;
    Ok(total_read)
}

pub trait ByteReader {
    fn read(&mut self) -> Result<Vec<u8>>;

    // todo: multiple intervals
    fn partial_read(&mut self, interval: Interval) -> Result<Vec<u8>> {
        let whole = ByteReader::read(self)?;
        let start = if interval.start < 0 {
            whole.len() + interval.start as usize
        } else {
            interval.start as usize
        };
        let stop = if let Some(e) = interval.end {
            if e < 0 {
                whole.len() + e as usize
            } else {
                interval.start as usize
            }
        } else {
            whole.len()
        };
        if stop < start {
            Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, "end is before start"))
        } else {
            Ok((&whole[start..stop]).iter().cloned().collect())
        }
    }
}

// todo: possible to do this without Seek?
// Only with non-negative offsets in interval
impl<R: Read + Seek> ByteReader for R {
    fn read(&mut self) -> Result<Vec<u8>> {
        self.seek(SeekFrom::Start(0))?;
        let mut b = Vec::default();
        self.read_to_end(&mut b)?;
        Ok(b)
    }

    fn partial_read(&mut self, interval: Interval) -> Result<Vec<u8>> {
        let (from, nbytes) = interval.as_seekfrom_nbytes(Some(self.seek(SeekFrom::End(0))? as usize));
        self.seek(from)?;
        let mut buf;
        if let Some(n) = nbytes {
            buf = vec![0; n];
            self.read(&mut buf)?;
        } else {
            buf = Vec::default();
            self.read_to_end(&mut buf)?;
        }
        Ok(buf)
    }
}

/// A codec which encodes and decodes between bytestrings.
pub trait BBCodec {
    fn encode(&self, decoded: &[u8]) -> Vec<u8>;

    fn decode(&self, encoded: Box<dyn ByteReader>) -> Result<Box<dyn ByteReader>>;

    fn partial_decode(&self, encoded: Box<dyn ByteReader>, interval: Interval) -> Result<Box<dyn ByteReader>> {
        let mut decoded = self.decode(encoded)?;
        let buf = decoded.partial_read(interval)?;
        Ok(Box::new(Cursor::new(buf)))
    }
}

#[derive(Clone, Serialize, Deserialize, PartialEq, Debug)]
#[serde(rename_all = "lowercase", tag = "codec", content = "configuration")]
pub enum BBCodecType {
    #[cfg(feature = "blosc")]
    Blosc(blosc_codec::BloscCodec),
    #[cfg(feature = "gzip")]
    Gzip(gzip_codec::GzipCodec),
}

impl MaybeNdim for BBCodecType {
    fn maybe_ndim(&self) -> Option<usize>  {
        None
    }
}

impl BBCodec for BBCodecType {
    fn encode(&self, raw: &[u8]) -> Vec<u8> {
        match self {
            #[cfg(feature = "gzip")]
            Self::Gzip(c) => c.encode(raw),

            #[cfg(feature = "blosc")]
            Self::Blosc(c) => c.encode(raw),
        }
    }

    fn decode(&self, encoded: Box<dyn ByteReader>) -> Result<Box<dyn ByteReader>> {
        match self {
            #[cfg(feature = "gzip")]
            Self::Gzip(c) => c.decode(encoded),

            #[cfg(feature = "blosc")]
            Self::Blosc(c) => c.decode(encoded),
        }
    }
}

impl BBCodec for &[BBCodecType] {
    fn encode(&self, raw: &[u8]) -> Vec<u8> {
        // todo: must be a better way
        let mut it = self.iter();

        let mut v;

        if let Some(c) = it.next() {
            v = c.encode(raw);
        } else {
            return raw.into();
        }

        for c in it {
            v = c.encode(&v);
        }

        v
    }

    fn decode(&self, encoded: Box<dyn ByteReader>) -> Result<Box<dyn ByteReader>> {
        let mut it = self.iter().rev();

        let mut v;

        if let Some(c) = it.next() {
            v = c.decode(encoded)?;
        } else {
            return Ok(encoded);
        }

        for c in it {
            v = c.decode(v)?;
        }

        Ok(v)
    }

    // todo: partial_decode
}

macro_rules! compression_from_impl {
    ($variant:ident, $c_type:ty) => {
        impl std::convert::From<$c_type> for BBCodecType {
            fn from(c: $c_type) -> Self {
                BBCodecType::$variant(c)
            }
        }
    };
}

#[cfg(feature = "gzip")]
compression_from_impl!(Gzip, gzip_codec::GzipCodec);

#[cfg(feature = "blosc")]
compression_from_impl!(Blosc, blosc_codec::BloscCodec);

struct SubReader<'r, R: Read + Seek> {
    offset: u64,
    nbytes: u64,
    reader: &'r mut R,
}

pub enum SeekEnd {
    SeekFrom(SeekFrom),
    NBytes(u64),
}

impl<'r, R: Read + Seek> SubReader<'r, R> {
    pub fn new(reader: &'r mut R, start: SeekFrom, end: SeekEnd) -> std::io::Result<Self> {
        let orig_pos = reader.stream_position()?;

        // this could be done in fewer seeks for particular start/end combinations
        let nbytes = match end {
            SeekEnd::NBytes(n) => n,
            SeekEnd::SeekFrom(sf) => {
                let pos = reader.seek(sf)?;
                reader.seek(SeekFrom::Start(orig_pos))?;
                pos
            }
        };

        let offset = reader.seek(start)?;

        Ok(Self {
            offset,
            nbytes,
            reader,
        })
    }

    pub fn end_offset(&self) -> u64 {
        self.offset + self.nbytes
    }
}

impl<'r, R: Read + Seek> Read for SubReader<'r, R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let pos = self.stream_position()?;
        let max_len = (self.nbytes - pos) as usize;
        if buf.len() > max_len {
            self.reader.read(&mut buf[..max_len])
        } else {
            self.reader.read(buf)
        }
    }
}

impl<'r, R: Read + Seek> Seek for SubReader<'r, R> {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        let new_pos = match pos {
            SeekFrom::Start(o) => self.reader.seek(SeekFrom::Start(self.offset + o))?,
            SeekFrom::End(o) => self
                .reader
                .seek(SeekFrom::Start((self.end_offset() as i64 + o) as u64))?,
            SeekFrom::Current(o) => {
                let orig_pos = self.stream_position()?;
                let new_pos = self.reader.seek(SeekFrom::Current(o))?;
                if new_pos < self.offset {
                    let out = Err(std::io::Error::new(
                        ErrorKind::InvalidInput,
                        "Seeked before start of SubReader",
                    ));
                    self.reader.seek(SeekFrom::Start(orig_pos))?;
                    out?
                }
                new_pos
            }
        };
        Ok(new_pos)
    }
}
