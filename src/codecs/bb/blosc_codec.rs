use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::io::{self, Cursor, Read, Write};
use thiserror::Error;

use crate::{codecs::bb::BBCodec, data_type::ReflectedType};
use blosc::{decompress_bytes, Context};
pub use blosc::{Clevel, Compressor, ShuffleMode};

use crate::codecs::fwrite::{FinalWrite, FinalWriter};

#[derive(Clone, Serialize, Deserialize, Debug, PartialEq, Eq)]
pub struct BloscCodec {
    #[serde(deserialize_with = "cname_from_str", serialize_with = "cname_to_str")]
    pub cname: Compressor,
    #[serde(deserialize_with = "clevel_from_str", serialize_with = "clevel_to_str")]
    pub clevel: Clevel,
    #[serde(with = "shuffle")]
    pub shuffle: ShuffleMode,
    pub blocksize: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub typesize: Option<usize>,
}

fn cname_from_str<'de, D>(deserializer: D) -> Result<Compressor, D::Error>
where
    D: Deserializer<'de>,
{
    match Deserialize::deserialize(deserializer)? {
        "lz4" => Ok(Compressor::LZ4),
        "lz4hc" => Ok(Compressor::LZ4HC),
        "blosclz" => Ok(Compressor::BloscLZ),
        "zstd" => Ok(Compressor::Zstd),
        "snappy" => Ok(Compressor::Snappy),
        "zlib" => Ok(Compressor::Zlib),
        _ => Err(serde::de::Error::custom("bad cname")),
    }
}

fn cname_to_str<S>(cname: &Compressor, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    match cname {
        Compressor::LZ4 => serializer.serialize_str("lz4"),
        Compressor::LZ4HC => serializer.serialize_str("lz4hc"),
        Compressor::BloscLZ => serializer.serialize_str("blosclz"),
        Compressor::Zstd => serializer.serialize_str("zstd"),
        Compressor::Snappy => serializer.serialize_str("snappy"),
        Compressor::Zlib => serializer.serialize_str("zlib"),
        Compressor::Invalid => Err(serde::ser::Error::custom("bad cname")),
    }
}

fn clevel_from_str<'de, D>(deserializer: D) -> Result<Clevel, D::Error>
where
    D: Deserializer<'de>,
{
    match Deserialize::deserialize(deserializer)? {
        0 => Ok(Clevel::None),
        1 => Ok(Clevel::L1),
        2 => Ok(Clevel::L2),
        3 => Ok(Clevel::L3),
        4 => Ok(Clevel::L4),
        5 => Ok(Clevel::L5),
        6 => Ok(Clevel::L6),
        7 => Ok(Clevel::L7),
        8 => Ok(Clevel::L8),
        9 => Ok(Clevel::L9),
        _ => Err(serde::de::Error::custom("bad clevel")),
    }
}

fn clevel_to_str<S>(cname: &Clevel, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    serializer.serialize_i32(*cname as i32)
}

mod shuffle {
    use blosc::ShuffleMode;
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn deserialize<'de, D>(deserializer: D) -> Result<ShuffleMode, D::Error>
    where
        D: Deserializer<'de>,
    {
        match Deserialize::deserialize(deserializer)? {
            "noshuffle" => Ok(ShuffleMode::None),
            "shuffle" => Ok(ShuffleMode::Byte),
            "bitshuffle" => Ok(ShuffleMode::Bit),
            s => Err(serde::de::Error::custom(&format!(
                "Unknown blosc shuffle \"{}\"",
                s
            ))),
        }
    }

    pub fn serialize<S>(cname: &ShuffleMode, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match cname {
            ShuffleMode::None => serializer.serialize_str("noshuffle"),
            ShuffleMode::Byte => serializer.serialize_str("shuffle"),
            ShuffleMode::Bit => serializer.serialize_str("bitshuffle"),
        }
    }
}

#[derive(Error, Debug)]
pub enum BloscBuildError {
    #[error("`typesize` must not be None if blosc codec shuffling is active (here `{0:?}`)")]
    TypesizeNeeded(ShuffleMode),
    #[error("Compressor not available in blosc: `{0:?}`")]
    UnavailableCompressor(Compressor),
}

impl BloscBuildError {
    fn check_compressor(cname: &Compressor) -> Result<(), Self> {
        if !compressor_supported(cname) {
            Err(Self::UnavailableCompressor(*cname))
        } else {
            Ok(())
        }
    }

    fn check_typesize(shuffle: &ShuffleMode, typesize: &Option<usize>) -> Result<(), Self> {
        if typesize.is_none() && *shuffle != ShuffleMode::None {
            Err(Self::TypesizeNeeded(*shuffle))
        } else {
            Ok(())
        }
    }
}

impl TryInto<Context> for &BloscCodec {
    type Error = BloscBuildError;

    fn try_into(self) -> Result<Context, Self::Error> {
        BloscBuildError::check_typesize(&self.shuffle, &self.typesize)?;
        let ctx = Context::new()
            .compressor(self.cname)
            .map_err(|_| BloscBuildError::UnavailableCompressor(self.cname))?
            .clevel(self.clevel)
            .shuffle(self.shuffle)
            .blocksize(if self.blocksize == 0 {
                None
            } else {
                Some(self.blocksize)
            })
            .typesize(self.typesize);
        Ok(ctx)
    }
}

fn compressor_supported(cname: &Compressor) -> bool {
    match Context::new().compressor(*cname) {
        Ok(_) => true,
        Err(_) => false,
    }
}

impl BloscCodec {
    pub fn new(
        cname: Compressor,
        clevel: Clevel,
        shuffle: ShuffleMode,
        blocksize: usize,
        typesize: Option<usize>,
    ) -> Result<Self, BloscBuildError> {
        let codec = Self {
            cname,
            clevel,
            shuffle,
            blocksize,
            typesize,
        };
        codec.validate()
    }

    pub fn for_type<T: ReflectedType>(
        cname: Compressor,
        clevel: Clevel,
        shuffle: ShuffleMode,
        blocksize: usize,
    ) -> Result<Self, BloscBuildError> {
        Self::new(
            cname,
            clevel,
            shuffle,
            blocksize,
            Some(std::mem::size_of::<T>()),
        )
    }

    fn validate(self) -> Result<Self, BloscBuildError> {
        BloscBuildError::check_compressor(&self.cname)?;
        BloscBuildError::check_typesize(&self.shuffle, &self.typesize)?;
        Ok(self)
    }
}

impl Default for BloscCodec {
    fn default() -> Self {
        Self {
            cname: Compressor::BloscLZ,
            clevel: Clevel::None,
            shuffle: ShuffleMode::None,
            blocksize: 0,
            typesize: None,
        }
    }
}

struct BloscReader<R: Read> {
    r: R,
    buf: Option<Cursor<Vec<u8>>>,
}

impl<R: Read> BloscReader<R> {
    fn new(r: R) -> Self {
        Self { r, buf: None }
    }

    /// This wraps an unsafe decompression of blosc-encoded bytes.
    /// It is relatively safe, because we are not changing the data type
    /// (decoding bytes into bytes).
    /// However, we cannot guarantee that the encoded data is trustworthy.
    fn unsafe_decompress(b: &[u8]) -> io::Result<Vec<u8>> {
        unsafe { decompress_bytes(b) }
            .map_err(|_| io::Error::new(io::ErrorKind::Other, "Blosc decode failure"))
    }

    fn buffer(&mut self) -> io::Result<&mut Cursor<Vec<u8>>> {
        if self.buf.is_none() {
            let mut compressed = Vec::default();
            self.r.read_to_end(&mut compressed)?;

            let decomp: Vec<u8> = Self::unsafe_decompress(&compressed)?;
            self.buf = Some(Cursor::new(decomp));
        }
        Ok(self.buf.as_mut().unwrap())
    }
}

impl<R: Read> Read for BloscReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        Read::read(&mut self.buffer()?, buf)
    }
}

struct BloscWriter<W: Write> {
    w: W,
    pub ctx: Context,
}

impl<W: Write> BloscWriter<W> {
    fn new(codec: &BloscCodec, w: W) -> Self {
        let ctx = codec.try_into().expect("Blosc codec not enabled");
        Self { w, ctx }
    }
}

impl<W: Write> Write for BloscWriter<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        // write to intermediate buffer instead, compress on flush?
        // or write to blocksize-sized buffer and write when full
        let compressed: Vec<_> = self.ctx.compress(buf).into();
        // input length if write successful, else actual written length.
        self.w.write(&compressed).map(|written| {
            if written == compressed.len() {
                buf.len()
            } else {
                written
            }
        })
    }

    fn flush(&mut self) -> io::Result<()> {
        self.w.flush()
    }
}

impl BBCodec for BloscCodec {
    fn encoder<'a, W: Write + 'a>(&self, w: W) -> Box<dyn FinalWrite + 'a> {
        Box::new(FinalWriter(BloscWriter::new(self, w)))
    }

    fn decoder<'a, R: Read + 'a>(&self, r: R) -> Box<dyn Read + 'a> {
        Box::new(BloscReader::new(r))
    }

    fn compute_encoded_size(&self, _input_size: Option<usize>) -> Option<usize> {
        None
    }
}
