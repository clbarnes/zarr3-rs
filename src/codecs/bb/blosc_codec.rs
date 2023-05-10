use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::io::{self, BufRead, Cursor, Read, Write};

use crate::codecs::bb::BBCodec;
use blosc::{decompress_bytes, Context};
pub use blosc::{Clevel, Compressor, ShuffleMode};

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct BloscCodec {
    #[serde(deserialize_with = "cname_from_str", serialize_with = "cname_to_str")]
    cname: Compressor,
    #[serde(deserialize_with = "clevel_from_str", serialize_with = "clevel_to_str")]
    clevel: Clevel,
    #[serde(
        deserialize_with = "shuffle_from_int",
        serialize_with = "shuffle_to_int"
    )]
    shuffle: ShuffleMode,
    blocksize: usize,
}

fn clevel_eq(c1: Clevel, c2: Clevel) -> bool {
    match c1 {
        Clevel::None => match c2 {
            Clevel::None => true,
            _ => false,
        },
        Clevel::L1 => match c2 {
            Clevel::L1 => true,
            _ => false,
        },
        Clevel::L2 => match c2 {
            Clevel::L2 => true,
            _ => false,
        },
        Clevel::L3 => match c2 {
            Clevel::L3 => true,
            _ => false,
        },
        Clevel::L4 => match c2 {
            Clevel::L4 => true,
            _ => false,
        },
        Clevel::L5 => match c2 {
            Clevel::L5 => true,
            _ => false,
        },
        Clevel::L6 => match c2 {
            Clevel::L6 => true,
            _ => false,
        },
        Clevel::L7 => match c2 {
            Clevel::L7 => true,
            _ => false,
        },
        Clevel::L8 => match c2 {
            Clevel::L8 => true,
            _ => false,
        },
        Clevel::L9 => match c2 {
            Clevel::L9 => true,
            _ => false,
        },
    }
}

fn shuffle_eq(s1: ShuffleMode, s2: ShuffleMode) -> bool {
    match s1 {
        ShuffleMode::None => match s2 {
            ShuffleMode::None => true,
            _ => false,
        },
        ShuffleMode::Byte => match s2 {
            ShuffleMode::Byte => true,
            _ => false,
        },
        ShuffleMode::Bit => match s2 {
            ShuffleMode::Bit => true,
            _ => false,
        },
    }
}

impl PartialEq for BloscCodec {
    fn eq(&self, other: &Self) -> bool {
        self.cname == other.cname
            && clevel_eq(self.clevel, other.clevel)
            && shuffle_eq(self.shuffle, other.shuffle)
            && self.blocksize == other.blocksize
    }
}

impl Eq for BloscCodec {}

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
    match cname {
        Clevel::None => serializer.serialize_u8(0),
        Clevel::L1 => serializer.serialize_u8(1),
        Clevel::L2 => serializer.serialize_u8(2),
        Clevel::L3 => serializer.serialize_u8(3),
        Clevel::L4 => serializer.serialize_u8(4),
        Clevel::L5 => serializer.serialize_u8(5),
        Clevel::L6 => serializer.serialize_u8(6),
        Clevel::L7 => serializer.serialize_u8(7),
        Clevel::L8 => serializer.serialize_u8(8),
        Clevel::L9 => serializer.serialize_u8(9),
    }
}

fn shuffle_from_int<'de, D>(deserializer: D) -> Result<ShuffleMode, D::Error>
where
    D: Deserializer<'de>,
{
    match Deserialize::deserialize(deserializer)? {
        0 => Ok(ShuffleMode::None),
        1 => Ok(ShuffleMode::Byte),
        2 => Ok(ShuffleMode::Bit),
        _ => Err(serde::de::Error::custom("bad shuffle")),
    }
}

fn shuffle_to_int<S>(cname: &ShuffleMode, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    match cname {
        ShuffleMode::None => serializer.serialize_u8(0),
        ShuffleMode::Byte => serializer.serialize_u8(1),
        ShuffleMode::Bit => serializer.serialize_u8(2),
    }
}

impl TryInto<Context> for &BloscCodec {
    type Error = ();

    fn try_into(self) -> Result<Context, Self::Error> {
        let ctx = Context::new()
            .compressor(self.cname)?
            .clevel(self.clevel)
            .shuffle(self.shuffle)
            .blocksize(if self.blocksize == 0 {
                None
            } else {
                Some(self.blocksize)
            });
        Ok(ctx)
    }
}

impl Default for BloscCodec {
    fn default() -> Self {
        Self {
            cname: Compressor::BloscLZ,
            clevel: Clevel::None,
            shuffle: ShuffleMode::None,
            blocksize: 0,
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
            .map_err(|_| io::Error::new(io::ErrorKind::Other, "Could not decompress with blosc"))
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
    fn encoder<'a, W: Write + 'a>(&self, w: W) -> Box<dyn Write + 'a> {
        Box::new(BloscWriter::new(self, w))
    }

    fn decoder<'a, R: Read + 'a>(&self, r: R) -> Box<dyn Read + 'a> {
        Box::new(BloscReader::new(r))
    }
}
