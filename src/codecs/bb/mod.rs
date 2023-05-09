use std::io::{Read, Write};

use serde::{Deserialize, Serialize};

use crate::MaybeNdim;

#[cfg(feature = "blosc")]
mod blosc_codec;
#[cfg(feature = "gzip")]
mod gzip_codec;

/// Common interface for compressing writers and decompressing readers.
pub trait BBCodec {
    fn decode<'a, R: Read + 'a>(&self, r: R) -> Box<dyn Read + 'a>;

    fn encode<'a, W: Write + 'a>(&self, w: W) -> Box<dyn Write + 'a>;
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
    fn maybe_ndim(&self) -> Option<usize> {
        None
    }
}

impl BBCodec for BBCodecType {
    fn encode<'a, W: Write + 'a>(&self, w: W) -> Box<dyn Write + 'a> {
        match self {
            #[cfg(feature = "gzip")]
            Self::Gzip(c) => c.encode(w),

            #[cfg(feature = "blosc")]
            Self::Blosc(c) => c.encode(w),
        }
    }

    fn decode<'a, R: Read + 'a>(&self, r: R) -> Box<dyn Read + 'a> {
        match self {
            #[cfg(feature = "gzip")]
            Self::Gzip(c) => c.decode(r),

            #[cfg(feature = "blosc")]
            Self::Blosc(c) => c.decode(r),
        }
    }
}

impl BBCodec for &[BBCodecType] {
    fn encode<'a, W: Write + 'a>(&self, w: W) -> Box<dyn Write + 'a> {
        // todo: must be a better way
        let mut it = self.iter();

        let mut out;

        if let Some(c) = it.next() {
            out = c.encode(w);
        } else {
            return Box::new(w);
        }

        for c in it {
            out = c.encode(out);
        }

        out
    }

    fn decode<'a, R: Read + 'a>(&self, r: R) -> Box<dyn Read + 'a> {
        let mut it = self.iter().rev();

        let mut out;

        if let Some(c) = it.next() {
            out = c.decode(r);
        } else {
            return Box::new(r);
        }

        for c in it {
            out = c.decode(out);
        }

        out
    }
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
