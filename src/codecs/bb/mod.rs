use std::io::{Read, Write};

use serde::{Deserialize, Serialize};

use crate::{variant_from_data, MaybeNdim};

#[cfg(feature = "blosc")]
pub mod blosc_codec;
#[cfg(feature = "gzip")]
pub mod gzip_codec;

/// Common interface for compressing writers and decompressing readers.
pub trait BBCodec {
    fn decoder<'a, R: Read + 'a>(&self, r: R) -> Box<dyn Read + 'a>;

    fn encoder<'a, W: Write + 'a>(&self, w: W) -> Box<dyn Write + 'a>;
}

#[derive(Clone, Serialize, Deserialize, PartialEq, Debug)]
#[serde(rename_all = "lowercase", tag = "name", content = "configuration")]
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
    fn encoder<'a, W: Write + 'a>(&self, w: W) -> Box<dyn Write + 'a> {
        match self {
            #[cfg(feature = "gzip")]
            Self::Gzip(c) => c.encoder(w),

            #[cfg(feature = "blosc")]
            Self::Blosc(c) => c.encoder(w),
        }
    }

    fn decoder<'a, R: Read + 'a>(&self, r: R) -> Box<dyn Read + 'a> {
        match self {
            #[cfg(feature = "gzip")]
            Self::Gzip(c) => c.decoder(r),

            #[cfg(feature = "blosc")]
            Self::Blosc(c) => c.decoder(r),
        }
    }
}

impl BBCodec for &[BBCodecType] {
    fn encoder<'a, W: Write + 'a>(&self, w: W) -> Box<dyn Write + 'a> {
        // todo: must be a better way
        let mut it = self.iter();

        let mut out;

        if let Some(c) = it.next() {
            out = c.encoder(w);
        } else {
            return Box::new(w);
        }

        for c in it {
            out = c.encoder(out);
        }

        out
    }

    fn decoder<'a, R: Read + 'a>(&self, r: R) -> Box<dyn Read + 'a> {
        let mut it = self.iter().rev();

        let mut out;

        if let Some(c) = it.next() {
            out = c.decoder(r);
        } else {
            return Box::new(r);
        }

        for c in it {
            out = c.decoder(out);
        }

        out
    }
}

#[cfg(feature = "gzip")]
variant_from_data!(BBCodecType, Gzip, gzip_codec::GzipCodec);

#[cfg(feature = "blosc")]
variant_from_data!(BBCodecType, Blosc, blosc_codec::BloscCodec);
