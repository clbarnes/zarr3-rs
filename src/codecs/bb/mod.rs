use bytes::Bytes;
use serde::{Deserialize, Serialize};

use crate::{variant_from_data, MaybeNdim};

#[cfg(feature = "blosc")]
pub mod blosc_codec;
#[cfg(feature = "gzip")]
pub mod gzip_codec;

/// Common interface for compressing writers and decompressing readers.
pub trait BBCodec {
    fn decode(&self, encoded: &[u8]) -> Bytes;

    fn encode(&self, decoded: &[u8]) -> Bytes;
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
    fn encode(&self, decoded: &[u8]) -> Bytes {
        match self {
            #[cfg(feature = "gzip")]
            Self::Gzip(c) => c.encode(decoded),

            #[cfg(feature = "blosc")]
            Self::Blosc(c) => c.encode(decoded),
        }
    }

    fn decode(&self, encoded: &[u8]) -> Bytes {
        match self {
            #[cfg(feature = "gzip")]
            Self::Gzip(c) => c.decode(encoded),

            #[cfg(feature = "blosc")]
            Self::Blosc(c) => c.decode(encoded),
        }
    }
}

impl BBCodec for &[BBCodecType] {
    fn encode(&self, decoded: &[u8]) -> Bytes {
        // todo: must be a better way
        let mut it = self.iter();

        let mut out;

        if let Some(c) = it.next() {
            out = c.encode(decoded);
        } else {
            return Bytes::from(decoded);
        }

        for c in it {
            out = c.encode(&out[..]);
        }

        out
    }

    fn decode(&self, encoded: &[u8]) -> Bytes {
        let mut it = self.iter().rev();

        let mut out;

        if let Some(c) = it.next() {
            out = c.decode(encoded);
        } else {
            return Bytes::from(encoded);
        }

        for c in it {
            out = c.decode(&out[..]);
        }

        out
    }
}

#[cfg(feature = "gzip")]
variant_from_data!(BBCodecType, Gzip, gzip_codec::GzipCodec);

#[cfg(feature = "blosc")]
variant_from_data!(BBCodecType, Blosc, blosc_codec::BloscCodec);
