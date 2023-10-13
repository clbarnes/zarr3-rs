use std::io::{Read, Write};

use serde::{Deserialize, Serialize};

use crate::{variant_from_data, MaybeNdim};

#[cfg(feature = "blosc")]
pub mod blosc_codec;
#[cfg(feature = "gzip")]
pub mod gzip_codec;

pub mod crc32c_codec;

use super::fwrite::{FinalWrite, FinalWriter};

/// Common interface for compressing writers and decompressing readers.
pub trait BBCodec {
    fn decoder<'a, R: Read + 'a>(&self, r: R) -> Box<dyn Read + 'a>;

    fn encoder<'a, W: Write + 'a>(&self, w: W) -> Box<dyn FinalWrite + 'a>;
}

#[derive(Clone, Serialize, Deserialize, PartialEq, Debug)]
#[serde(rename_all = "lowercase", tag = "name", content = "configuration")]
pub enum BBCodecType {
    #[cfg(feature = "blosc")]
    Blosc(blosc_codec::BloscCodec),
    #[cfg(feature = "gzip")]
    Gzip(gzip_codec::GzipCodec),
    // Option because configuration could be missing or null (there is nothing to configure)
    Crc32c(Option<crc32c_codec::Crc32cCodec>),
}

impl MaybeNdim for BBCodecType {
    fn maybe_ndim(&self) -> Option<usize> {
        None
    }
}

impl BBCodec for BBCodecType {
    fn encoder<'a, W: Write + 'a>(&self, w: W) -> Box<dyn FinalWrite + 'a> {
        match self {
            #[cfg(feature = "gzip")]
            Self::Gzip(c) => c.encoder(w),

            #[cfg(feature = "blosc")]
            Self::Blosc(c) => c.encoder(w),
            Self::Crc32c(c) => c.unwrap_or_default().encoder(w),
        }
    }

    fn decoder<'a, R: Read + 'a>(&self, r: R) -> Box<dyn Read + 'a> {
        match self {
            #[cfg(feature = "gzip")]
            Self::Gzip(c) => c.decoder(r),

            #[cfg(feature = "blosc")]
            Self::Blosc(c) => c.decoder(r),
            Self::Crc32c(c) => c.unwrap_or_default().decoder(r),
        }
    }
}

impl BBCodec for &[BBCodecType] {
    fn encoder<'a, W: Write + 'a>(&self, w: W) -> Box<dyn FinalWrite + 'a> {
        // todo: must be a better way
        let mut it = self.iter();

        let mut out;

        if let Some(c) = it.next() {
            out = c.encoder(w);
        } else {
            return Box::new(FinalWriter(w));
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

#[cfg(test)]
mod tests {
    use super::*;
    use gzip_codec::GzipLevel;

    #[cfg(feature = "gzip")]
    #[test]
    fn can_deser_gzip() {
        let s = r#"{"name": "gzip", "configuration": {"level": 1}}"#;
        let codec: BBCodecType = serde_json::from_str(s).unwrap();
        if let BBCodecType::Gzip(c) = codec {
            assert_eq!(c.level, GzipLevel::L1);
        } else {
            panic!("Didn't deserialize gzip");
        }
    }

    #[test]
    fn can_deser_crc32c() {
        let s = r#"{"name": "crc32c", "configuration": {}}"#;
        let codec: BBCodecType = serde_json::from_str(s).unwrap();

        match codec {
            BBCodecType::Crc32c(_) => (),
            _ => panic!("Didn't deserialize crc32c"),
        }
    }

    #[test]
    fn can_deser_crc32c_noconfig() {
        let s = r#"{"name": "crc32c"}"#;
        let codec: BBCodecType = serde_json::from_str(s).unwrap();

        match codec {
            BBCodecType::Crc32c(_) => (),
            _ => panic!("Didn't deserialize crc32c"),
        }
    }
}
