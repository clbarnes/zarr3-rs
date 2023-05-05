use serde::{Deserialize, Serialize};
use std::io::{Cursor, Read, Result, Write};

use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression as GzCompression;

use crate::codecs::bb::BBCodec;

use super::ByteReader;

#[derive(Clone, Serialize, Deserialize, PartialEq, Debug)]
pub struct GzipCodec {
    pub level: u32,
}

impl Default for GzipCodec {
    fn default() -> Self {
        Self { level: 6 }
    }
}

impl BBCodec for GzipCodec {
    fn encode(&self, raw: &[u8]) -> Vec<u8> {
        let mut encoder = GzEncoder::new(Vec::default(), GzCompression::new(self.level));
        encoder.write_all(raw).expect("Could not encode bytes");
        encoder.finish().expect("Could not write buffer")
    }

    fn decode(&self, mut encoded: Box<dyn ByteReader>) -> Result<Box<dyn ByteReader>> {
        let mut decoder = GzDecoder::new(Cursor::new(encoded.read()?));
        let mut out = Vec::default();
        decoder.read_to_end(&mut out).expect("Could not decode");
        Ok(Box::new(Cursor::new(out)))
    }
}
