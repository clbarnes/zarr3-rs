use serde::{Deserialize, Serialize};
use std::io::{Read, Write};

use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression as GzCompression;

use log::warn;

use crate::codecs::bb::BBCodec;

#[derive(Clone, Serialize, Deserialize, PartialEq, Debug)]
pub struct GzipCodec {
    pub level: i32,
}

fn default_gzip_level() -> i32 {
    6
}

impl GzipCodec {
    fn get_compression(&self) -> GzCompression {
        if self.level < 0 || self.level > 9 {
            warn!(
                "GZip level {} is out of range 0-9; using default",
                self.level
            );
            GzCompression::default()
        } else {
            GzCompression::new(self.level as u32)
        }
    }
}

impl Default for GzipCodec {
    fn default() -> Self {
        Self {
            level: default_gzip_level(),
        }
    }
}

impl BBCodec for GzipCodec {
    fn encode<'a, W: Write + 'a>(&self, w: W) -> Box<dyn Write + 'a> {
        Box::new(GzEncoder::new(w, self.get_compression()))
    }

    fn decode<'a, R: Read + 'a>(&self, r: R) -> Box<dyn Read + 'a> {
        Box::new(GzDecoder::new(r))
    }
}
