use serde::{Deserialize, Serialize};
use thiserror::Error;
use std::io::{Read, Write};

use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression as GzCompression;

use crate::codecs::bb::BBCodec;

#[derive(Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Debug)]
#[repr(u32)]
pub enum GzipLevel {
    None = 0,
    L1 = 1,
    L2 = 2,
    L3 = 3,
    L4 = 4,
    L5 = 5,
    L6 = 6,
    L7 = 7,
    L8 = 8,
    L9 = 9,
}

#[derive(Error, Debug)]
#[error("Invalid GZIP level {0} (must be 0-9)")]
pub struct InvalidGzipLevel(u32);

impl TryFrom<u32> for GzipLevel {
    type Error = InvalidGzipLevel;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::None),
            1 => Ok(Self::L1),
            2 => Ok(Self::L2),
            3 => Ok(Self::L3),
            4 => Ok(Self::L4),
            5 => Ok(Self::L5),
            6 => Ok(Self::L6),
            7 => Ok(Self::L7),
            8 => Ok(Self::L8),
            9 => Ok(Self::L9),
            other => Err(InvalidGzipLevel(other)),
        }
    }
}

#[derive(Clone, Serialize, Deserialize, PartialEq, Debug)]
pub struct GzipCodec {
    pub level: GzipLevel,
}

fn default_gzip_level() -> GzipLevel {
    GzipLevel::L6
}

impl GzipCodec {
    pub fn from_level(level: u32) -> Result<Self, InvalidGzipLevel> {
        Ok(Self {
            level: level.try_into()?
        })
    }

    pub fn best() -> Self {
        Self {
            level: GzipLevel::L9
        }
    }

    pub fn fastest() -> Self {
        Self {
            level: GzipLevel::L1
        }
    }

    pub fn none() -> Self {
        Self { level: GzipLevel::None }
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
    fn encoder<'a, W: Write + 'a>(&self, w: W) -> Box<dyn Write + 'a> {
        Box::new(GzEncoder::new(w, GzCompression::new(self.level as u32)))
    }

    fn decoder<'a, R: Read + 'a>(&self, r: R) -> Box<dyn Read + 'a> {
        Box::new(GzDecoder::new(r))
    }
}
