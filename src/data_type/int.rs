use std::fmt::Display;

use byteorder::WriteBytesExt;
use serde::{Deserialize, Serialize};
use serde_with::serde_as;

use crate::codecs::ab::bytes_codec::Endian;

use super::{DataType, NBytes, ReflectedType};

#[serde_as]
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum IntSize {
    b8,
    b16,
    b32,
    b64,
}

impl TryFrom<usize> for IntSize {
    type Error = &'static str;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        match value {
            8 => Ok(Self::b8),
            16 => Ok(Self::b16),
            32 => Ok(Self::b32),
            64 => Ok(Self::b64),
            _ => Err("not a valid integer size"),
        }
    }
}

impl Display for IntSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.nbits())
    }
}

impl NBytes for IntSize {
    fn nbytes(&self) -> usize {
        match self {
            Self::b8 => 1,
            Self::b16 => 2,
            Self::b32 => 4,
            Self::b64 => 8,
        }
    }
}

impl ReflectedType for u8 {
    const ZARR_TYPE: DataType = DataType::UInt(IntSize::b8);

    fn encoder(_endian: Endian) -> Box<dyn Fn(Self, &mut [u8])> {
        Box::new(|v: Self, buf: &mut [u8]| buf[0] = v)
    }

    fn decoder(_endian: Endian) -> Box<dyn Fn(&mut [u8]) -> Self> {
        Box::new(|buf: &mut [u8]| buf[0])
    }
}

impl ReflectedType for i8 {
    const ZARR_TYPE: DataType = DataType::UInt(IntSize::b8);

    fn encoder(_endian: Endian) -> Box<dyn Fn(Self, &mut [u8])> {
        Box::new(|v: Self, mut buf: &mut [u8]| buf.write_i8(v).unwrap())
    }

    fn decoder(_endian: Endian) -> Box<dyn Fn(&mut [u8]) -> Self> {
        // todo: kludge to get type bounds to work, should be a better way
        Box::new(|buf: &mut [u8]| Self::from_le_bytes([buf[0]]))
    }
}
