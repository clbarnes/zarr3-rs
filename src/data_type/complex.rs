use std::fmt::Display;

use byteorder::{BigEndian, ByteOrder, LittleEndian, WriteBytesExt};
use serde::{Deserialize, Serialize};
use serde_with::serde_as;

use crate::codecs::ab::bytes_codec::Endian;

use super::{DataType, NBytes, ReflectedType};

#[allow(non_camel_case_types)]
pub type c64 = num_complex::Complex32;
#[allow(non_camel_case_types)]
pub type c128 = num_complex::Complex64;

#[serde_as]
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum ComplexSize {
    b64,
    b128,
}

impl Display for ComplexSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.nbits())
    }
}

impl TryFrom<usize> for ComplexSize {
    type Error = &'static str;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        match value {
            64 => Ok(Self::b64),
            128 => Ok(Self::b128),
            _ => Err("not a valid complex size"),
        }
    }
}

impl NBytes for ComplexSize {
    fn nbytes(&self) -> usize {
        match self {
            Self::b64 => 8,
            Self::b128 => 16,
        }
    }
}

impl ReflectedType for c64 {
    const ZARR_TYPE: DataType = DataType::Complex(ComplexSize::b64);

    fn encoder(endian: Endian) -> Box<dyn Fn(Self, &mut [u8])> {
        Box::new(match endian {
            Endian::Big => |v: Self, mut buf: &mut [u8]| {
                buf.write_f32::<BigEndian>(v.re).unwrap();
                buf.write_f32::<BigEndian>(v.im).unwrap();
            },
            Endian::Little => |v, mut buf| {
                buf.write_f32::<LittleEndian>(v.re).unwrap();
                buf.write_f32::<LittleEndian>(v.im).unwrap();
            },
        })
    }

    fn decoder(endian: Endian) -> Box<dyn Fn(&mut [u8]) -> Self> {
        Box::new(match endian {
            Endian::Big => |buf| {
                let re = BigEndian::read_f32(buf);
                let im = BigEndian::read_f32(buf);
                Self::new(re, im)
            },
            Endian::Little => |buf| {
                let re = LittleEndian::read_f32(buf);
                let im = LittleEndian::read_f32(buf);
                Self::new(re, im)
            },
        })
    }
}

impl ReflectedType for c128 {
    const ZARR_TYPE: DataType = DataType::Complex(ComplexSize::b64);

    fn encoder(endian: Endian) -> Box<dyn Fn(Self, &mut [u8])> {
        Box::new(match endian {
            Endian::Big => |v: Self, mut buf: &mut [u8]| {
                buf.write_f64::<BigEndian>(v.re).unwrap();
                buf.write_f64::<BigEndian>(v.im).unwrap();
            },
            Endian::Little => |v, mut buf| {
                buf.write_f64::<LittleEndian>(v.re).unwrap();
                buf.write_f64::<LittleEndian>(v.im).unwrap();
            },
        })
    }

    fn decoder(endian: Endian) -> Box<dyn Fn(&mut [u8]) -> Self> {
        Box::new(match endian {
            Endian::Big => |buf| {
                let re = BigEndian::read_f64(buf);
                let im = BigEndian::read_f64(buf);
                Self::new(re, im)
            },
            Endian::Little => |buf| {
                let re = LittleEndian::read_f64(buf);
                let im = LittleEndian::read_f64(buf);
                Self::new(re, im)
            },
        })
    }
}
