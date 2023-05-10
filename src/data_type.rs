use std::{
    fmt::Display,
    io::{self, BufReader, BufWriter, Read, Write},
    ops::{Deref, DerefMut},
    str::FromStr,
};

use byteorder::{BigEndian, ByteOrder, LittleEndian, ReadBytesExt, WriteBytesExt};
use half::f16;
use ndarray::{Array, ArrayD, Data};
use serde::{de, Deserialize, Deserializer, Serialize, Serializer};
use serde_with::serde_as;
use smallvec::{smallvec, SmallVec};

use crate::codecs::ab::endian::Endian;

pub trait NBytes {
    fn nbytes(&self) -> usize;

    fn nbits(&self) -> usize {
        self.nbytes() * 8
    }

    fn has_endianness(&self) -> bool {
        self.nbytes() > 1
    }
}

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

#[serde_as]
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum FloatSize {
    // b16,
    b32,
    b64,
}

impl Display for FloatSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.nbits())
    }
}

impl TryFrom<usize> for FloatSize {
    type Error = &'static str;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        match value {
            // 16 => Ok(Self::b16),
            32 => Ok(Self::b32),
            64 => Ok(Self::b64),
            _ => Err("not a valid float size"),
        }
    }
}

impl NBytes for FloatSize {
    fn nbytes(&self) -> usize {
        match self {
            // Self::b16 => 2,
            Self::b32 => 4,
            Self::b64 => 8,
        }
    }
}

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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UnknownDataType {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    configuration: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    fallback: Option<Box<ExtensibleDataType>>,
}

// Adding extensions to this enum makes ser/deser much harder;
// would probably need to drop FromStr/Display impl
#[serde_as]
#[derive(Debug, Clone, PartialEq)]
pub enum DataType {
    Bool,
    Int(IntSize),
    UInt(IntSize),
    Float(FloatSize),
    Complex(ComplexSize),
    Raw(usize),
}

impl TryFrom<ExtensibleDataType> for DataType {
    type Error = &'static str;

    fn try_from(value: ExtensibleDataType) -> Result<Self, Self::Error> {
        match value {
            ExtensibleDataType::Known(d) => Ok(d),
            ExtensibleDataType::Unknown(u) => {
                if let Some(f) = u.fallback {
                    (*f).try_into()
                } else {
                    Err("Extension datatype has no known fallback")
                }
            }
        }
    }
}

// todo: as extension dtypes are added, we can either separate by
// known/unknown or core/ extension.
// Extension must continue to support unknown with fallbacks,
// which is tricky to ser/deser.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ExtensibleDataType {
    Known(DataType),
    Unknown(UnknownDataType),
}

impl Serialize for DataType {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.to_string())
        // when extension data types are supported,
        // enumerate core data types and handle extensions as "other"
    }
}

impl<'de> Deserialize<'de> for DataType {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        FromStr::from_str(&s).map_err(de::Error::custom)
    }
}

impl NBytes for DataType {
    fn nbytes(&self) -> usize {
        match self {
            Self::Bool => 1,
            Self::Int(s) | Self::UInt(s) => s.nbytes(),
            Self::Float(s) => s.nbytes(),
            Self::Complex(s) => s.nbytes(),
            Self::Raw(s) => *s / 8,
        }
    }

    fn has_endianness(&self) -> bool {
        if let Self::Raw(_) = self {
            false
        } else {
            self.nbytes() > 1
        }
    }
}

impl Display for DataType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let nbits = self.nbits();
        let s = match self {
            Self::Bool => "bool".into(),
            Self::Int(_s) => format!("int{nbits}"),
            Self::UInt(_s) => format!("uint{nbits}"),
            Self::Float(_s) => format!("float{nbits}"),
            Self::Complex(_s) => format!("complex{nbits}"),
            Self::Raw(_s) => format!("r{nbits}"),
        };
        write!(f, "{}", s)
    }
}

fn split_str_num<'a>(s: &'a str) -> (&'a str, Option<usize>) {
    let clos = |c: char| c.is_ascii_digit();
    if let Some(idx) = s.find(clos) {
        (
            &s[0..idx],
            Some(s[idx..].parse().expect("non-digit after digit")),
        )
    } else {
        (s, None)
    }
}

impl FromStr for DataType {
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (s, nbits) = split_str_num(s);
        if let Some(n) = nbits {
            match s {
                "int" => Ok(Self::Int(n.try_into()?)),
                "uint" => Ok(Self::UInt(n.try_into()?)),
                "float" => Ok(Self::Float(n.try_into()?)),
                "complex" => Ok(Self::Complex(n.try_into()?)),
                "r" => {
                    if n % 8 == 0 {
                        Ok(Self::Raw(n))
                    } else {
                        Err("Raw width is not a multiple of 8")
                    }
                }
                _ => Err("Unknown data type"),
            }
        } else if s == "bool" {
            return Ok(Self::Bool);
        } else {
            Err("Could not parse data type")
        }
    }

    type Err = &'static str;
}

#[allow(non_camel_case_types)]
pub type c64 = num_complex::Complex32;
#[allow(non_camel_case_types)]
pub type c128 = num_complex::Complex64;

pub trait WriteNdArray<T: ReflectedType> {
    fn write_to<W: Write>(self, w: W, endian: Endian) -> io::Result<()>;
}

pub trait ReadToNdArray<T: ReflectedType>: Sized {
    fn read_from<R: Read>(r: R, endian: Endian, shape: Vec<usize>) -> Result<Self, &'static str>;
}

fn chunk_iter<T, I: Iterator<Item = T>>(it: &mut I, buf: &mut [T]) -> usize {
    let mut count = 0;
    for item in it.take(buf.len()) {
        buf[count] = item;
        count += 1;
    }
    count
}

/// Trait implemented by primitive types that are reflected in Zarr.
///
/// The supertraits are not necessary for this trait, but are used to
/// remove redundant bounds elsewhere when operating generically over
/// data types.
// `DeserializedOwned` is necessary for deserialization of metadata `fill_value`.
// TODO: spec does not say how to deserialize complex fill_value; we'll go with whatever num_complex has implemented
pub trait ReflectedType:
    Send + Sync + Clone + Default + serde::de::DeserializeOwned + 'static + Sized
{
    const ZARR_TYPE: DataType;

    // this is to avoid a match in a hot loop but the Box deref might be slower anyway?
    /// Produce a routine which writes the bytes of a self-typed value
    /// into the given buffer.
    fn encoder(endian: Endian) -> Box<dyn Fn(Self, &mut [u8])>;

    /// Produce a routine which reads a self-typed value from
    /// the given byte buffer.
    fn decoder(endian: Endian) -> Box<dyn Fn(&mut [u8]) -> Self>;

    // todo: replace array reading/writing with these
    // use bufreader & bufwriter, read however many bytes we need for a single item, use std (to|from)_[lb]e_bytes
    fn write_array_to<W: Write>(array: ArrayD<Self>, mut w: W, endian: Endian) -> io::Result<()> {
        let mut bw = BufWriter::new(w);
        let mut buf = vec![0u8; Self::ZARR_TYPE.nbytes()];
        let encoder = Self::encoder(endian);

        for val in array.into_iter() {
            encoder(val, buf.as_mut());
            bw.write(buf.as_mut()).unwrap();
        }
        bw.flush()
    }

    fn read_array_from<R: Read>(mut r: R, endian: Endian, shape: &[usize]) -> ArrayD<Self> {
        let mut br = BufReader::new(r);
        let mut buf = vec![0u8; Self::ZARR_TYPE.nbytes()];
        let decoder = Self::decoder(endian);

        let numel = shape.iter().cloned().reduce(|a, b| a * b).unwrap_or(1);

        let mut data = Vec::with_capacity(numel);

        for _ in 0..numel {
            br.read_exact(buf.as_mut()).unwrap();
            data.push(decoder(buf.as_mut()));
        }

        ArrayD::from_shape_vec(shape.to_vec(), data).unwrap()
    }

    // fn create_data_chunk(grid_position: &GridCoord, num_el: u32) -> VecDataChunk<Self> {
    //     VecDataChunk::<Self>::new(
    //         grid_position.clone(),
    //         vec![Self::default(); num_el as usize],
    //     )
    // }
}

macro_rules! reflected_primitive {
    ($d_name:expr, $d_type:ty, $bo_read_fn:ident, $bo_write_fn:ident) => {
        impl ReflectedType for $d_type {
            const ZARR_TYPE: DataType = $d_name;

            fn encoder(endian: Endian) -> Box<dyn Fn(Self, &mut [u8])> {
                Box::new(match endian {
                    Endian::Big => |v: Self, mut buf: &mut [u8]| BigEndian::$bo_write_fn(buf, v),
                    Endian::Little => {
                        |v: Self, mut buf: &mut [u8]| LittleEndian::$bo_write_fn(buf, v)
                    }
                })
            }

            /// Produce a routine which reads a self-typed value from
            /// the given byte buffer.
            fn decoder(endian: Endian) -> Box<dyn Fn(&mut [u8]) -> Self> {
                Box::new(match endian {
                    Endian::Big => |mut buf: &mut [u8]| BigEndian::$bo_read_fn(buf),
                    Endian::Little => |mut buf: &mut [u8]| LittleEndian::$bo_read_fn(buf),
                })
            }
        }
    };
}

impl ReflectedType for bool {
    const ZARR_TYPE: DataType = DataType::Bool;

    fn encoder(_endian: Endian) -> Box<dyn Fn(Self, &mut [u8])> {
        Box::new(|v: Self, mut buf: &mut [u8]| buf[0] = if v { 1 } else { 0 })
    }

    fn decoder(_endian: Endian) -> Box<dyn Fn(&mut [u8]) -> Self> {
        Box::new(|mut buf: &mut [u8]| if buf[0] == 0 { false } else { true })
    }
}

impl ReflectedType for u8 {
    const ZARR_TYPE: DataType = DataType::UInt(IntSize::b8);

    fn encoder(_endian: Endian) -> Box<dyn Fn(Self, &mut [u8])> {
        Box::new(|v: Self, mut buf: &mut [u8]| buf[0] = v)
    }

    fn decoder(_endian: Endian) -> Box<dyn Fn(&mut [u8]) -> Self> {
        Box::new(|mut buf: &mut [u8]| buf[0])
    }
}

impl ReflectedType for i8 {
    const ZARR_TYPE: DataType = DataType::UInt(IntSize::b8);

    fn encoder(_endian: Endian) -> Box<dyn Fn(Self, &mut [u8])> {
        Box::new(|v: Self, mut buf: &mut [u8]| buf.write_i8(v).unwrap())
    }

    fn decoder(_endian: Endian) -> Box<dyn Fn(&mut [u8]) -> Self> {
        // todo: kludge to get type bounds to work, should be a better way
        Box::new(|mut buf: &mut [u8]| Self::from_le_bytes([buf[0]]))
    }
}

reflected_primitive!(DataType::UInt(IntSize::b16), u16, read_u16, write_u16);
reflected_primitive!(DataType::UInt(IntSize::b32), u32, read_u32, write_u32);
reflected_primitive!(DataType::UInt(IntSize::b64), u64, read_u64, write_u64);
reflected_primitive!(DataType::Int(IntSize::b16), i16, read_i16, write_i16);
reflected_primitive!(DataType::Int(IntSize::b32), i32, read_i32, write_i32);
reflected_primitive!(DataType::Int(IntSize::b64), i64, read_i64, write_i64);
reflected_primitive!(DataType::Float(FloatSize::b32), f32, read_f32, write_f32);
reflected_primitive!(DataType::Float(FloatSize::b64), f64, read_f64, write_f64);

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
            Endian::Big => |mut buf| {
                let re = BigEndian::read_f32(buf);
                let im = BigEndian::read_f32(buf);
                Self::new(re, im)
            },
            Endian::Little => |mut buf| {
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
            Endian::Big => |mut buf| {
                let re = BigEndian::read_f64(buf);
                let im = BigEndian::read_f64(buf);
                Self::new(re, im)
            },
            Endian::Little => |mut buf| {
                let re = LittleEndian::read_f64(buf);
                let im = LittleEndian::read_f64(buf);
                Self::new(re, im)
            },
        })
    }
}

macro_rules! reflected_raw {
    ($($nbytes:expr), *) => {
        $(
        impl ReflectedType for [u8; $nbytes] {
            const ZARR_TYPE: DataType = DataType::Raw($nbytes * 8);

            fn encoder(_endian: Endian) -> Box<dyn Fn(Self, &mut [u8])> {
                Box::new(|v: Self, buf: &mut[u8]| {
                    buf.copy_from_slice(&v);
                })
            }

            /// Produce a routine which reads a self-typed value from
            /// the given byte buffer.
            fn decoder(endian: Endian) -> Box<dyn Fn(&mut [u8]) -> Self> {
                Box::new(|mut buf: &mut[u8]| {
                    let mut out = [0; $nbytes];
                    out.as_mut().copy_from_slice(buf);
                    out
                })
            }
        }
    )*
    }
}

reflected_raw!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);

// reflected_type!(DataType::Bool, bool);
// reflected_type!(DataType::UInt(IntSize::b8), u8);
// reflected_type!(DataType::UInt(IntSize::b16), u16);
// reflected_type!(DataType::UInt(IntSize::b32), u32);
// reflected_type!(DataType::UInt(IntSize::b64), u64);
// reflected_type!(DataType::Int(IntSize::b8), i8);
// reflected_type!(DataType::Int(IntSize::b16), i16);
// reflected_type!(DataType::Int(IntSize::b32), i32);
// reflected_type!(DataType::Int(IntSize::b64), i64);
// reflected_type!(DataType::Float(FloatSize::b16), f16);
// reflected_type!(DataType::Float(FloatSize::b32), f32);
// reflected_type!(DataType::Float(FloatSize::b64), f64);
// reflected_type!(DataType::Complex(ComplexSize::b64), c64);
// reflected_type!(DataType::Complex(ComplexSize::b128), c128);

// macro_rules! reflected_raw {
//     ($($nbytes:expr),*) => {
//         $(
//             reflected_type!(DataType::Raw($nbytes*8), [u8; $nbytes]);

//             impl WriteNdArray<[u8; $nbytes]> for ArrayIo<[u8; $nbytes]> {
//                 fn write_to<W: Write>(self, mut w: W, _endian: Endian) -> io::Result<()> {
//                     let CHUNK: usize = 256;
//                     let mut buf: Vec<u8> = vec![0; CHUNK * $nbytes];

//                     let mut idx = 0;
//                     for item in self.into_iter() {
//                         for i in item.into_iter() {
//                             buf[idx] = i;
//                             idx += 1;
//                         }
//                         if idx >= buf.len() {
//                             w.write_all(&buf[..])?;
//                             idx = 0;
//                         }
//                     }
//                     w.write_all(&buf[..idx])
//                 }
//             }

//             impl ReadToNdArray<[u8; $nbytes]> for ArrayIo<[u8; $nbytes]> {
//                 fn read_from<R: Read>(mut r: R, _endian: Endian, shape: Vec<usize>) -> Result<Self, &'static str> {
//                     let mut v = Vec::default();
//                     let mut br = BufReader::new(r);
//                     let mut buf = [0u8; $nbytes];
//                     loop {
//                         match br.read_exact(&mut buf[..]) {
//                             Ok(_) => v.push(buf.clone()),
//                             Err(_) => break,
//                         };
//                     }
//                     ArrayD::from_shape_vec(shape, v).map_err(|_| "Incompatible shape").map(|a| a.into())
//                 }
//             }
//         )*
//     }
// }

// reflected_raw!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_dtypes() {
        use DataType::*;
        let strs = vec![
            (r#""bool""#, Bool),
            (r#""int8""#, Int(IntSize::b8)),
            (r#""int16""#, Int(IntSize::b16)),
            (r#""int32""#, Int(IntSize::b32)),
            (r#""int64""#, Int(IntSize::b64)),
            (r#""uint8""#, UInt(IntSize::b8)),
            (r#""uint16""#, UInt(IntSize::b16)),
            (r#""uint32""#, UInt(IntSize::b32)),
            (r#""uint64""#, UInt(IntSize::b64)),
            // (r#""float16""#, Float(FloatSize::b16)),
            (r#""float32""#, Float(FloatSize::b32)),
            (r#""float64""#, Float(FloatSize::b64)),
            (r#""complex64""#, Complex(ComplexSize::b64)),
            (r#""complex128""#, Complex(ComplexSize::b128)),
            (r#""r8""#, Raw(8)),
            (r#""r16""#, Raw(16)),
            (r#""r128""#, Raw(128)),
        ];
        for (s, expected) in strs {
            let dt: DataType = serde_json::from_str(s).expect(&format!("Couldn't parse '{}'", s));
            assert_eq!(dt, expected, "Got {:?}, expected {:?}", dt, expected);

            let s2 = serde_json::to_string(&dt).expect(&format!("Couldn't serialize {:?}", dt));
            assert_eq!(s, &s2, "Got {:?}, expected {:?}", s2, s);
        }
    }

    #[test]
    fn parse_unknown() {
        use ExtensibleDataType::*;

        let s = r#"{"name":"newtype","fallback":"uint8"}"#;
        let dt: ExtensibleDataType =
            serde_json::from_str(s).expect(&format!("Couldn't parse '{}'", s));
        match &dt {
            Unknown(d) => assert_eq!(d.name, "newtype"),
            _ => panic!("got wrong dtype"),
        };
        let d: DataType = dt.try_into().expect("Could not fall back");
        assert_eq!(d, DataType::UInt(IntSize::b8));
    }
}
