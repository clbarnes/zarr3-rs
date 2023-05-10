use std::{
    fmt::Display,
    io::{self, BufReader, Read, Write},
    str::FromStr, ops::{Deref, DerefMut},
};

use byteorder::{BigEndian, ByteOrder, LittleEndian, ReadBytesExt};
use half::f16;
use ndarray::{Array, ArrayD};
use serde::{de, Deserialize, Deserializer, Serialize, Serializer};
use serde_with::serde_as;

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
    b16,
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
            16 => Ok(Self::b16),
            32 => Ok(Self::b32),
            64 => Ok(Self::b64),
            _ => Err("not a valid float size"),
        }
    }
}

impl NBytes for FloatSize {
    fn nbytes(&self) -> usize {
        match self {
            Self::b16 => 2,
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

#[derive(Clone, Debug)]
pub(crate) struct ArrayIo<T: ReflectedType>(pub ArrayD<T>);

impl<T: ReflectedType> Deref for ArrayIo<T> {
    type Target = ArrayD<T>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T: ReflectedType> DerefMut for ArrayIo<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T: ReflectedType> Into<ArrayD<T>> for ArrayIo<T> {
    fn into(self) -> ArrayD<T> {
        self.0
    }
}

impl<T: ReflectedType> From<ArrayD<T>> for ArrayIo<T> {
    fn from(a: ArrayD<T>) -> Self {
        Self(a)
    }
}

macro_rules! data_io_impl {
    ($ty_name:ty, $bo_read_fn:ident, $bo_write_fn:ident) => {
        impl WriteNdArray<$ty_name> for ArrayIo<$ty_name> {
            fn write_to<W: Write>(self, mut w: W, endian: Endian) -> io::Result<()> {
                let CHUNK: usize = 256;
                let type_len = <$ty_name>::ZARR_TYPE.nbytes();
                let mut item_buf: Vec<$ty_name> = vec![0 as $ty_name; CHUNK];
                let mut buf: Vec<u8> = vec![0; CHUNK * type_len];
                let mut items = self.into_iter();

                let mut n_items = CHUNK;

                while n_items >= CHUNK {
                    let n_items = chunk_iter(&mut items, &mut item_buf[..]);
                    let n_bytes = n_items * type_len;
                    match endian {
                        Endian::Big => {
                            BigEndian::$bo_write_fn(&item_buf[..n_items], &mut buf[..n_bytes])
                        }
                        Endian::Little => {
                            LittleEndian::$bo_write_fn(&item_buf[..n_items], &mut buf[..n_bytes])
                        }
                    };
                    w.write_all(&buf[..n_bytes])?;
                }
                Ok(())
            }
        }

        impl ReadToNdArray<$ty_name> for ArrayIo<$ty_name> {
            fn read_from<R: Read>(
                mut r: R,
                endian: Endian,
                shape: Vec<usize>,
            ) -> Result<Self, &'static str> {
                let mut v = Vec::default();
                match endian {
                    Endian::Big => r.$bo_read_fn::<BigEndian>(v.as_mut()),
                    Endian::Little => r.$bo_read_fn::<LittleEndian>(v.as_mut()),
                };
                ArrayD::from_shape_vec(shape, v).map_err(|_| "Incompatible shape").map(|a| a.into())
            }
        }
    };
}

// Wrapper trait to erase a generic trait argument for consistent ByteOrder
// signatures.
trait ReadBytesExtI8: ReadBytesExt {
    fn read_i8_into_wrapper<B: ByteOrder>(&mut self, dst: &mut [i8]) -> io::Result<()> {
        self.read_i8_into(dst)
    }
}
impl<T: ReadBytesExt> ReadBytesExtI8 for T {}

data_io_impl!(u16, read_u16_into, write_u16_into);
data_io_impl!(u32, read_u32_into, write_u32_into);
data_io_impl!(u64, read_u64_into, write_u64_into);
data_io_impl!(i8, read_i8_into_wrapper, write_i8_into);
data_io_impl!(i16, read_i16_into, write_i16_into);
data_io_impl!(i32, read_i32_into, write_i32_into);
data_io_impl!(i64, read_i64_into, write_i64_into);
data_io_impl!(f32, read_f32_into, write_f32_into);
data_io_impl!(f64, read_f64_into, write_f64_into);

impl WriteNdArray<c64> for ArrayIo<c64> {
    fn write_to<W: Write>(self, mut w: W, endian: Endian) -> io::Result<()> {
        let CHUNK: usize = 256;
        let type_len = <c64>::ZARR_TYPE.nbytes();
        let mut item_buf: Vec<f32> = vec![0.0; CHUNK * 2];
        let mut buf: Vec<u8> = vec![0; CHUNK * type_len];
        let mut items = self.into_iter().flat_map(|c| [c.re, c.im].into_iter());

        let mut n_items = CHUNK;

        while n_items >= CHUNK {
            let n_items = chunk_iter(&mut items, &mut item_buf[..]);
            let n_bytes = n_items * type_len;
            match endian {
                Endian::Big => BigEndian::write_f32_into(&item_buf[..n_items], &mut buf[..n_bytes]),
                Endian::Little => {
                    LittleEndian::write_f32_into(&item_buf[..n_items], &mut buf[..n_bytes])
                }
            };
            w.write_all(&buf[..n_bytes])?;
        }
        Ok(())
    }
}

impl WriteNdArray<c128> for ArrayIo<c128> {
    fn write_to<W: Write>(self, mut w: W, endian: Endian) -> io::Result<()> {
        let CHUNK: usize = 256;
        let type_len = <c128>::ZARR_TYPE.nbytes();
        let mut item_buf: Vec<f64> = vec![0.0; CHUNK * 2];
        let mut buf: Vec<u8> = vec![0; CHUNK * type_len];
        let mut items = self.into_iter().flat_map(|c| [c.re, c.im].into_iter());

        let mut n_items = CHUNK;

        while n_items >= CHUNK {
            let n_items = chunk_iter(&mut items, &mut item_buf[..]);
            let n_bytes = n_items * type_len;
            match endian {
                Endian::Big => BigEndian::write_f64_into(&item_buf[..n_items], &mut buf[..n_bytes]),
                Endian::Little => {
                    LittleEndian::write_f64_into(&item_buf[..n_items], &mut buf[..n_bytes])
                }
            };
            w.write_all(&buf[..n_bytes])?;
        }
        Ok(())
    }
}

impl ReadToNdArray<c64> for ArrayIo<c64> {
    fn read_from<R: Read>(
        mut r: R,
        endian: Endian,
        shape: Vec<usize>,
    ) -> Result<Self, &'static str> {
        let mut floats: Vec<f32> = Vec::default();
        match endian {
            Endian::Big => r.read_f32_into::<BigEndian>(floats.as_mut()),
            Endian::Little => r.read_f32_into::<LittleEndian>(floats.as_mut()),
        };

        let v: Vec<_> = floats
            .chunks_exact(2)
            .map(|re_im| c64::new(re_im[0], re_im[1]))
            .collect();
        ArrayD::from_shape_vec(shape, v).map_err(|_| "Incompatible shape").map(|a| a.into())
    }
}

impl ReadToNdArray<c128> for ArrayIo<c128> {
    fn read_from<R: Read>(
        mut r: R,
        endian: Endian,
        shape: Vec<usize>,
    ) -> Result<Self, &'static str> {
        let mut floats: Vec<f64> = Vec::default();
        match endian {
            Endian::Big => r.read_f64_into::<BigEndian>(floats.as_mut()),
            Endian::Little => r.read_f64_into::<LittleEndian>(floats.as_mut()),
        };

        let v: Vec<_> = floats
            .chunks_exact(2)
            .map(|re_im| c128::new(re_im[0], re_im[1]))
            .collect();
        ArrayD::from_shape_vec(shape, v).map_err(|_| "Incompatible shape").map(|a| a.into())
    }
}

impl WriteNdArray<u8> for ArrayIo<u8> {
    fn write_to<W: Write>(self, mut w: W, endian: Endian) -> io::Result<()> {
        const CHUNK: usize = 256;
        let mut buf: [u8; CHUNK] = [0; CHUNK];

        let mut idx = 0;
        for item in self.into_iter() {
            buf[idx] = item;
            idx += 1;
            if idx >= buf.len() {
                w.write_all(&mut buf[..]);
                idx = 0;
            }
        }
        w.write_all(&buf[..idx])
    }
}

impl ReadToNdArray<u8> for ArrayIo<u8> {
    fn read_from<R: Read>(
        mut r: R,
        endian: Endian,
        shape: Vec<usize>,
    ) -> Result<Self, &'static str> {
        let mut v = Vec::default();
        r.read_to_end(&mut v);
        ArrayD::from_shape_vec(shape, v).map_err(|_| "Incompatible shape").map(|a| a.into())
    }
}

impl WriteNdArray<bool> for ArrayIo<bool> {
    fn write_to<W: Write>(self, mut w: W, endian: Endian) -> io::Result<()> {
        const CHUNK: usize = 256;
        let mut buf: [u8; CHUNK] = [0; CHUNK];

        let mut idx = 0;
        for item in self.into_iter() {
            buf[idx] = if item { 1 } else { 0 };
            idx += 1;
            if idx >= buf.len() {
                w.write_all(&mut buf[..]);
                idx = 0;
            }
        }
        w.write_all(&buf[..idx])
    }
}

impl ReadToNdArray<bool> for ArrayIo<bool> {
    fn read_from<R: Read>(
        mut r: R,
        endian: Endian,
        shape: Vec<usize>,
    ) -> Result<Self, &'static str> {
        let mut br = BufReader::new(r);
        let v: Vec<_> = br.bytes().map(|b| b.unwrap() > 0).collect();
        ArrayD::from_shape_vec(shape, v).map_err(|_| "Incompatible shape").map(|a| a.into())
    }
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

    // todo: replace array reading/writing with these
    // use bufreader & bufwriter, read however many bytes we need for a single item, use std (to|from)_[lb]e_bytes
    fn write_array_to<W: Write>(array: ArrayD<Self>, w: W, endian: Endian) -> io::Result<()>;

    fn read_array_from<R: Read>(r: R, endian: Endian, shape: &[usize]) -> ArrayD<Self>;

    // fn create_data_chunk(grid_position: &GridCoord, num_el: u32) -> VecDataChunk<Self> {
    //     VecDataChunk::<Self>::new(
    //         grid_position.clone(),
    //         vec![Self::default(); num_el as usize],
    //     )
    // }
}

macro_rules! reflected_type {
    ($d_name:expr, $d_type:ty) => {
        impl ReflectedType for $d_type {
            const ZARR_TYPE: DataType = $d_name;
        }
    };
}

reflected_type!(DataType::Bool, bool);
reflected_type!(DataType::UInt(IntSize::b8), u8);
reflected_type!(DataType::UInt(IntSize::b16), u16);
reflected_type!(DataType::UInt(IntSize::b32), u32);
reflected_type!(DataType::UInt(IntSize::b64), u64);
reflected_type!(DataType::Int(IntSize::b8), i8);
reflected_type!(DataType::Int(IntSize::b16), i16);
reflected_type!(DataType::Int(IntSize::b32), i32);
reflected_type!(DataType::Int(IntSize::b64), i64);
reflected_type!(DataType::Float(FloatSize::b16), f16);
reflected_type!(DataType::Float(FloatSize::b32), f32);
reflected_type!(DataType::Float(FloatSize::b64), f64);
reflected_type!(DataType::Complex(ComplexSize::b64), c64);
reflected_type!(DataType::Complex(ComplexSize::b128), c128);

macro_rules! reflected_raw {
    ($($nbytes:expr),*) => {
        $(
            reflected_type!(DataType::Raw($nbytes*8), [u8; $nbytes]);

            impl WriteNdArray<[u8; $nbytes]> for ArrayIo<[u8; $nbytes]> {
                fn write_to<W: Write>(self, mut w: W, _endian: Endian) -> io::Result<()> {
                    let CHUNK: usize = 256;
                    let mut buf: Vec<u8> = vec![0; CHUNK * $nbytes];

                    let mut idx = 0;
                    for item in self.into_iter() {
                        for i in item.into_iter() {
                            buf[idx] = i;
                            idx += 1;
                        }
                        if idx >= buf.len() {
                            w.write_all(&buf[..])?;
                            idx = 0;
                        }
                    }
                    w.write_all(&buf[..idx])
                }
            }

            impl ReadToNdArray<[u8; $nbytes]> for ArrayIo<[u8; $nbytes]> {
                fn read_from<R: Read>(mut r: R, _endian: Endian, shape: Vec<usize>) -> Result<Self, &'static str> {
                    let mut v = Vec::default();
                    let mut br = BufReader::new(r);
                    let mut buf = [0u8; $nbytes];
                    loop {
                        match br.read_exact(&mut buf[..]) {
                            Ok(_) => v.push(buf.clone()),
                            Err(_) => break,
                        };
                    }
                    ArrayD::from_shape_vec(shape, v).map_err(|_| "Incompatible shape").map(|a| a.into())
                }
            }
        )*
    }
}

reflected_raw!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);

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
            (r#""float16""#, Float(FloatSize::b16)),
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
