use std::{
    fmt::Display,
    io::{self, BufReader, BufWriter, Read, Write},
    str::FromStr,
};

use byteorder::{BigEndian, ByteOrder, LittleEndian, WriteBytesExt};

use ndarray::ArrayD;
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

impl DataType {
    pub fn default_fill_value(&self) -> serde_json::Value {
        match self {
            DataType::Bool => serde_json::Value::from(false),
            DataType::Int(_) | DataType::UInt(_) => serde_json::Value::from(0),
            DataType::Float(_) => serde_json::Value::from(0),
            // N.B. this presumes complex ser format
            DataType::Complex(_) => serde_json::Value::from(vec![0.0, 0.0]),
            DataType::Raw(s) => serde_json::Value::from(vec![0; s / 8]),
        }
    }

    pub fn validate_json_value(&self, value: &serde_json::Value) -> Result<(), serde_json::Error> {
        let v = value.clone();
        match self {
            DataType::Bool => {
                serde_json::from_value::<bool>(v)?;
            }
            DataType::Int(s) => match s {
                IntSize::b8 => {
                    serde_json::from_value::<i8>(v)?;
                }
                IntSize::b16 => {
                    serde_json::from_value::<i16>(v)?;
                }
                IntSize::b32 => {
                    serde_json::from_value::<i32>(v)?;
                }
                IntSize::b64 => {
                    serde_json::from_value::<i64>(v)?;
                }
            },
            DataType::UInt(s) => match s {
                IntSize::b8 => {
                    serde_json::from_value::<u8>(v)?;
                }
                IntSize::b16 => {
                    serde_json::from_value::<u16>(v)?;
                }
                IntSize::b32 => {
                    serde_json::from_value::<u32>(v)?;
                }
                IntSize::b64 => {
                    serde_json::from_value::<u64>(v)?;
                }
            },
            DataType::Float(s) => match s {
                FloatSize::b32 => {
                    serde_json::from_value::<f32>(v)?;
                }
                FloatSize::b64 => {
                    serde_json::from_value::<f64>(v)?;
                }
            },
            DataType::Complex(s) => match s {
                ComplexSize::b64 => {
                    serde_json::from_value::<c64>(v)?;
                }
                ComplexSize::b128 => {
                    serde_json::from_value::<c128>(v)?;
                }
            },
            DataType::Raw(s) => {
                let b = serde_json::from_value::<Vec<u8>>(v)?;
                if b.len() != *s {
                    return Err(de::Error::invalid_length(b.len(), &"Wrong length"));
                }
            }
        };
        Ok(())
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

/// Trait implemented by primitive types that are reflected in Zarr.
///
/// The supertraits are not necessary for this trait, but are used to
/// remove redundant bounds elsewhere when operating generically over
/// data types.
// `DeserializedOwned` is necessary for deserialization of metadata `fill_value`.
// TODO: spec does not say how to deserialize complex fill_value; we'll go with whatever num_complex has implemented
pub trait ReflectedType:
    Send
    + Sync
    + Clone
    + Default
    + serde::de::DeserializeOwned
    + 'static
    + Sized
    + serde::ser::Serialize
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
    fn write_array_to<W: Write>(array: ArrayD<Self>, w: W, endian: Endian) -> io::Result<()> {
        let mut bw = BufWriter::new(w);
        let mut buf = vec![0u8; Self::ZARR_TYPE.nbytes()];
        let encoder = Self::encoder(endian);

        for val in array.into_iter() {
            encoder(val, buf.as_mut());
            bw.write(buf.as_mut()).unwrap();
        }
        bw.flush()
    }

    fn read_array_from<R: Read>(r: R, endian: Endian, shape: &[usize]) -> ArrayD<Self> {
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
                    Endian::Big => |v: Self, buf: &mut [u8]| BigEndian::$bo_write_fn(buf, v),
                    Endian::Little => |v: Self, buf: &mut [u8]| LittleEndian::$bo_write_fn(buf, v),
                })
            }

            /// Produce a routine which reads a self-typed value from
            /// the given byte buffer.
            fn decoder(endian: Endian) -> Box<dyn Fn(&mut [u8]) -> Self> {
                Box::new(match endian {
                    Endian::Big => |buf: &mut [u8]| BigEndian::$bo_read_fn(buf),
                    Endian::Little => |buf: &mut [u8]| LittleEndian::$bo_read_fn(buf),
                })
            }
        }
    };
}

impl ReflectedType for bool {
    const ZARR_TYPE: DataType = DataType::Bool;

    fn encoder(_endian: Endian) -> Box<dyn Fn(Self, &mut [u8])> {
        Box::new(|v: Self, buf: &mut [u8]| buf[0] = if v { 1 } else { 0 })
    }

    fn decoder(_endian: Endian) -> Box<dyn Fn(&mut [u8]) -> Self> {
        Box::new(|buf: &mut [u8]| if buf[0] == 0 { false } else { true })
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
            fn decoder(_endian: Endian) -> Box<dyn Fn(&mut [u8]) -> Self> {
                Box::new(|buf: &mut[u8]| {
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

    #[test]
    /// Ensure that DataType's default fill value is reflected type default value
    fn reflected_defaults() {
        assert_eq!(
            bool::default(),
            serde_json::from_value::<bool>(DataType::Bool.default_fill_value()).unwrap()
        );

        assert_eq!(
            u8::default(),
            serde_json::from_value::<u8>(DataType::UInt(IntSize::b8).default_fill_value()).unwrap()
        );
        assert_eq!(
            u16::default(),
            serde_json::from_value::<u16>(DataType::UInt(IntSize::b16).default_fill_value())
                .unwrap()
        );
        assert_eq!(
            u32::default(),
            serde_json::from_value::<u32>(DataType::UInt(IntSize::b32).default_fill_value())
                .unwrap()
        );
        assert_eq!(
            u64::default(),
            serde_json::from_value::<u64>(DataType::UInt(IntSize::b64).default_fill_value())
                .unwrap()
        );

        assert_eq!(
            i8::default(),
            serde_json::from_value::<i8>(DataType::Int(IntSize::b8).default_fill_value()).unwrap()
        );
        assert_eq!(
            i16::default(),
            serde_json::from_value::<i16>(DataType::Int(IntSize::b16).default_fill_value())
                .unwrap()
        );
        assert_eq!(
            i32::default(),
            serde_json::from_value::<i32>(DataType::Int(IntSize::b32).default_fill_value())
                .unwrap()
        );
        assert_eq!(
            i64::default(),
            serde_json::from_value::<i64>(DataType::Int(IntSize::b64).default_fill_value())
                .unwrap()
        );

        assert_eq!(
            f32::default(),
            serde_json::from_value::<f32>(DataType::Float(FloatSize::b32).default_fill_value())
                .unwrap()
        );
        assert_eq!(
            f64::default(),
            serde_json::from_value::<f64>(DataType::Float(FloatSize::b64).default_fill_value())
                .unwrap()
        );

        assert_eq!(
            c64::default(),
            serde_json::from_value::<c64>(DataType::Complex(ComplexSize::b64).default_fill_value())
                .unwrap()
        );
        assert_eq!(
            c128::default(),
            serde_json::from_value::<c128>(
                DataType::Complex(ComplexSize::b128).default_fill_value()
            )
            .unwrap()
        );

        assert_eq!(
            <[u8; 1]>::default(),
            serde_json::from_value::<[u8; 1]>(DataType::Raw(8).default_fill_value()).unwrap()
        );
        assert_eq!(
            <[u8; 16]>::default(),
            serde_json::from_value::<[u8; 16]>(DataType::Raw(128).default_fill_value()).unwrap()
        );
    }
}
