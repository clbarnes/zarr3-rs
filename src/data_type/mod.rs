use std::{
    fmt::{Debug, Display},
    io::{self, BufReader, BufWriter, Read, Write},
    str::FromStr,
};

use serde::{de, Deserialize, Deserializer, Serialize, Serializer};
use serde_with::serde_as;

use crate::{codecs::ab::bytes_codec::Endian, ArcArrayD};
mod complex;
mod raw;

pub use complex::{c128, c64, ComplexSize};
mod int;
pub use int::IntSize;
mod float;
pub use float::FloatSize;

pub trait NBytes {
    // todo - might need variable at some point
    /// Number of bytes in the data type
    fn nbytes(&self) -> usize;

    /// Number of bits in the data type
    fn nbits(&self) -> usize {
        self.nbytes() * 8
    }

    /// Whether the data type should have an endianness.
    fn has_endianness(&self) -> bool {
        self.nbytes() > 1
    }

    /// A valid endianness for this data type.
    ///
    /// Uses the given endianness if [Some], or a meaningless default if the data type does not require one (e.g. single-byte) and [None] is given, or an error if an endianness is needed but not given.
    fn valid_endian(&self, endian: Option<Endian>) -> Result<Endian, &'static str> {
        match endian {
            Some(e) => Ok(e),
            None => {
                if self.has_endianness() {
                    Err("Endianness undefined for dtype which requires it (multi-byte, not raw)")
                } else {
                    Ok(Default::default())
                }
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UnknownDataType {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    configuration: Option<serde_json::Value>,
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
            ExtensibleDataType::Unknown(_) => Err("Unknown data type"),
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
                // todo: check for NaN, +-Inf
                FloatSize::b32 => {
                    serde_json::from_value::<f32>(v)?;
                }
                FloatSize::b64 => {
                    serde_json::from_value::<f64>(v)?;
                }
            },
            DataType::Complex(s) => match s {
                // todo: check for NaN, +-Inf
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

fn split_str_num(s: &str) -> (&str, Option<usize>) {
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

type PrimitiveEncoder<T> = Box<dyn Fn(T, &mut [u8])>;
type PrimitiveDecoder<T> = Box<dyn Fn(&mut [u8]) -> T>;

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
    + Copy
    + Default
    + serde::de::DeserializeOwned
    + 'static
    + Sized
    + serde::ser::Serialize
    + PartialEq
    + Debug
{
    const ZARR_TYPE: DataType;

    // this is to avoid a match in a hot loop but the Box deref might be slower anyway?
    /// Produce a routine which writes the bytes of a self-typed value
    /// into the given buffer.
    fn encoder(endian: Endian) -> PrimitiveEncoder<Self>;

    /// Produce a routine which reads a self-typed value from
    /// the given byte buffer.
    fn decoder(endian: Endian) -> PrimitiveDecoder<Self>;

    // todo: replace array reading/writing with these
    // use bufreader & bufwriter, read however many bytes we need for a single item, use std (to|from)_[lb]e_bytes
    fn write_array_to<W: Write>(array: ArcArrayD<Self>, w: W, endian: Endian) -> io::Result<()> {
        let mut bw = BufWriter::new(w);
        let mut buf = vec![0u8; Self::ZARR_TYPE.nbytes()];
        let encoder = Self::encoder(endian);

        for val in array.into_iter() {
            encoder(val, buf.as_mut());
            bw.write_all(buf.as_mut()).unwrap();
        }
        bw.flush()
    }

    fn read_array_from<R: Read>(r: R, endian: Endian, shape: &[usize]) -> ArcArrayD<Self> {
        let mut br = BufReader::new(r);
        let mut buf = vec![0u8; Self::ZARR_TYPE.nbytes()];
        let decoder = Self::decoder(endian);

        let numel = shape.iter().cloned().reduce(|a, b| a * b).unwrap_or(1);

        let mut data = Vec::with_capacity(numel);

        for _ in 0..numel {
            br.read_exact(buf.as_mut()).unwrap();
            data.push(decoder(buf.as_mut()));
        }

        ArcArrayD::from_shape_vec(shape.to_vec(), data).unwrap()
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
                use byteorder::ByteOrder;
                Box::new(match endian {
                    Endian::Big => {
                        |v: Self, buf: &mut [u8]| byteorder::BigEndian::$bo_write_fn(buf, v)
                    }
                    Endian::Little => {
                        |v: Self, buf: &mut [u8]| byteorder::LittleEndian::$bo_write_fn(buf, v)
                    }
                })
            }

            /// Produce a routine which reads a self-typed value from
            /// the given byte buffer.
            fn decoder(endian: Endian) -> Box<dyn Fn(&mut [u8]) -> Self> {
                use byteorder::ByteOrder;
                Box::new(match endian {
                    Endian::Big => |buf: &mut [u8]| byteorder::BigEndian::$bo_read_fn(buf),
                    Endian::Little => |buf: &mut [u8]| byteorder::LittleEndian::$bo_read_fn(buf),
                })
            }
        }
    };
}

pub(crate) use reflected_primitive;

impl ReflectedType for bool {
    const ZARR_TYPE: DataType = DataType::Bool;

    fn encoder(_endian: Endian) -> Box<dyn Fn(Self, &mut [u8])> {
        Box::new(|v: Self, buf: &mut [u8]| buf[0] = if v { 1 } else { 0 })
    }

    fn decoder(_endian: Endian) -> Box<dyn Fn(&mut [u8]) -> Self> {
        Box::new(|buf: &mut [u8]| buf[0] != 0)
    }
}

reflected_primitive!(DataType::Float(FloatSize::b32), f32, read_f32, write_f32);
reflected_primitive!(DataType::Float(FloatSize::b64), f64, read_f64, write_f64);
reflected_primitive!(DataType::UInt(IntSize::b16), u16, read_u16, write_u16);
reflected_primitive!(DataType::UInt(IntSize::b32), u32, read_u32, write_u32);
reflected_primitive!(DataType::UInt(IntSize::b64), u64, read_u64, write_u64);
reflected_primitive!(DataType::Int(IntSize::b16), i16, read_i16, write_i16);
reflected_primitive!(DataType::Int(IntSize::b32), i32, read_i32, write_i32);
reflected_primitive!(DataType::Int(IntSize::b64), i64, read_i64, write_i64);

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
            let dt: DataType =
                serde_json::from_str(s).unwrap_or_else(|_| panic!("Couldn't parse '{}'", s));
            assert_eq!(dt, expected, "Got {:?}, expected {:?}", dt, expected);

            let s2 = serde_json::to_string(&dt)
                .unwrap_or_else(|_| panic!("Couldn't serialize {:?}", dt));
            assert_eq!(s, &s2, "Got {:?}, expected {:?}", s2, s);
        }
    }

    #[test]
    fn parse_unknown() {
        use ExtensibleDataType::*;

        let s = r#"{"name":"newtype"}"#;
        let dt: ExtensibleDataType =
            serde_json::from_str(s).unwrap_or_else(|_| panic!("Couldn't parse '{}'", s));
        match &dt {
            Unknown(d) => assert_eq!(d.name, "newtype"),
            _ => panic!("got wrong dtype"),
        };
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

    #[test]
    fn can_validate_endian() {
        for dt in vec![
            DataType::Bool,
            DataType::UInt(IntSize::b8),
            DataType::Int(IntSize::b8),
            DataType::Raw(1),
            DataType::Raw(2),
            DataType::Raw(4),
        ] {
            for e in vec![Endian::Little, Endian::Big] {
                dt.valid_endian(Some(e)).unwrap();
            }
            dt.valid_endian(None).unwrap();
        }

        for dt in vec![
            DataType::UInt(IntSize::b16),
            DataType::Int(IntSize::b32),
            DataType::Float(FloatSize::b32),
            DataType::Complex(ComplexSize::b64),
        ] {
            for e in vec![Endian::Little, Endian::Big] {
                dt.valid_endian(Some(e)).unwrap();
            }
            assert!(dt.valid_endian(None).is_err());
        }
    }
}
