use std::{fmt::Display, str::FromStr};

use serde::{de, Deserialize, Deserializer, Serialize, Serializer};
use serde_with::serde_as;

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

#[serde_as]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DataType {
    Bool,
    Int(IntSize),
    UInt(IntSize),
    Float(FloatSize),
    Complex(ComplexSize),
    Raw(usize),
}

impl Serialize for DataType {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.to_string())
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
            Self::Bool => 8,
            Self::Int(s) => s.nbytes(),
            Self::UInt(s) => s.nbytes(),
            Self::Float(s) => s.nbytes(),
            Self::Complex(s) => s.nbytes(),
            Self::Raw(s) => *s / 8,
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
}
