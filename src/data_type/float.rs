use std::fmt::Display;

use serde::{Deserialize, Serialize};
use serde_with::serde_as;

use super::NBytes;

/// Zarr's default 32-bit float NaN.
///
/// Like all NaNs, all bits of the exponent are 1,
/// and at least one bit of the mantissa is also 1.
/// In this NaN, the sign bit is 0,
/// and only the first bit of the mantissa is 1.
fn nan32() -> f32 {
    // sign bit is 0, all 8 exponent bits are 1, first mantissa bit is 1, rest are 0
    f32::from_bits(0b0111_1111__1100_0000__0000_0000__0000_0000)
}

/// Zarr's default 64-bit float NaN.
///
/// Like all NaNs, all bits of the exponent are 1,
/// and at least one bit of the mantissa is also 1.
/// In this NaN, the sign bit is 0,
/// and only the first bit of the mantissa is 1.
fn nan64() -> f64 {
    // sign bit is 0, all 11 exponent bits are 1, first mantissa bit is 1, rest are 0
    f64::from_bits(
        0b0111_1111__1111_1000__0000_0000__0000_0000__0000_0000__0000_0000__0000_0000__0000_0000,
    )
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

#[derive(Debug, Clone, Copy, PartialEq)]
enum SpecialF32 {
    NaN,
    Infinity,
    NegInfinity,
    Value(f32),
    Bytes([u8; 4]),
}

pub fn decode_hex<const B: usize>(s: &str) -> Result<[u8; B], ParseIntError> {
    let mut out = [0; B];
    for (idx, val) in (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16))
        .enumerate()
    {
        out[idx] = val?;
    }
    Ok(out)
}

pub fn encode_hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(2 + bytes.len() * 2);
    write!(&mut s, "0x").unwrap();
    for &b in bytes {
        write!(&mut s, "{:02x}", b).unwrap();
    }
    s
}

impl FromStr for SpecialF32 {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let out = match s {
            "NaN" => Self::NaN,
            "Infinity" => Self::Infinity,
            "-Infinity" => Self::NegInfinity,
            s if s.starts_with("0x") && s.len() == 10 => {
                let b = decode_hex::<4>(&s[2..]).map_err(|_e| "Could not parse hex")?;
                Self::Value(f32::from_be_bytes(b))
            }
            _ => return Err("Could not parse SpecialF32"),
        };
        Ok(out)
    }
}

impl ToString for SpecialF32 {
    fn to_string(&self) -> String {
        use SpecialF32::*;
        match self {
            NaN => "NaN".to_owned(),
            Infinity => "Infinity".to_owned(),
            NegInfinity => "-Infinity".to_owned(),
            Value(v) => format!("{:#06X?}", v.to_be_bytes()),
            Bytes(b) => format!("{:#06X?}", b),
        }
    }
}

impl Serialize for SpecialF32 {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use SpecialF32::*;
        match self {
            Value(v) => serializer.serialize_f32(*v),
            _ => serializer.serialize_str(&self.to_string()),
        }
    }
}

fn deser_f32<'de, D: Deserializer<'de>>(deserializer: D) -> Result<f32, D::Error> {
    Ok(match Value::deserialize(deserializer)? {
        Value::String(s) => s.parse().map_err(de::Error::custom)?,
        Value::Number(num) => num.as_f64().ok_or(de::Error::custom("Invalid number"))? as f32,
        _ => return Err(de::Error::custom("wrong type")),
    })
}

impl Into<f32> for SpecialF32 {
    fn into(self) -> f32 {
        match self {
            Self::NaN => nan32(),
            Self::Infinity => f32::INFINITY,
            Self::NegInfinity => f32::NEG_INFINITY,
            Self::Value(f) => f,
            Self::Bytes(b) => f32::from_be_bytes(b),
        }
    }
}

impl Into<SpecialF32> for f32 {
    fn into(self) -> SpecialF32 {
        use SpecialF32::*;
        if self.is_nan() {
            if self.to_bits() == nan32().to_bits() {
                NaN
            } else {
                Bytes(self.to_be_bytes())
            }
        } else if self.is_infinite() {
            if self.is_sign_negative() {
                NegInfinity
            } else {
                Infinity
            }
        } else {
            Value(self)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_nan32() {
        // WARNING: might break on different endian platform
        let n32 = nan32();
        // bytes values from spec
        let reference_u32 = u32::from_str_radix("7fc00000", 16).unwrap();
        assert_eq!(n32.to_bits(), reference_u32);
        assert!(n32.is_nan());
    }

    #[test]
    fn check_nan64_isnan() {
        assert!(nan64().is_nan());
    }
}
