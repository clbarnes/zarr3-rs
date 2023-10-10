use std::{
    collections::HashSet,
    io::{Read, Write},
};

use serde::{de, ser::SerializeSeq, Deserialize, Deserializer, Serialize};
use thiserror::Error;

pub mod aa;
pub mod ab;
pub mod bb;

use aa::{AACodec, AACodecType};
use ab::{ABCodec, ABCodecType};
use bb::{BBCodec, BBCodecType};

use crate::{
    data_type::{DataType, ReflectedType},
    ArcArrayD, GridCoord, MaybeNdim,
};

#[derive(Clone, PartialEq, Debug)]
pub struct CodecChain {
    pub aa_codecs: Vec<AACodecType>,
    // this is closer to the serialized form.
    // However, it necessitates extra clones in ab_codec function.
    // May be better to just reify it early and not round-trip?
    pub ab_codec: Option<ABCodecType>,
    pub bb_codecs: Vec<BBCodecType>,
}

impl CodecChain {
    pub fn new(
        aa_codecs: Vec<AACodecType>,
        ab_codec: Option<ABCodecType>,
        bb_codecs: Vec<BBCodecType>,
    ) -> Self {
        Self {
            aa_codecs,
            ab_codec,
            bb_codecs,
        }
    }

    pub fn ab_codec(&self) -> ABCodecType {
        // todo: unnecessary clones?
        // would be nice to return a ref but can't with the default
        self.ab_codec.clone().unwrap_or_default()
    }

    pub fn aa_codecs_mut(&mut self) -> &mut Vec<AACodecType> {
        &mut self.aa_codecs
    }

    pub fn bb_codecs_mut(&mut self) -> &mut Vec<BBCodecType> {
        &mut self.bb_codecs
    }

    pub fn replace_ab_codec<T: Into<ABCodecType>>(
        &mut self,
        ab_codec: Option<T>,
    ) -> Option<ABCodecType> {
        std::mem::replace(&mut self.ab_codec, ab_codec.map(|c| c.into()))
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn len(&self) -> usize {
        let mut out = self.aa_codecs.len() + self.bb_codecs.len();
        if self.ab_codec.is_some() {
            out += 1
        }
        out
    }
}

impl MaybeNdim for CodecChain {
    fn maybe_ndim(&self) -> Option<usize> {
        for c in self.aa_codecs.iter() {
            if let Some(n) = c.maybe_ndim() {
                return Some(n);
            }
        }
        if let Some(c) = self.ab_codec.as_ref() {
            if let Some(n) = c.maybe_ndim() {
                return Some(n);
            }
        }
        // BB codecs can't have dimensionality
        None
    }

    fn validate_ndim(&self) -> Result<(), &'static str> {
        let mut ndims = HashSet::with_capacity(self.len());

        for ndim in self.aa_codecs.iter().filter_map(|c| c.maybe_ndim()) {
            ndims.insert(ndim);
        }
        if let Some(c) = self.ab_codec.as_ref() {
            if let Some(n) = c.maybe_ndim() {
                ndims.insert(n);
            }
        }
        if ndims.len() > 1 {
            Err("Inconsistent codec dimensionalities")
        } else {
            Ok(())
        }
    }
}

impl Default for CodecChain {
    fn default() -> Self {
        Self::new(Vec::default(), None, Vec::default())
    }
}

impl Serialize for CodecChain {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut seq = serializer.serialize_seq(Some(self.len()))?;
        for aa in self.aa_codecs.iter() {
            seq.serialize_element(aa)?;
        }
        if let Some(ab) = &self.ab_codec {
            seq.serialize_element(ab)?;
        }
        for bb in self.bb_codecs.iter() {
            seq.serialize_element(bb)?;
        }
        seq.end()
    }
}

impl<'de> Deserialize<'de> for CodecChain {
    fn deserialize<D>(deserializer: D) -> Result<CodecChain, D::Error>
    where
        D: Deserializer<'de>,
    {
        let codecs: Vec<CodecType> = Vec::<CodecType>::deserialize(deserializer)?;
        let chain = codecs
            .into_iter()
            .collect::<Result<CodecChain, CodecChainConstructionError>>()
            .map_err(de::Error::custom)?;
        Ok(chain)
    }
}

impl ABCodec for CodecChain {
    fn encode<T: ReflectedType, W: Write>(&self, decoded: ArcArrayD<T>, w: W) {
        let bb_w = self.bb_codecs.as_slice().encoder(w);
        let arr = self.aa_codecs.as_slice().encode(decoded);
        self.ab_codec().encode::<T, _>(arr, bb_w);
    }

    fn decode<T: ReflectedType, R: Read>(&self, r: R, decoded_repr: ArrayRepr<T>) -> ArcArrayD<T> {
        let ab_repr = self
            .aa_codecs
            .as_slice()
            .compute_encoded_representation(decoded_repr);
        let bb_r = self.bb_codecs.as_slice().decoder(r);
        let arr = self.ab_codec().decode::<T, _>(bb_r, ab_repr);
        self.aa_codecs.as_slice().decode(arr)
    }
}

#[derive(Error, Debug)]
pub enum CodecChainConstructionError {
    #[error("More than one array->bytes codec found")]
    MultipleAB,
    #[error("Illegal codec order: {0} codec found after {1} codec")]
    IllegalOrder(&'static str, &'static str),
}

impl FromIterator<CodecType> for Result<CodecChain, CodecChainConstructionError> {
    fn from_iter<T: IntoIterator<Item = CodecType>>(iter: T) -> Self {
        let mut aa_codecs = Vec::default();
        let mut ab_codec = None;
        let mut bb_codecs = Vec::default();

        for ce in iter {
            match ce {
                CodecType::AA(c) => {
                    if ab_codec.is_some() {
                        return Err(CodecChainConstructionError::IllegalOrder("AA", "AB"));
                    }
                    if !bb_codecs.is_empty() {
                        return Err(CodecChainConstructionError::IllegalOrder("AA", "BB"));
                    }
                    aa_codecs.push(c);
                }
                CodecType::AB(c) => {
                    if ab_codec.is_some() {
                        return Err(CodecChainConstructionError::MultipleAB);
                    }
                    if !bb_codecs.is_empty() {
                        return Err(CodecChainConstructionError::IllegalOrder("AB", "BB"));
                    }
                    ab_codec = Some(c);
                }
                CodecType::BB(c) => bb_codecs.push(c),
            }
        }

        Ok(CodecChain::new(aa_codecs, ab_codec, bb_codecs))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[enum_delegate::implement(MaybeNdim)]
#[serde(untagged)]
pub enum CodecType {
    AA(AACodecType),
    AB(ABCodecType),
    BB(BBCodecType),
}

#[derive(Debug, Clone)]
pub struct ArrayRepr<T: ReflectedType> {
    pub shape: GridCoord,
    pub fill_value: T,
}

impl<T: ReflectedType> ArrayRepr<T> {
    pub fn new(shape: &[u64], fill_value: T) -> Self {
        let s = shape.iter().cloned().collect();
        ArrayRepr {
            shape: s,
            fill_value,
        }
    }

    pub fn data_type(&self) -> &DataType {
        &T::ZARR_TYPE
    }

    pub fn empty_array(&self) -> ArcArrayD<T> {
        let sh = self.shape.iter().map(|s| *s as usize).collect::<Vec<_>>();
        ArcArrayD::from_elem(sh.as_slice(), self.fill_value)
    }
}

impl<T: ReflectedType> From<GridCoord> for ArrayRepr<T> {
    fn from(value: GridCoord) -> Self {
        ArrayRepr {
            shape: value,
            fill_value: Default::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::codecs::ab::endian::EndianCodec;
    use crate::codecs::bb::gzip_codec::GzipCodec;
    use crate::ArcArrayD;

    use super::*;

    const SHAPE: [usize; 3] = [3, 4, 5];

    fn make_arr() -> ArcArrayD<f64> {
        ArcArrayD::from_shape_vec(SHAPE.to_vec(), (0..60).map(|v| v as f64).collect()).unwrap()
    }

    #[test]
    fn array_roundtrip_simple() {
        let arr = make_arr();
        let chain = CodecChain::default();
        let mut buf: Vec<u8> = Vec::default();

        chain.encode(arr.clone(), &mut buf);
        assert_ne!(buf.len(), 0);

        let repr = ArrayRepr {
            shape: SHAPE.iter().map(|s| *s as u64).collect(),
            fill_value: 0.0f64,
        };

        let arr2 = chain.decode::<f64, _>(buf.as_slice(), repr);

        assert_eq!(&arr, &arr2);
    }

    #[cfg(feature = "gzip")]
    #[test]
    fn array_meta_roundtrip_complicated() {
        use crate::codecs::aa::TransposeCodec;

        let arr = make_arr();
        let chain = CodecChain::new(
            vec![
                AACodecType::Transpose(TransposeCodec::new_f()),
                AACodecType::Transpose(TransposeCodec::new_f()),
                AACodecType::Transpose(TransposeCodec::new_f()),
            ],
            Some(ABCodecType::Endian(EndianCodec::new_big())),
            vec![
                BBCodecType::Gzip(GzipCodec::default()),
                BBCodecType::Gzip(GzipCodec::from_level(2).unwrap()),
            ],
        );
        let mut buf: Vec<u8> = Vec::default();

        chain.encode(arr.clone(), &mut buf);
        assert_ne!(buf.len(), 0);

        let repr = ArrayRepr {
            shape: SHAPE.iter().map(|s| *s as u64).collect(),
            fill_value: 0.0f64,
        };

        let arr2 = chain.decode::<f64, _>(buf.as_slice(), repr);

        assert_eq!(&arr, &arr2);
    }
}
