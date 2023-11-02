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

pub(super) mod fwrite;

use crate::{
    data_type::{DataType, ReflectedType},
    ArcArrayD, GridCoord, MaybeNdim,
};

#[derive(Clone, PartialEq, Debug)]
pub struct CodecChain {
    pub aa_codecs: Vec<AACodecType>,
    pub ab_codec: ABCodecType,
    pub bb_codecs: Vec<BBCodecType>,
}

impl CodecChain {
    pub fn new(
        aa_codecs: Vec<AACodecType>,
        ab_codec: ABCodecType,
        bb_codecs: Vec<BBCodecType>,
    ) -> Self {
        Self {
            aa_codecs,
            ab_codec,
            bb_codecs,
        }
    }

    pub fn ab_codec(&self) -> &ABCodecType {
        // todo: unnecessary clones?
        // would be nice to return a ref but can't with the default
        &self.ab_codec
    }

    pub fn aa_codecs_mut(&mut self) -> &mut Vec<AACodecType> {
        &mut self.aa_codecs
    }

    pub fn bb_codecs_mut(&mut self) -> &mut Vec<BBCodecType> {
        &mut self.bb_codecs
    }

    pub fn replace_ab_codec<T: Into<ABCodecType>>(&mut self, ab_codec: T) -> ABCodecType {
        std::mem::replace(&mut self.ab_codec, ab_codec.into())
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn len(&self) -> usize {
        self.aa_codecs.len() + self.bb_codecs.len() + 1
    }

    // pub(crate) fn iter_ndims<'a>(&self) -> impl Iterator<Item = usize> + '_ {
    //     self.aa_codecs
    //         .iter()
    //         .filter_map(|c| c.maybe_ndim())
    //         .chain(self.ab_codec.maybe_ndim().iter().cloned())
    // }

    // pub fn validate_index(&self) -> Result<()> {}
}

impl MaybeNdim for CodecChain {
    fn maybe_ndim(&self) -> Option<usize> {
        self.aa_codecs
            .iter()
            .filter_map(|c| c.maybe_ndim())
            .next()
            .or_else(|| self.ab_codec.maybe_ndim())
    }

    fn validate_ndim(&self) -> Result<(), &'static str> {
        let mut ndims = HashSet::with_capacity(2);

        for n in self
            .aa_codecs
            .iter()
            .filter_map(|c| c.maybe_ndim())
            .chain(self.ab_codec.maybe_ndim().iter().cloned())
        {
            if ndims.insert(n) && ndims.len() > 1 {
                return Err("Inconsistent codec dimensionalities");
            }
        }
        Ok(())
    }
}

impl Default for CodecChain {
    fn default() -> Self {
        Self::new(Vec::default(), Default::default(), Vec::default())
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
        seq.serialize_element(&self.ab_codec)?;
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
        let mut bb_w = self.bb_codecs.as_slice().encoder(w);
        let arr = self.aa_codecs.as_slice().encode(decoded);
        self.ab_codec().encode::<T, _>(arr, &mut bb_w);
        bb_w.finalize().unwrap();
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

    fn endian(&self) -> Option<ab::bytes_codec::Endian> {
        self.ab_codec.endian()
    }
}

#[derive(Error, Debug)]
pub enum CodecChainConstructionError {
    #[error("More than one array->bytes codec found")]
    MultipleAB,
    #[error("Illegal codec order: {0} codec found after {1} codec")]
    IllegalOrder(&'static str, &'static str),
    #[error("No array->bytes codec")]
    NoAB,
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

        Ok(CodecChain::new(
            aa_codecs,
            ab_codec.ok_or(CodecChainConstructionError::NoAB)?,
            bb_codecs,
        ))
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
    use crate::codecs::ab::bytes_codec::BytesCodec;
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
                AACodecType::Transpose(TransposeCodec::new_transpose(SHAPE.len())),
                AACodecType::Transpose(TransposeCodec::new_transpose(SHAPE.len())),
                AACodecType::Transpose(TransposeCodec::new_transpose(SHAPE.len())),
            ],
            ABCodecType::Bytes(BytesCodec::new_big()),
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
