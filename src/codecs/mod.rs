use ndarray::Array;
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub mod aa;
pub mod ab;
pub mod bb;

use aa::AACodecType;
use ab::ABCodecType;
use bb::BBCodecType;

use crate::MaybeNdim;

// pub trait ArrayReader {
//     fn read(&self) -> Array<>
// }

struct CodecChain {
    pub aa_codecs: Vec<AACodecType>,
    pub ab_codec: ABCodecType,
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
            ab_codec: ab_codec.unwrap_or_else(|| ABCodecType::default()),
            bb_codecs,
        }
    }
}

/// This implementation bulldozes potential errors,
/// distributing AA and BB codecs correctly
/// and only using the last AB codec.
/// It may have unexpected behaviour when passing region descriptions through.
// impl FromIterator<CodecType> for CodecChain {

//     fn from_iter<T: IntoIterator<Item = CodecType>>(iter: T) -> Self {
//         let mut aa_codecs = Vec::default();
//         let mut ab_codec = None;
//         let mut bb_codecs = Vec::default();

//         for ce in iter {
//             match ce {
//                 CodecType::AA(c) => aa_codecs.push(c),
//                 CodecType::AB(c) => {ab_codec = Some(c); },
//                 CodecType::BB(c) => bb_codecs.push(c),
//             }
//         }

//         CodecChain { aa_codecs, ab_codec, bb_codecs }
//     }
// }

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
pub enum CodecType {
    AA(AACodecType),
    AB(ABCodecType),
    BB(BBCodecType),
}
