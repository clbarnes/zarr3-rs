use std::io::SeekFrom;

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

pub struct Interval {
    pub start: isize,
    pub end: Option<isize>,
}

#[derive(Error, Debug)]
pub enum OutOfBounds {
    #[error("Index is -{0}")]
    BeforeStart(usize),
    #[error("Index is {idx} with max length of {max_len}")]
    AfterEnd { idx: usize, max_len: usize },
}

impl OutOfBounds {
    pub fn clamp(&self) -> usize {
        match self {
            Self::BeforeStart(_) => 0,
            Self::AfterEnd { idx, max_len } => *max_len,
        }
    }
}

fn pos_idx(idx: isize, len: usize) -> Result<usize, OutOfBounds> {
    if idx >= 0 {
        let pos_offset = idx as usize;
        if pos_offset <= len {
            return Ok(pos_offset);
        } else {
            return Err(OutOfBounds::AfterEnd {
                idx: pos_offset,
                max_len: len,
            });
        }
    }

    let neg_offset = idx.abs() as usize;
    if neg_offset > len {
        return Err(OutOfBounds::BeforeStart(neg_offset - len));
    }
    Ok(len - neg_offset)
}

fn int_to_seekfrom(i: isize) -> SeekFrom {
    if i < 0 {
        SeekFrom::End(i as i64)
    } else {
        SeekFrom::Start(i as u64)
    }
}

impl Interval {
    pub fn as_seekfrom_nbytes(&self, len: Option<usize>) -> (SeekFrom, Option<usize>) {
        let end = len.map(|l| {
            self.end
                .map(|e| pos_idx(e, l).unwrap_or_else(|e| e.clamp()))
                .unwrap_or(l)
        });
        (int_to_seekfrom(self.start), end)
    }

    pub fn as_offsets(&self, len: usize) -> (usize, usize) {
        let start = pos_idx(self.start, len).unwrap_or_else(|e| e.clamp());
        let end = self
            .end
            .map(|e| pos_idx(e, len).unwrap_or_else(|e| e.clamp()))
            .unwrap_or(len);
        (start, end)
    }
}

pub trait ByteReader {
    fn read(&self) -> Vec<u8>;

    fn read_partial<'a>(&self, ranges: &[Interval]) -> Vec<Vec<u8>> {
        let all = self.read();

        ranges
            .iter()
            .map(|r| {
                let (start, stop) = r.as_offsets(all.len());
                all[start..stop].iter().cloned().collect()
            })
            .collect()
    }
}

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

// impl ABCodec for CodecChain {

// }

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
