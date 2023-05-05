use std::collections::HashMap;

use codecs::CodecType;
use data_types::DataType;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

mod chunk_arr;
pub mod chunk_key_encoding;
pub mod codecs;
mod data_types;
mod ndarr;
mod util;

use chunk_key_encoding::ChunkKeyEncoding;

const COORD_SMALLVEC_SIZE: usize = 6;

pub type CoordVec<T> = SmallVec<[T; COORD_SMALLVEC_SIZE]>;
pub type ChunkCoord = CoordVec<u32>;
pub type GridCoord = CoordVec<u64>;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RegularChunkGrid {
    chunk_shape: GridCoord,
}

impl Ndim for RegularChunkGrid {
    fn ndim(&self) -> usize {
        self.chunk_shape.len()
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(tag = "name", content = "configuration")]
#[enum_delegate::implement(MaybeNdim)]
pub enum ChunkGrid {
    Regular(RegularChunkGrid),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "name", content = "configuration")]
pub enum StorageTransformer {}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Extension(serde_json::Value);

impl Extension {
    pub fn try_understand(&self) -> Result<(), &'static str> {
        let mut map: HashMap<String, serde_json::Value> =
            serde_json::from_value(self.0.clone()).map_err(|_| "Extension is not an object")?;
        let mu_value = map
            .remove("must_understand")
            .ok_or("Extension does not define \"must_understand\"")?;
        let mu: bool = serde_json::from_value(mu_value)
            .map_err(|_| "Extension's \"must_understand\" is not a boolean")?;
        if mu {
            Err("Extension must be understood")
        } else {
            Ok(())
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ArrayMetadata {
    zarr_format: usize,
    shape: GridCoord,
    data_type: DataType,
    chunk_grid: ChunkGrid,
    chunk_key_encoding: ChunkKeyEncoding,
    fill_value: serde_json::Value,
    #[serde(default = "Vec::default")]
    storage_transformers: Vec<StorageTransformer>,
    #[serde(default = "Vec::default")]
    codecs: Vec<CodecType>,
    #[serde(default = "HashMap::default")]
    attributes: HashMap<String, serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dimension_names: Option<CoordVec<Option<String>>>,
    #[serde(flatten)]
    extensions: HashMap<String, Extension>,
}

impl Ndim for ArrayMetadata {
    fn ndim(&self) -> usize {
        self.shape.len()
    }
}

impl ArrayMetadata {
    /// Ensures that all unknown extensions do not require understanding.
    pub fn try_understand_extensions(&self) -> Result<(), &'static str> {
        self.extensions
            .iter()
            .map(|(_name, config)| config.try_understand())
            .collect()
    }

    /// Ensure that all dimensioned metadata is consistent.
    fn validate_dimensions(&self) -> Result<(), &'static str> {
        self.union_ndim(&self.chunk_grid)?;
        if let Some(d) = &self.dimension_names {
            if d.len() != self.ndim() {
                return Err("Inconsistent dimensionality");
            }
        }
        for c in self.codecs.iter() {
            self.union_ndim(c)?;
        }

        Ok(())
    }
}

#[enum_delegate::register]
pub trait Ndim {
    fn ndim(&self) -> usize;

    fn same_ndim<T: Ndim>(&self, other: &T) -> Result<usize, &'static str> {
        let n = self.ndim();
        if n == other.ndim() {
            Ok(n)
        } else {
            Err("Inconsistent dimensionalities")
        }
    }
}

#[enum_delegate::register]
pub trait MaybeNdim {
    fn maybe_ndim(&self) -> Option<usize>;

    fn union_ndim<T: MaybeNdim>(&self, other: &T) -> Result<Option<usize>, &'static str> {
        if let Some(n1) = self.maybe_ndim() {
            if let Some(n2) = other.maybe_ndim() {
                if n1 == n2 {
                    return Ok(Some(n1));
                } else {
                    return Err("Inconsistent dimensionalities");
                }
            } else {
                return Ok(Some(n1));
            }
        } else {
            return Ok(other.maybe_ndim());
        }
    }

    fn validate_ndim(&self) -> Result<(), &'static str> {
        Ok(())
    }
}

impl<T: Ndim> MaybeNdim for T {
    fn maybe_ndim(&self) -> Option<usize> {
        Some(self.ndim())
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct GroupMetadata {
    zarr_format: usize,
    #[serde(default = "HashMap::default")]
    attributes: HashMap<String, serde_json::Value>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(tag = "node_type", rename_all = "lowercase")]
pub enum Metadata {
    Array(ArrayMetadata),
    Group(GroupMetadata),
}

#[cfg(test)]
mod tests {
    use super::*;

    // #[test]
    // fn it_works() {
    //     let result = add(2, 2);
    //     assert_eq!(result, 4);
    // }
}
