use std::collections::HashMap;

use codecs::CodecChain;
use data_type::{DataType, ReflectedType};
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

mod chunk_arr;
pub mod chunk_key_encoding;
pub mod codecs;
mod data_type;
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
#[serde(tag = "name", content = "configuration", rename_all = "lowercase")]
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
    // todo: store CodecChain instead
    #[serde(default = "CodecChain::default")]
    codecs: CodecChain,
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
    pub fn validate_dimensions(&self) -> Result<(), &'static str> {
        self.union_ndim(&self.chunk_grid)?;
        if let Some(d) = &self.dimension_names {
            if d.len() != self.ndim() {
                return Err("Inconsistent dimensionality");
            }
        }
        self.codecs.validate_ndim()?;
        self.union_ndim(&self.codecs)?;

        Ok(())
    }

    pub fn get_effective_fill_value<T: ReflectedType>(&self) -> Result<T, &'static str> {
        serde_json::from_value(self.fill_value.clone())
            .map_err(|_| "Could not deserialize fill value")
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

    // from the spec, although with "extensions" removed
    const EXAMPLE_ARRAY_META: &'static str = r#"
        {
            "zarr_format": 3,
            "node_type": "array",
            "shape": [10000, 1000],
            "dimension_names": ["rows", "columns"],
            "data_type": "float64",
            "chunk_grid": {
                "name": "regular",
                "configuration": {
                    "chunk_shape": [1000, 100]
                }
            },
            "chunk_key_encoding": {
                "name": "default",
                "configuration": {
                    "separator": "/"
                }
            },
            "codecs": [{
                "name": "gzip",
                "configuration": {
                    "level": 1
                }
            }],
            "fill_value": "NaN",
            "attributes": {
                "foo": 42,
                "bar": "apples",
                "baz": [1, 2, 3, 4]
            }
        }
    "#;

    // as above, minus array-specific keys
    const EXAMPLE_GROUP_META: &'static str = r#"
        {
            "zarr_format": 3,
            "node_type": "group",
            "attributes": {
                "foo": 42,
                "bar": "apples",
                "baz": [1, 2, 3, 4]
            }
        }
    "#;

    #[test]
    fn array_meta_roundtrip() {
        let meta: Metadata =
            serde_json::from_str(EXAMPLE_ARRAY_META).expect("Could not deserialise array metadata");
        match meta {
            Metadata::Array(_) => (),
            _ => panic!("Expected array metadata"),
        };
        let _s2 = serde_json::to_string(&meta).expect("Couldn't serialize array metadata");
    }

    #[test]
    fn group_meta_roundtrip() {
        let meta: Metadata =
            serde_json::from_str(EXAMPLE_GROUP_META).expect("Could not deserialise group metadata");
        match meta {
            Metadata::Group(_) => (),
            _ => panic!("Expected group metadata"),
        };
        let _s2 = serde_json::to_string(&meta).expect("Couldn't serialize group metadata");
    }
}
