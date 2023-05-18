mod array;
use std::collections::HashMap;

pub use array::{
    ArrayMetadata, ArrayMetadataBuilder, ChunkGrid, Extension, RegularChunkGrid, StorageTransformer,
};
use serde::{Deserialize, Serialize};

use crate::variant_from_data;

pub type JsonObject = HashMap<String, serde_json::Value>;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct GroupMetadata {
    zarr_format: usize,
    #[serde(default = "HashMap::default")]
    attributes: JsonObject,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(tag = "node_type", rename_all = "lowercase")]
pub enum Metadata {
    Array(ArrayMetadata),
    Group(GroupMetadata),
}

variant_from_data!(Metadata, Array, ArrayMetadata);
variant_from_data!(Metadata, Group, GroupMetadata);

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
