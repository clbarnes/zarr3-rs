mod array;
use std::collections::HashMap;

pub use array::{
    ArrayMetadata, ArrayMetadataBuilder, ChunkGrid, Extension, RegularChunkGrid, StorageTransformer,
};
mod group;
pub use group::GroupMetadata;
use serde::{de::DeserializeOwned, Deserialize, Serialize};

use crate::variant_from_data;

pub type JsonObject = HashMap<String, serde_json::Value>;

pub trait NodeMetadata {
    fn get_zarr_format(&self) -> usize;

    fn get_attributes(&self) -> &JsonObject;

    fn has_attribute(&self, key: &str) -> bool {
        self.get_attributes().contains_key(key)
    }

    // todo: is it worth having this?
    fn get_attribute_value(&self, key: &str) -> Option<serde_json::Value> {
        self.get_attributes().get(key).map(|v| v.clone())
    }

    fn get_attribute<T: DeserializeOwned>(
        &self,
        key: &str,
    ) -> Option<Result<T, serde_json::Error>> {
        self.get_attribute_value(key)
            .map(|v| serde_json::from_value(v.clone()))
    }

    fn get_attributes_mut(&mut self) -> &mut JsonObject;

    fn set_attribute<S: Serialize>(
        &mut self,
        key: &str,
        value: S,
    ) -> Result<Option<serde_json::Value>, serde_json::Error> {
        let v = serde_json::to_value(value)?;
        Ok(self.get_attributes_mut().insert(key.to_string(), v))
    }

    fn clear_attributes(&mut self) {
        self.get_attributes_mut().clear()
    }

    fn remove_attribute(&mut self, key: &str) -> Option<serde_json::Value> {
        self.get_attributes_mut().remove(key)
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(tag = "node_type", rename_all = "lowercase")]
pub enum Metadata {
    Array(ArrayMetadata),
    Group(GroupMetadata),
}

impl Metadata {
    pub fn is_array(&self) -> bool {
        match self {
            Self::Array(_) => true,
            _ => false,
        }
    }
}

impl Default for Metadata {
    fn default() -> Self {
        Self::Group(GroupMetadata::default())
    }
}

impl NodeMetadata for Metadata {
    fn get_zarr_format(&self) -> usize {
        match self {
            Metadata::Array(m) => m.get_zarr_format(),
            Metadata::Group(m) => m.get_zarr_format(),
        }
    }

    fn get_attributes(&self) -> &JsonObject {
        match self {
            Metadata::Array(m) => m.get_attributes(),
            Metadata::Group(m) => m.get_attributes(),
        }
    }

    fn get_attributes_mut(&mut self) -> &mut JsonObject {
        match self {
            Metadata::Array(m) => m.get_attributes_mut(),
            Metadata::Group(m) => m.get_attributes_mut(),
        }
    }
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

    #[cfg(feature = "filesystem")]
    mod filesystem {
        use crate::{
            data_type::{DataType, FloatSize},
            node::group::Group,
            store::{filesystem::FileSystemStore, NodeKey},
            ArcArrayD,
        };
        use smallvec::smallvec;

        use super::*;
        use tempdir::TempDir;

        #[test]
        fn roundtrip() {
            let tmp = tempdir::TempDir::new("zarr3-test").unwrap();
            let path = tmp.path().join("root.zarr");
            let store = FileSystemStore::create(path, true).unwrap();

            let g = Group::new(&store, Default::default(), Default::default());
            g.write_meta().unwrap();
            let g2 = g.create_group("child".parse().unwrap()).unwrap();

            let ameta =
                ArrayMetadataBuilder::new(smallvec![30, 40], DataType::Float(FloatSize::b32))
                    .chunk_grid(vec![5, 10].as_slice())
                    .unwrap()
                    .build();

            let arr = g2
                .create_array::<f32>("array".parse().unwrap(), ameta)
                .unwrap();
            let chunk = ArcArrayD::from_elem(vec![5, 10].as_slice(), 1.0);
            arr.write_chunk(&smallvec![0, 0, 0], chunk.clone()).unwrap();

            let g_again = Group::from_store(&store, Default::default()).unwrap();
            let g2_key: NodeKey = vec!["child".parse().unwrap()].into_iter().collect();
            let g2_again = g_again.get_group(g2_key).unwrap().unwrap();
            let arr_again = g2_again
                .get_array::<f32>(vec!["array".parse().unwrap()].into_iter().collect())
                .unwrap()
                .unwrap();
            let chunk2 = arr_again.read_chunk(&smallvec![0, 0, 0]).unwrap();
            assert_eq!(chunk, chunk2);

            let chunk3 = arr_again.read_chunk(&smallvec![1, 1, 1]).unwrap();
            assert_eq!(chunk3.shape(), chunk2.shape());
            assert!(chunk3.iter().all(|v| *v == 0.0))
        }
    }
}
