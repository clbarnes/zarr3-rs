use std::io::{self, ErrorKind};

pub use crate::chunk_grid::ArrayRegion;
pub use crate::data_type::ReflectedType;
pub use crate::node::{
    Array, ArrayMetadata, ArrayMetadataBuilder, Group, GroupMetadata, ReadableMetadata,
    WriteableMetadata,
};
use crate::store::NodeKey;
pub use crate::store::{ListableStore, ReadableStore, WriteableStore};

pub use ndarray;
pub use serde::{Deserialize, Serialize};
pub use serde_json;
pub use smallvec;

pub fn create_root_group<S: WriteableStore>(
    store: &S,
    metadata: GroupMetadata,
) -> io::Result<Group<S>> {
    let mut key = NodeKey::default();
    key.with_metadata();
    if store.has_key(&key)? {
        return Err(io::Error::new(
            ErrorKind::AlreadyExists,
            "Node exists at root",
        ));
    }
    let g = Group::new(store, Default::default(), metadata);
    g.write_meta()?;
    Ok(g)
}

pub fn create_root_array<T: ReflectedType, S: WriteableStore>(
    store: &S,
    metadata: ArrayMetadata,
) -> io::Result<Array<S, T>> {
    let mut key = NodeKey::default();
    key.with_metadata();
    if store.has_key(&key)? {
        return Err(io::Error::new(
            ErrorKind::AlreadyExists,
            "Node exists at root",
        ));
    }
    let a = Array::new(store, Default::default(), metadata).unwrap();
    a.write_meta()?;
    Ok(a)
}
