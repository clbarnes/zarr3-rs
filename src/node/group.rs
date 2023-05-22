use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    io::{self, ErrorKind},
};

use crate::{
    data_type::ReflectedType,
    store::{ListableStore, NodeKey, NodeName, ReadableStore, Store, WriteableStore},
    ZARR_FORMAT,
};

use super::{array::Array, ArrayMetadata, JsonObject, ReadableMetadata, WriteableMetadata};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct GroupMetadata {
    zarr_format: usize,
    #[serde(default = "HashMap::default")]
    attributes: JsonObject,
}

impl ReadableMetadata for GroupMetadata {
    fn get_attributes(&self) -> &JsonObject {
        &self.attributes
    }

    fn get_zarr_format(&self) -> usize {
        self.zarr_format
    }

    fn is_array(&self) -> bool {
        false
    }
}

impl WriteableMetadata for GroupMetadata {
    fn mutate_attributes<F, R>(&mut self, f: F) -> R
    where
        F: FnOnce(&mut JsonObject) -> R,
    {
        f(&mut self.attributes)
    }
}

// For implicit groups
impl Default for GroupMetadata {
    fn default() -> Self {
        Self {
            zarr_format: ZARR_FORMAT,
            attributes: JsonObject::default(),
        }
    }
}

pub struct Group<'s, S: Store> {
    store: &'s S,
    key: NodeKey,
    meta_key: NodeKey,
    metadata: GroupMetadata,
}

impl<'s, S: Store> Group<'s, S> {
    pub(crate) fn new(store: &'s S, key: NodeKey, metadata: GroupMetadata) -> Self {
        let mut meta_key = key.clone();
        meta_key.with_metadata();
        Self {
            store,
            key,
            meta_key,
            metadata,
        }
    }

    fn key(&self) -> &NodeKey {
        &self.key
    }

    fn child_key(&self, name: NodeName) -> NodeKey {
        let mut key = self.key.clone();
        key.push(name);
        key
    }

    fn descendant_key(&self, subkey: &NodeKey) -> NodeKey {
        let mut key = self.key.clone();
        for n in subkey.as_slice().iter() {
            key.push(n.clone());
        }
        key
    }

    fn meta_key(&self) -> &NodeKey {
        &self.meta_key
    }

    fn store(&self) -> &'s S {
        self.store
    }
}

impl<'s, S: ReadableStore> Group<'s, S> {
    pub(crate) fn read_meta(&mut self) -> io::Result<()> {
        if let Some(r) = self.store.get(&self.meta_key())? {
            let meta: GroupMetadata = serde_json::from_reader(r).expect("deser error");
            self.metadata = meta;
            Ok(())
        } else {
            Err(io::Error::new(
                ErrorKind::NotFound,
                "Group metadata not found",
            ))
        }
    }

    pub fn from_store(store: &'s S, key: NodeKey) -> io::Result<Self> {
        let mut meta_key = key.clone();
        meta_key.with_metadata();
        if let Some(r) = store.get(&meta_key)? {
            let meta: GroupMetadata = serde_json::from_reader(r).expect("deser error");
            Ok(Self::new(store, key, meta))
        } else {
            Err(io::Error::new(
                ErrorKind::NotFound,
                "Group metadata not found",
            ))
        }
    }

    pub fn get_group(&self, subkey: NodeKey) -> io::Result<Option<Self>> {
        let mut key = self.key().clone();
        key.extend(subkey);
        match Self::from_store(self.store, key) {
            Ok(s) => Ok(Some(s)),
            Err(e) if e.kind() == ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e),
        }
    }

    pub fn get_array<T: ReflectedType>(&self, subkey: NodeKey) -> io::Result<Option<Array<S, T>>> {
        let mut key = self.key().clone();
        key.extend(subkey);
        match Array::from_store(self.store, key) {
            Ok(s) => Ok(Some(s)),
            Err(e) if e.kind() == ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e),
        }
    }
}

impl<'s, S: ListableStore> Group<'s, S> {
    pub fn child_keys(&self) -> io::Result<Vec<NodeKey>> {
        let (_, keys) = self.store.list_dir(&self.key)?;
        Ok(keys)
    }
}

impl<'s, S: WriteableStore> Group<'s, S> {
    pub(crate) fn write_meta(&self) -> io::Result<()> {
        self.store.set(&self.meta_key, |mut w| {
            Ok(serde_json::to_writer_pretty(&mut w, &self.metadata)
                .expect("could not serialise metadata"))
        })
    }

    /// Deletes any existing group.
    pub fn create_group(&self, name: NodeName) -> io::Result<Self> {
        let key = self.child_key(name);
        self.store.erase_prefix(&key)?;
        let g = Self::new(self.store, key, GroupMetadata::default());
        g.write_meta()?;
        Ok(g)
    }

    pub fn create_array<T: ReflectedType>(
        &self,
        name: NodeName,
        metadata: ArrayMetadata,
    ) -> io::Result<Array<S, T>> {
        let mut key = self.key.clone();
        key.push(name);
        let arr = Array::new(self.store, key.clone(), metadata).expect("Bad array arguments");
        self.store.erase_prefix(&key)?;
        arr.write_meta()?;
        Ok(arr)
    }

    pub fn erase(self) -> io::Result<()> {
        self.store.erase_prefix(&self.key)?;
        Ok(())
    }

    pub fn erase_child(&self, name: NodeName) -> io::Result<bool> {
        let mut key = self.key.clone();
        key.push(name);
        self.store.erase_prefix(&key)
    }
}
