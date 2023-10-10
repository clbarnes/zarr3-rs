use itertools::Itertools;
use log::warn;
use smallvec::SmallVec;
use std::{
    collections::{HashMap, HashSet},
    fmt::Display,
    io::{self, Cursor, Error, Read, Write},
    str::FromStr,
};

mod hashmap;
pub use hashmap::HashMapStore;

use crate::RangeRequest;

#[cfg(feature = "filesystem")]
pub mod filesystem;

#[cfg(feature = "http")]
pub mod http;

const NODE_KEY_SIZE: usize = 10;
const METADATA_NAME: &str = "zarr.json";
pub(crate) const KEY_SEP: &str = "/";

fn metadata_name() -> NodeName {
    METADATA_NAME.parse().unwrap()
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NodeName(String);

impl NodeName {
    pub fn new(s: String) -> Result<Self, InvalidNodeName> {
        Self::validate(&s)?;
        Ok(Self::new_unchecked(s))
    }

    pub(crate) fn new_unchecked(s: String) -> Self {
        Self(s)
    }

    fn validate(s: &str) -> Result<(), InvalidNodeName> {
        let mut is_periods = true;
        let mut is_underscore = true;
        let mut has_non_recommended = false;
        let mut len: usize = 0;
        for c in s.chars() {
            if is_periods && c != '.' {
                is_periods = false;
            }
            if is_underscore {
                if len >= 2 {
                    return Err(InvalidNodeName::ReservedPrefix);
                }
                if c != '_' {
                    is_underscore = false;
                }
            }
            if c == '/' {
                return Err(InvalidNodeName::HasSlash);
            }

            if !has_non_recommended && !c.is_ascii_alphanumeric() && c != '-' && c != '_' && c != '.' {
                has_non_recommended = true;
                warn!("Node name has non-recommended character `{}`; prefer `a-z`, `A-Z`, `0-9`, `-`, `_`, `.`", c);
            }

            len += 1;
        }
        if len == 0 {
            return Err(InvalidNodeName::Empty);
        }
        if is_periods {
            return Err(InvalidNodeName::IsPeriods);
        }
        Ok(())
    }
}

impl Display for NodeName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl AsRef<str> for NodeName {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl FromStr for NodeName {
    type Err = InvalidNodeName;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::new(s.to_owned())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[derive(Default)]
pub struct NodeKey(SmallVec<[NodeName; NODE_KEY_SIZE]>);

#[derive(thiserror::Error, Debug)]
pub enum InvalidNodeName {
    #[error("Node name is empty string")]
    Empty,
    #[error("Node name contains '/'")]
    HasSlash,
    #[error("Node name is comprised only of periods")]
    IsPeriods,
    #[error("Node name starts with reserved prefix '__'")]
    ReservedPrefix,
}

impl FromIterator<NodeName> for NodeKey {
    fn from_iter<T: IntoIterator<Item = NodeName>>(iter: T) -> Self {
        Self(iter.into_iter().collect())
    }
}

impl NodeKey {
    /// Adds a new key component in-place.
    ///
    /// Returns the new number of components.
    pub fn push(&mut self, name: NodeName) -> usize {
        self.0.push(name);
        self.0.len()
    }

    /// Append all elements of the other key onto this one in-place.
    ///
    /// Returns the new number of components.
    pub fn extend(&mut self, other: NodeKey) -> usize {
        let mut len = self.0.len();
        for k in other.0.into_iter() {
            len = self.push(k);
        }
        len
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Pop the last key component.
    ///
    /// None if we are at the root.
    pub fn pop(&mut self) -> Option<NodeName> {
        self.0.pop()
    }

    /// Find the longest ancestor key shared this and another key.
    ///
    /// May be the root (empty key).
    pub fn common_root(&self, other: &NodeKey) -> NodeKey {
        self.as_slice()
            .iter()
            .zip(other.as_slice().iter())
            .take_while(|(a, b)| a == b)
            .map(|(a, _)| a.clone())
            .collect()
    }

    /// Check whether this key starts with (or equals) the other key.
    pub fn starts_with(&self, other: &NodeKey) -> bool {
        self.len() >= other.len() && &self.as_ref()[..other.len()] == other.as_ref()
    }

    // Check whether the other key starts with this key and is longer.
    pub fn is_ancestor_of(&self, other: &NodeKey) -> bool {
        other.len() > self.len() && self.as_ref() == &other.as_ref()[..self.len()]
    }

    /// Create a new key relative to this one.
    ///
    /// `"."` refers to the current key (no-op),
    /// `".."` refers to the parent key.
    /// Any other invalid key returns an [InvalidNodeName] error.
    /// Traversing above the root returns None.
    pub fn relative(&self, items: &[&str]) -> Result<Option<Self>, InvalidNodeName> {
        let mut new = self.clone();
        for n in items.iter() {
            match n {
                &"." => continue,
                &".." => {
                    let popped = new.pop();
                    if popped.is_none() {
                        return Ok(None);
                    }
                }
                other => {
                    new.push(other.parse()?);
                }
            };
        }
        Ok(Some(new))
    }

    pub fn is_root(&self) -> bool {
        self.0.is_empty()
    }

    pub fn with_metadata(&mut self) -> usize {
        self.push(metadata_name())
    }

    pub fn as_slice(&self) -> &[NodeName] {
        self.0.as_slice()
    }

    /// Encode the key as a string by joining its parts with `/`.
    pub fn encode(&self) -> String {
        self.0.iter().map(|n| n.as_ref()).join(KEY_SEP)
    }

    /// Encode the key as a string, adding a trailing `/`.
    pub fn encode_prefix(&self) -> String {
        self.encode() + "/"
    }
}

impl Display for NodeKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = self.0.iter().map(|n| n.as_ref()).join(KEY_SEP);
        f.write_str(&s)
    }
}

impl FromStr for NodeKey {
    type Err = InvalidNodeName;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut k = Self::default();
        for n in s.split(KEY_SEP) {
            k.push(NodeName::new(n.to_owned())?);
        }
        Ok(k)
    }
}



impl AsRef<[NodeName]> for NodeKey {
    fn as_ref(&self) -> &[NodeName] {
        &self.0
    }
}

pub trait Store {}

pub trait ReadableStore: Store {
    type Readable: Read;
    // todo: different type for partial reads?

    /// TODO: not in zarr spec
    fn has_key(&self, key: &NodeKey) -> io::Result<bool> {
        self.get(key).map(|o| o.is_some())
    }

    /// Get a [Read]er representing the contents of the key.
    fn get(&self, key: &NodeKey) -> Result<Option<Self::Readable>, Error>;

    /// Get a number of [Read]ers for partial reads.
    ///
    /// The trait's default implementation is inefficient in most cases
    /// and should be replaced by implementors.
    fn get_partial_values(
        &self,
        key_ranges: &[(NodeKey, RangeRequest)],
    ) -> Result<Vec<Option<Box<dyn Read>>>, Error> {
        // could rely on other caching here?
        let mut bufs = HashMap::with_capacity(key_ranges.len());
        let mut out = Vec::with_capacity(key_ranges.len());
        for (key, range) in key_ranges.iter() {
            if !bufs.contains_key(key) {
                match self.get(key) {
                    Ok(Some(mut r)) => {
                        let mut buf = Vec::default();
                        r.read_to_end(&mut buf)?;
                        bufs.insert(key.clone(), Some(buf));
                        Ok(())
                    }
                    Ok(None) => {
                        bufs.insert(key.clone(), None);
                        Ok(())
                    }
                    Err(e) => Err(e),
                }?;
            }
            let rd = bufs.get(key).unwrap().as_ref().map(|b| Box::new(Cursor::new(
                    // todo: unnecessary clone?
                    // consider bytes::Bytes
                    range.slice(b.as_slice()).to_vec(),
                )) as Box<dyn Read>);
            out.push(rd);
        }
        Ok(out)
    }

    // /// TODO: not in zarr spec
    // fn uri(&self, key: &NodeKey) -> Result<String, Error>;
}

/// Calculate [ListableStore::list_prefix] result from all keys.
pub fn list_prefix_from_all_keys<I: IntoIterator<Item = NodeKey>>(
    all_keys: I,
    prefix: &NodeKey,
) -> Vec<NodeKey> {
    all_keys
        .into_iter()
        .filter(|k| prefix.is_ancestor_of(k))
        .collect()
}

/// Calculate [ListableStore::list_prefix] result from all keys (as references).
pub fn list_prefix_from_all_keys_ref<'i, I: IntoIterator<Item = &'i NodeKey>>(
    all_keys: I,
    prefix: &NodeKey,
) -> Vec<NodeKey> {
    all_keys
        .into_iter()
        .filter(|k| prefix.is_ancestor_of(k))
        .cloned()
        .collect()
}

/// Calculate [ListableStore::list_prefix] result using [ListableStore::list_dir]
pub fn list_prefix_from_list_dir(
    store: &impl ListableStore,
    key: &NodeKey,
) -> Result<Vec<NodeKey>, Error> {
    let mut to_visit = Vec::default();
    to_visit.push(key.clone());
    let mut result = Vec::default();

    while let Some(next) = to_visit.pop() {
        let dir = store.list_dir(&next)?;
        result.extend(dir.0);
        to_visit.extend(dir.1);
    }

    Ok(result)
}

/// Calculate [ListableStore::list_dir] result from all keys.
pub fn list_dir_from_all_keys<I: IntoIterator<Item = NodeKey>>(
    all_keys: I,
    key: &NodeKey,
) -> (Vec<NodeKey>, Vec<NodeKey>) {
    let mut keys = Vec::default();
    let mut prefix_names: HashSet<NodeName> = HashSet::default();
    for k in all_keys.into_iter().filter(|k| key.is_ancestor_of(k)) {
        if k.len() == key.len() + 1 {
            keys.push(k);
        } else {
            let name = k.as_slice().get(key.len()).unwrap();
            if prefix_names.contains(name) {
                continue;
            }
            prefix_names.insert(name.clone());
        }
    }

    let prefixes = prefix_names
        .into_iter()
        .map(|n| {
            let mut k = key.clone();
            k.push(n);
            k
        })
        .collect();

    (keys, prefixes)
}

/// Calculate [ListableStore::list_dir] result from all keys (as references).
pub fn list_dir_from_all_keys_ref<'i, I: IntoIterator<Item = &'i NodeKey>>(
    all_keys: I,
    key: &NodeKey,
) -> (Vec<NodeKey>, Vec<NodeKey>) {
    let mut keys = Vec::default();
    let mut prefix_names: HashSet<NodeName> = HashSet::default();
    for k in all_keys.into_iter().filter(|k| key.is_ancestor_of(k)) {
        if k.len() == key.len() + 1 {
            keys.push(k.clone());
        } else {
            let name = k.as_slice().get(key.len()).unwrap();
            if prefix_names.contains(name) {
                continue;
            }
            prefix_names.insert(name.clone());
        }
    }

    let prefixes = prefix_names
        .into_iter()
        .map(|n| {
            let mut k = key.clone();
            k.push(n);
            k
        })
        .collect();

    (keys, prefixes)
}

/// Calculate [ListableStore::list_dir] result using [ListableStore::list_prefix]
pub fn list_dir_from_list_prefix(
    store: &impl ListableStore,
    key: &NodeKey,
) -> Result<(Vec<NodeKey>, Vec<NodeKey>), Error> {
    let mut keys = Vec::default();
    let mut prefix_names: HashSet<NodeName> = HashSet::default();

    for k in store.list_prefix(key)?.into_iter() {
        if k.len() == key.len() + 1 {
            keys.push(k);
        } else {
            let name = k.as_slice().get(key.len()).unwrap();
            if prefix_names.contains(name) {
                continue;
            }
            prefix_names.insert(name.clone());
        }
    }

    let prefixes = prefix_names
        .into_iter()
        .map(|n| {
            let mut k = key.clone();
            k.push(n);
            k
        })
        .collect();

    Ok((keys, prefixes))
}

/// Calculate [ListableStore::list] result using [ListableStore::list_prefix]
pub fn list_from_list_prefix(store: &impl ListableStore) -> Result<Vec<NodeKey>, Error> {
    store.list_prefix(&NodeKey::default())
}

pub trait ListableStore: Store {
    /// Retrieve all keys in the store.
    ///
    /// If this store has an efficient [ListableStore::list_prefix] implementation,
    /// consider implementing this method using [list_from_list_prefix].
    fn list(&self) -> Result<Vec<NodeKey>, Error>;

    /// Retrieve all keys with a given prefix.
    ///
    /// If this store has an efficient [ListableStore::list_dir] implementation,
    /// consider implementing this method using [list_prefix_from_list_dir].
    /// If this must be calculated via a call to [ListableStore::list],
    /// consider implementing this method using [list_prefix_from_list].
    fn list_prefix(&self, key: &NodeKey) -> Result<Vec<NodeKey>, Error>;

    /// Retrieve all keys and prefixes with a given prefix and which do not
    /// contain the character “/” after the given prefix.
    ///
    /// If this store has an efficient [ListableStore::list_prefix] implementation,
    /// consider implementing this method using [list_dir_from_list_prefix].
    /// If this must be implemented via a call to [ListableStore::list],
    /// consider implementing this method using [list_dir_from_list].
    fn list_dir(&self, prefix: &NodeKey) -> Result<(Vec<NodeKey>, Vec<NodeKey>), Error>;
}

// Readable constraint needed for partial writes
pub trait WriteableStore: ReadableStore + ListableStore {
    type Writeable: Write;

    /// Write the contents of a key's entire value using the given function.
    fn set<F>(&self, key: &NodeKey, value: F) -> io::Result<()>
    where
        F: FnOnce(&mut Self::Writeable) -> io::Result<()>;

    /// Set partial regions with the given byte vecs.
    ///
    /// The trait's default implementation is inefficient in most cases
    /// and should be replaced by implementors.
    fn set_partial_values(
        &self,
        key_offset_values: Vec<(NodeKey, usize, Vec<u8>)>,
    ) -> Result<(), Error> {
        let mut bufs = HashMap::with_capacity(key_offset_values.len());

        for (key, range, vals) in key_offset_values.into_iter() {
            let length = range + vals.len();

            let buf = bufs.entry(key).or_insert_with_key(|k| {
                match self.get(k).expect("failed to read") {
                    Some(mut r) => {
                        let mut v = Vec::default();
                        r.read_to_end(&mut v).expect("failed to read");
                        if v.len() < length {
                            v.resize(length, 0)
                        }
                        v
                    }
                    None => vec![0; length],
                }
            });

            buf.splice(0..length, vals);
        }

        for (key, mut buf) in bufs {
            self.set(&key, |w| w.write_all(buf.as_mut()))?;
        }

        Ok(())
    }

    // TODO differs from spec in that it returns a bool indicating existence of the key at the end of the operation.
    /// Delete an object at a given key.
    fn erase(&self, key: &NodeKey) -> Result<bool, Error>;

    // TODO
    /// Delete all objects whose keys start with the given key.
    ///
    /// The trait's default implementation may be inefficient.
    fn erase_prefix(&self, key_prefix: &NodeKey) -> Result<bool, Error> {
        for key in self.list_prefix(key_prefix)? {
            self.erase(&key)?;
        }
        Ok(false)
    }
}
