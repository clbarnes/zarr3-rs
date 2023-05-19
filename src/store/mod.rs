use log::warn;
use smallvec::SmallVec;
use std::{
    collections::HashMap,
    fmt::Display,
    io::{self, Cursor, Error, Read, Write},
    str::FromStr,
};

use crate::ByteRange;

mod hashmap;
pub use hashmap::HashMapStore;

#[cfg(feature = "filesystem")]
pub mod filesystem;

const NODE_KEY_SIZE: usize = 10;
const METADATA_NAME: &str = "zarr.json";
pub(crate) const KEY_SEP: &'static str = "/";

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

            if !has_non_recommended {
                if !c.is_ascii_alphanumeric() && c != '-' && c != '_' && c != '.' {
                    has_non_recommended = true;
                    warn!("Node name has non-recommended character `{}`; prefer `a-z`, `A-Z`, `0-9`, `-`, `_`, `.`", c);
                }
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
    /// Validates and adds a new key component in-place.
    ///
    /// If Ok, returns the new number of components.
    pub fn push(&mut self, name: NodeName) -> usize {
        self.0.push(name);
        self.0.len()
    }

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

    pub fn common_root(&self, other: &NodeKey) -> NodeKey {
        self.as_slice()
            .iter()
            .zip(other.as_slice().iter())
            .take_while(|(a, b)| a == b)
            .map(|(a, _)| a.clone())
            .collect()
    }

    pub fn starts_with(&self, other: &NodeKey) -> bool {
        self.len() >= other.len() && {
            self.as_slice()
                .iter()
                .zip(other.as_slice().iter())
                .all(|(a, b)| a == b)
        }
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
}

impl Default for NodeKey {
    fn default() -> Self {
        Self(Default::default())
    }
}

pub trait Store {}

pub trait ReadableStore: Store {
    // /// TODO: not in zarr spec
    // fn exists(&self, key: &NodeKey) -> Result<bool, Error>;

    fn get(&self, key: &NodeKey) -> Result<Option<Box<dyn Read>>, Error>;

    fn get_partial_values(
        &self,
        key_ranges: &[(NodeKey, ByteRange)],
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
            let rd = if let Some(b) = bufs.get(key).unwrap() {
                Some(Box::new(Cursor::new(
                    // todo: unnecessary clone?
                    range.slice(b.as_slice()).to_vec(),
                )) as Box<dyn Read>)
            } else {
                None
            };
            out.push(rd);
        }
        Ok(out)
    }

    // /// TODO: not in zarr spec
    // fn uri(&self, key: &NodeKey) -> Result<String, Error>;
}

pub trait ListableStore: Store {
    /// Retrieve all keys in the store.
    fn list(&self) -> Result<Vec<NodeKey>, Error> {
        self.list_prefix(&NodeKey::default())
    }

    /// Retrieve all keys with a given prefix.
    fn list_prefix(&self, key: &NodeKey) -> Result<Vec<NodeKey>, Error> {
        let mut to_visit = vec![key.clone()];
        let mut result = vec![];

        while let Some(next) = to_visit.pop() {
            let dir = self.list_dir(&next)?;
            result.extend(dir.0);
            to_visit.extend(dir.1);
        }

        Ok(result)
    }

    /// Retrieve all keys and prefixes with a given prefix and which do not
    /// contain the character “/” after the given prefix.
    fn list_dir(&self, prefix: &NodeKey) -> Result<(Vec<NodeKey>, Vec<NodeKey>), Error>;
}

// Readable constraint needed for partial writes
pub trait WriteableStore: ReadableStore {
    fn set(&self, key: &NodeKey) -> io::Result<Box<dyn Write>>;

    // fn set_partial_values<F: FnOnce(Box<dyn Write>) -> Result<(), Error>>(
    //     &self, key_range_values: Vec<(NodeKey, ByteRange, F)>
    // ) -> Result<(), Error> {
    //     let mut bufs = HashMap::with_capacity(key_range_values.len());
    //     for (key, range, func) in key_range_values.into_iter() {
    //         if !bufs.contains_key(&key) {
    //             match self.get(&key) {
    //                 Ok(Some(mut r)) => {
    //                     let mut buf = Vec::default();
    //                     r.read_to_end(&mut buf)?;
    //                     bufs.insert(key.clone(), buf);
    //                     Ok(())
    //                 }
    //                 Ok(None) => {
    //                     let mut buf = Vec::default();
    //                     bufs.insert(key.clone(), buf);
    //                     Ok(())
    //                 }
    //                 Err(e) => Err(e),
    //             }?;
    //         }

    //         let buf = bufs.get_mut(&key).unwrap();

    //         let s = match range.offset {
    //             Offset::Start(o) => {
    //                 if let Some(n) = range.nbytes {
    //                     o + n
    //                 } else {
    //                     o
    //                 }
    //             },
    //             Offset::End(o) => o
    //         };
    //         if buf.len() < s {
    //             buf.resize(s, 0);
    //         }
    //         let mut sliced = range.slice_mut(buf.as_mut_slice());

    //         func(Box::new(sliced))?;
    //     }
    //     Ok(())
    // }

    // TODO differs from spec in that it returns a bool indicating existence of the key at the end of the operation.
    fn erase(&self, key: &NodeKey) -> Result<bool, Error>;

    // TODO
    fn erase_prefix(&self, key_prefix: &NodeKey) -> Result<bool, Error>;
}
