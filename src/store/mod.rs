use log::warn;
use smallvec::SmallVec;
use std::{
    collections::HashMap,
    io::{Cursor, Error, Read, Write},
};

use crate::{ByteRange, Offset};

const NODE_KEY_SIZE: usize = 10;
const METADATA_KEY: &'static str = "zarr.json";
pub(crate) const KEY_SEP: &'static str = "/";

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NodeKey(SmallVec<[String; NODE_KEY_SIZE]>);

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

impl InvalidNodeName {
    pub fn validate_name(name: &str) -> Result<&str, Self> {
        let mut is_periods = true;
        let mut is_underscore = true;
        let mut has_non_recommended = false;
        let mut len: usize = 0;
        for c in name.chars() {
            if is_periods && c != '.' {
                is_periods = false;
            }
            if is_underscore {
                if len >= 2 {
                    return Err(Self::ReservedPrefix);
                }
                if c != '_' {
                    is_underscore = false;
                }
            }
            if c == '/' {
                return Err(Self::HasSlash);
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
            return Err(Self::Empty);
        }
        if is_periods {
            return Err(Self::IsPeriods);
        }
        Ok(name)
    }
}

impl NodeKey {
    /// Validates and adds a new key component in-place.
    ///
    /// If Ok, returns the new number of components.
    pub fn push(&mut self, s: &str) -> Result<usize, InvalidNodeName> {
        InvalidNodeName::validate_name(s)?;
        self.0.push(s.to_owned());
        Ok(self.0.len())
    }

    pub(crate) fn push_unchecked(&mut self, s: &str) -> usize {
        self.0.push(s.to_owned());
        self.0.len()
    }

    /// Pop the last key component.
    ///
    /// None if we are at the root.
    pub fn pop(&mut self) -> Option<String> {
        self.0.pop()
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
                    new.push(other)?;
                }
            };
        }
        Ok(Some(new))
    }

    pub(crate) fn key(&self) -> String {
        self.0.join(KEY_SEP)
    }

    pub(crate) fn prefix(&self) -> String {
        self.key() + KEY_SEP
    }

    pub(crate) fn metadata_key(&self) -> String {
        self.prefix() + METADATA_KEY
    }
}

impl Default for NodeKey {
    fn default() -> Self {
        Self(Default::default())
    }
}

impl TryFrom<&str> for NodeKey {
    type Error = InvalidNodeName;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        value
            .split(KEY_SEP)
            .map(|n| InvalidNodeName::validate_name(n).map(|s| s.to_owned()))
            .collect::<Result<SmallVec<_>, Self::Error>>()
            .map(|c| Self(c))
    }
}

impl TryFrom<&[&str]> for NodeKey {
    type Error = InvalidNodeName;

    fn try_from(value: &[&str]) -> Result<Self, Self::Error> {
        value
            .iter()
            .map(|n| InvalidNodeName::validate_name(n).map(|s| s.to_owned()))
            .collect::<Result<SmallVec<_>, Self::Error>>()
            .map(|c| Self(c))
    }
}

pub trait ReadableStore {
    /// TODO: not in zarr spec
    fn exists(&self, key: &NodeKey) -> Result<bool, Error>;

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

pub trait ListableStore {
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
    fn set(&self, key: &NodeKey) -> Box<dyn Write>;

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
