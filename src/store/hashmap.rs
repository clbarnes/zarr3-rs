use std::{
    collections::{HashMap, HashSet},
    io::{self, Cursor, Read},
};

use super::{ListableStore, NodeKey, ReadableStore, Store, WriteableStore};

pub struct HashMapStore {
    map: HashMap<NodeKey, Vec<u8>>,
}

impl Store for HashMapStore {}

impl ReadableStore for HashMapStore {
    fn get(&self, key: &NodeKey) -> Result<Option<Box<dyn std::io::Read>>, std::io::Error> {
        // todo: avoid this clone
        let out = self
            .map
            .get(key)
            .map(|v| Box::new(Cursor::new(v.clone())) as Box<dyn Read>);
        Ok(out)
    }

    fn get_partial_values(
        &self,
        key_ranges: &[(NodeKey, crate::ByteRange)],
    ) -> Result<Vec<Option<Box<dyn Read>>>, std::io::Error> {
        let mut out = Vec::with_capacity(key_ranges.len());
        for (key, range) in key_ranges.iter() {
            let r = self.map.get(key).map(|v| {
                let copied = range.slice(v.as_slice()).to_vec();
                Box::new(Cursor::new(copied)) as Box<dyn Read>
            });
            out.push(r);
        }
        Ok(out)
    }
}

impl ListableStore for HashMapStore {
    fn list(&self) -> io::Result<Vec<NodeKey>> {
        Ok(self.map.keys().cloned().collect::<Vec<_>>())
    }

    fn list_prefix(&self, prefix: &NodeKey) -> io::Result<Vec<NodeKey>> {
        Ok(self
            .map
            .keys()
            .filter(|k| k.len() > prefix.len() && k.starts_with(prefix))
            .cloned()
            .collect())
    }

    fn list_dir(&self, prefix: &NodeKey) -> Result<(Vec<NodeKey>, Vec<NodeKey>), std::io::Error> {
        let mut keys = Vec::default();
        let mut prefixes: HashSet<NodeKey> = HashSet::default();

        for k in self.map.keys() {
            if k.len() == prefix.len() + 1 {
                if k.starts_with(prefix) {
                    keys.push(k.clone());
                }
            } else if k.len() > prefix.len() && k.starts_with(prefix) {
                let mut value = prefix.clone();
                value.push(prefix.as_slice()[value.len()].clone());
                prefixes.insert(value);
            }
        }

        Ok((keys, prefixes.into_iter().collect()))
    }
}

// todo: WriteableStore breaks because it needs &mut self, and because returning a writer is hard
