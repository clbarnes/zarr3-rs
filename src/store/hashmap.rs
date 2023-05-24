use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    io::{self, Cursor},
};

use super::{ListableStore, NodeKey, ReadableStore, Store, WriteableStore};

pub struct HashMapStore {
    // this locks whole map for read of single key
    // consider https://crates.io/crates/lockable
    map: RefCell<HashMap<NodeKey, Vec<u8>>>,
}

impl Store for HashMapStore {}

impl ReadableStore for HashMapStore {
    type Readable = Cursor<Vec<u8>>;

    fn get(&self, key: &NodeKey) -> Result<Option<Self::Readable>, std::io::Error> {
        let map = self.map.borrow();

        // todo: avoid this clone, maybe using bytes::Bytes?
        let out = map.get(key).map(|v| Cursor::new(v.clone()));
        Ok(out)
    }

    // fn get_partial_values(
    //     &self,
    //     key_ranges: &[(NodeKey, crate::ByteRange)],
    // ) -> Result<Vec<Option<Box<dyn Read>>>, std::io::Error> {
    //     let map = self.map.borrow();
    //     let mut out = Vec::with_capacity(key_ranges.len());
    //     for (key, range) in key_ranges.iter() {
    //         let r = map.get(key).map(|v| {
    //             let copied = range.slice(v.as_slice()).to_vec();
    //             Box::new(Cursor::new(copied)) as Box<dyn Read>
    //         });
    //         out.push(r);
    //     }
    //     Ok(out)
    // }

    fn has_key(&self, key: &NodeKey) -> io::Result<bool> {
        let map = self.map.borrow();
        Ok(map.contains_key(key))
    }
}

impl ListableStore for HashMapStore {
    fn list(&self) -> io::Result<Vec<NodeKey>> {
        let map = self.map.borrow();
        Ok(map.keys().cloned().collect::<Vec<_>>())
    }

    fn list_prefix(&self, prefix: &NodeKey) -> io::Result<Vec<NodeKey>> {
        let map = self.map.borrow();
        Ok(map
            .keys()
            .filter(|k| k.len() > prefix.len() && k.starts_with(prefix))
            .cloned()
            .collect())
    }

    fn list_dir(&self, prefix: &NodeKey) -> Result<(Vec<NodeKey>, Vec<NodeKey>), std::io::Error> {
        let map = self.map.borrow();
        let mut keys = Vec::default();
        let mut prefixes: HashSet<NodeKey> = HashSet::default();

        for k in map.keys() {
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

impl WriteableStore for HashMapStore {
    type Writeable = Cursor<Vec<u8>>;

    fn set<F>(&self, key: &NodeKey, value: F) -> io::Result<()>
    where
        F: FnOnce(&mut Self::Writeable) -> io::Result<()>,
    {
        let mut map = self.map.borrow_mut();
        let mut w = Cursor::new(Vec::default());
        value(&mut w)?;
        map.insert(key.clone(), w.into_inner());
        Ok(())
    }

    fn erase(&self, key: &NodeKey) -> Result<bool, io::Error> {
        let mut map = self.map.borrow_mut();
        map.remove(key);
        Ok(false)
    }

    fn erase_prefix(&self, key_prefix: &NodeKey) -> Result<bool, io::Error> {
        let mut map = self.map.borrow_mut();
        map.retain(|k, _v| !k.starts_with(key_prefix));
        Ok(false)
    }
}
