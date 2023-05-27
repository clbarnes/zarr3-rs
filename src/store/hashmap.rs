use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    io::{self, Read},
};

use bytes::{
    buf::{Reader, Writer},
    Buf, BufMut, Bytes, BytesMut,
};

use super::{ListableStore, NodeKey, ReadableStore, Store, WriteableStore};

pub struct HashMapStore {
    // this locks whole map for read of single key
    // consider https://crates.io/crates/lockable
    map: RefCell<HashMap<NodeKey, Bytes>>,
}

impl Store for HashMapStore {}

impl ReadableStore for HashMapStore {
    type Readable = Reader<Bytes>;

    fn get(&self, key: &NodeKey) -> Result<Option<Self::Readable>, std::io::Error> {
        let map = self.map.borrow();

        // not sure if this clone is expensive
        Ok(map.get(key).map(|b| b.clone().reader()))
    }

    fn get_partial_values(
        &self,
        key_ranges: &[(NodeKey, crate::RangeRequest)],
    ) -> Result<Vec<Option<Box<dyn Read>>>, std::io::Error> {
        let map = self.map.borrow();
        let mut out = Vec::with_capacity(key_ranges.len());
        for (key, range) in key_ranges.iter() {
            let r = map.get(key).map(|v| {
                let b = v.slice(range.to_range(v.len()));
                Box::new(b.reader()) as Box<dyn Read>
            });
            out.push(r);
        }
        Ok(out)
    }

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
            .filter(|k| k.is_descendant_of(prefix))
            .cloned()
            .collect())
    }

    fn list_dir(&self, prefix: &NodeKey) -> Result<(Vec<NodeKey>, Vec<NodeKey>), std::io::Error> {
        let map = self.map.borrow();
        let mut keys = Vec::default();
        let mut prefixes: HashSet<NodeKey> = HashSet::default();

        for k in map.keys().filter(|k| k.is_descendant_of(prefix)).cloned() {
            if k.len() == prefix.len() + 1 {
                keys.push(k);
            } else {
                let mut value = prefix.clone();
                value.push(k.as_slice()[value.len()].clone());
                prefixes.insert(value);
            }
        }

        Ok((keys, prefixes.into_iter().collect()))
    }
}

impl WriteableStore for HashMapStore {
    type Writeable = Writer<BytesMut>;

    fn set<F>(&self, key: &NodeKey, value: F) -> io::Result<()>
    where
        F: FnOnce(&mut Self::Writeable) -> io::Result<()>,
    {
        let mut map = self.map.borrow_mut();

        let mut w = BytesMut::new().writer();
        value(&mut w)?;
        map.insert(key.clone(), w.into_inner().into());
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
