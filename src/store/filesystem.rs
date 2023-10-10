use std::{
    fs::{self, File},
    io::{self, ErrorKind, Read, Seek, SeekFrom},
    path::PathBuf,
};

use fs4::FileExt;
use log::warn;
use walkdir::WalkDir;

use super::{
    list_from_list_prefix, list_prefix_from_list_dir, ListableStore, NodeKey, NodeName,
    ReadableStore, Store, WriteableStore,
};
use crate::RangeRequest;

pub struct FileSystemStore {
    base_path: PathBuf,
}

impl FileSystemStore {
    /// Does not check or modify path.
    pub fn new_unchecked(path: PathBuf) -> Self {
        Self { base_path: path }
    }

    /// Canonicalizes path and checks that it is an extant directory.
    pub fn open(path: PathBuf) -> io::Result<Self> {
        let base_path = path.canonicalize()?;
        let meta = fs::metadata(&base_path)?;
        if meta.is_file() {
            Err(io::Error::new(
                ErrorKind::Other,
                "Path exists, but it is a file",
            ))
        } else {
            Ok(Self { base_path })
        }
    }

    /// Canonicalizes path and checks that it is an extant directory.
    pub fn create(path: PathBuf, parents: bool) -> io::Result<Self> {
        if path.exists() {
            return Err(io::Error::new(ErrorKind::AlreadyExists, "Already exists"));
        } else if parents {
            fs::create_dir_all(&path)?;
        } else {
            fs::create_dir(&path)?;
        }
        Ok(Self {
            base_path: path.canonicalize()?,
        })
    }

    /// Canonicalizes path and, if the directory does not exist, creates it.
    pub fn open_or_create(path: PathBuf, parents: bool) -> io::Result<Self> {
        let base_path = path.canonicalize()?;
        if base_path.exists() {
            let meta = fs::metadata(&base_path)?;
            if meta.is_file() {
                return Err(io::Error::new(
                    ErrorKind::Other,
                    "Path exists, but it is a file",
                ));
            }
            return Ok(Self { base_path });
        } else if parents {
            fs::create_dir_all(path)?;
        } else {
            fs::create_dir(path)?;
        }
        Ok(Self { base_path })
    }

    fn get_path(&self, key: &NodeKey) -> PathBuf {
        let mut p = self.base_path.clone();
        for k in key.as_slice().iter() {
            p.push(k.as_ref());
        }
        p
    }

    fn file_reader(&self, key: &NodeKey) -> io::Result<Option<File>> {
        let target = self.get_path(key);
        match File::open(target) {
            Ok(f) => {
                f.lock_shared()?;
                Ok(Some(f))
            }
            Err(e) if e.kind() == ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e),
        }
    }
}

impl ReadableStore for FileSystemStore {
    // todo: buf?
    type Readable = File;

    fn get(&self, key: &NodeKey) -> Result<Option<Self::Readable>, io::Error> {
        self.file_reader(key)
    }

    fn get_partial_values(
        &self,
        key_ranges: &[(NodeKey, RangeRequest)],
    ) -> Result<Vec<Option<Box<dyn Read>>>, std::io::Error> {
        let mut out = Vec::with_capacity(key_ranges.len());

        for (key, range) in key_ranges.iter() {
            let to_push = if let Some(f) = self.file_reader(key)? {
                let r = SubReader::new(f, *range)?;
                Some(Box::new(r) as Box<dyn Read>)
            } else {
                None
            };
            out.push(to_push);
        }

        Ok(out)
    }
}

impl ListableStore for FileSystemStore {
    fn list(&self) -> Result<Vec<NodeKey>, io::Error> {
        list_from_list_prefix(self)
    }

    fn list_prefix(&self, key: &NodeKey) -> Result<Vec<NodeKey>, io::Error> {
        // todo: more efficient to use walkdir?
        list_prefix_from_list_dir(self, key)
    }

    fn list_dir(&self, prefix: &NodeKey) -> Result<(Vec<NodeKey>, Vec<NodeKey>), io::Error> {
        // This may be inconsistent with other implementations if a directory tree has no files in it.
        // Directories are not prefixes unless there is a file somewhere beneath them.
        let mut keys = Vec::default();
        let mut prefixes = Vec::default();

        let target = self.get_path(prefix);
        for maybe_file in fs::read_dir(target)? {
            let file = maybe_file?;
            let mut key = prefix.clone();
            let fname = file.file_name();
            let name = if let Some(n) = fname.to_str() {
                n
            } else {
                warn!("Skipping node with non-UTF8 name: {:?}", fname);
                continue;
            };
            match name.parse::<NodeName>() {
                Ok(n) => {
                    key.push(n);
                }
                Err(_) => continue,
            };

            let meta = fs::metadata(file.path())?;

            if meta.is_file() {
                keys.push(key);
            } else {
                prefixes.push(key)
            }
        }

        Ok((keys, prefixes))
    }
}

impl Store for FileSystemStore {}

impl WriteableStore for FileSystemStore {
    type Writeable = File;

    fn set<F>(&self, key: &NodeKey, value: F) -> io::Result<()>
    where
        F: FnOnce(&mut Self::Writeable) -> io::Result<()>,
    {
        let path = self.get_path(key);
        if !key.is_root() {
            let parent = path.parent().expect("Key is filesystem root");
            fs::create_dir_all(parent)?;
        }

        let mut f = fs::OpenOptions::new()
            .write(true)
            .truncate(true)
            .create(true)
            .open(path)?;
        f.lock_exclusive()?;
        value(&mut f)
    }

    fn erase(&self, key: &NodeKey) -> io::Result<bool> {
        let path = self.get_path(key);

        match File::open(path.clone()) {
            Ok(f) => {
                f.lock_exclusive()?;
                fs::remove_file(&path)?;
                Ok(false)
            }
            Err(e) if e.kind() == ErrorKind::NotFound => Ok(false),
            Err(e) => Err(e),
        }
        // is this sufficient to guarantee that return is correct?
        // todo: what if it's a directory?
    }

    fn erase_prefix(&self, key_prefix: &NodeKey) -> io::Result<bool> {
        let path = self.get_path(key_prefix);

        if path.exists() {
            for entry in WalkDir::new(&path).contents_first(true).follow_links(true) {
                // todo: follow_links(true) allows for recursion, not good
                let entry = entry?;

                if entry.file_type().is_dir() {
                    fs::remove_dir(entry.path())?;
                } else {
                    let file = File::open(entry.path())?;
                    file.lock_exclusive()?;
                    fs::remove_file(entry.path())?;
                }
            }
        }

        Ok(!path.exists())
    }
}

struct SubReader<R: Read + Seek> {
    offset: u64,
    nbytes: u64,
    reader: R,
}

// Replace if/ when
// https://doc.rust-lang.org/stable/std/io/trait.Seek.html#method.stream_len
// stabilises
fn stream_len<S: Seek>(s: &mut S) -> Result<u64, io::Error> {
    let orig = SeekFrom::Start(s.stream_position()?);
    let len = s.seek(SeekFrom::End(0))?;
    s.seek(orig)?;
    Ok(len)
}

impl<R: Read + Seek> SubReader<R> {
    pub fn new(mut reader: R, range: RangeRequest) -> std::io::Result<Self> {
        let orig_len = Some(stream_len(&mut reader)? as usize);

        let start = range.start(orig_len).unwrap() as u64;
        let end = range.end(orig_len).unwrap() as u64;
        // todo: handle before-start case

        Ok(Self {
            offset: start,
            nbytes: end - start,
            reader,
        })
    }

    pub fn end_offset(&self) -> u64 {
        self.offset + self.nbytes
    }
}

impl<R: Read + Seek> Read for SubReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let pos = self.stream_position()?;
        let max_len = (self.nbytes - pos) as usize;
        if buf.len() > max_len {
            self.reader.read(&mut buf[..max_len])
        } else {
            self.reader.read(buf)
        }
    }
}

impl<R: Read + Seek> Seek for SubReader<R> {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        let new_pos = match pos {
            SeekFrom::Start(o) => self.reader.seek(SeekFrom::Start(self.offset + o))?,
            SeekFrom::End(o) => self
                .reader
                .seek(SeekFrom::Start((self.end_offset() as i64 + o) as u64))?,
            SeekFrom::Current(o) => {
                let orig_pos = self.stream_position()?;
                let new_pos = self.reader.seek(SeekFrom::Current(o))?;
                if new_pos < self.offset {
                    let out = Err(std::io::Error::new(
                        ErrorKind::InvalidInput,
                        "Seeked before start of SubReader",
                    ));
                    self.reader.seek(SeekFrom::Start(orig_pos))?;
                    out?
                }
                new_pos
            }
        };
        Ok(new_pos)
    }
}
