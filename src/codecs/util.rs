pub struct Interval {
    pub start: isize,
    pub end: Option<isize>,
}

#[derive(Error, Debug)]
pub enum OutOfBounds {
    #[error("Index is -{0}")]
    BeforeStart(usize),
    #[error("Index is {idx} with max length of {max_len}")]
    AfterEnd { idx: usize, max_len: usize },
}

impl OutOfBounds {
    pub fn clamp(&self) -> usize {
        match self {
            Self::BeforeStart(_) => 0,
            Self::AfterEnd { idx, max_len } => *max_len,
        }
    }
}

fn pos_idx(idx: isize, len: usize) -> Result<usize, OutOfBounds> {
    if idx >= 0 {
        let pos_offset = idx as usize;
        if pos_offset <= len {
            return Ok(pos_offset);
        } else {
            return Err(OutOfBounds::AfterEnd {
                idx: pos_offset,
                max_len: len,
            });
        }
    }

    let neg_offset = idx.abs() as usize;
    if neg_offset > len {
        return Err(OutOfBounds::BeforeStart(neg_offset - len));
    }
    Ok(len - neg_offset)
}

fn int_to_seekfrom(i: isize) -> SeekFrom {
    if i < 0 {
        SeekFrom::End(i as i64)
    } else {
        SeekFrom::Start(i as u64)
    }
}

impl Interval {
    pub fn as_seekfrom_nbytes(&self, len: Option<usize>) -> (SeekFrom, Option<usize>) {
        let end = len.map(|l| {
            self.end
                .map(|e| pos_idx(e, l).unwrap_or_else(|e| e.clamp()))
                .unwrap_or(l)
        });
        (int_to_seekfrom(self.start), end)
    }

    pub fn as_offsets(&self, len: usize) -> (usize, usize) {
        let start = pos_idx(self.start, len).unwrap_or_else(|e| e.clamp());
        let end = self
            .end
            .map(|e| pos_idx(e, len).unwrap_or_else(|e| e.clamp()))
            .unwrap_or(len);
        (start, end)
    }
}

pub trait ByteReader {
    fn read(&self) -> Vec<u8>;

    fn read_partial<'a>(&self, ranges: &[Interval]) -> Vec<Vec<u8>> {
        let all = self.read();

        ranges
            .iter()
            .map(|r| {
                let (start, stop) = r.as_offsets(all.len());
                all[start..stop].iter().cloned().collect()
            })
            .collect()
    }
}

// todo: possible to do this without Seek?
// Only with non-negative offsets in interval
impl<R: Read + Seek> ByteReader for R {
    fn read(&mut self) -> Result<Vec<u8>> {
        self.seek(SeekFrom::Start(0))?;
        let mut b = Vec::default();
        self.read_to_end(&mut b)?;
        Ok(b)
    }

    fn partial_read(&mut self, interval: Interval) -> Result<Vec<u8>> {
        let (from, nbytes) =
            interval.as_seekfrom_nbytes(Some(self.seek(SeekFrom::End(0))? as usize));
        self.seek(from)?;
        let mut buf;
        if let Some(n) = nbytes {
            buf = vec![0; n];
            self.read(&mut buf)?;
        } else {
            buf = Vec::default();
            self.read_to_end(&mut buf)?;
        }
        Ok(buf)
    }
}

pub trait ByteReader {
    fn read(&mut self) -> Result<Vec<u8>>;

    // todo: multiple intervals
    fn partial_read(&mut self, interval: Interval) -> Result<Vec<u8>> {
        let whole = ByteReader::read(self)?;
        let start = if interval.start < 0 {
            whole.len() + interval.start as usize
        } else {
            interval.start as usize
        };
        let stop = if let Some(e) = interval.end {
            if e < 0 {
                whole.len() + e as usize
            } else {
                interval.start as usize
            }
        } else {
            whole.len()
        };
        if stop < start {
            Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "end is before start",
            ))
        } else {
            Ok((&whole[start..stop]).iter().cloned().collect())
        }
    }
}

struct SubReader<'r, R: Read + Seek> {
    offset: u64,
    nbytes: u64,
    reader: &'r mut R,
}

pub enum SeekEnd {
    SeekFrom(SeekFrom),
    NBytes(u64),
}

impl<'r, R: Read + Seek> SubReader<'r, R> {
    pub fn new(reader: &'r mut R, start: SeekFrom, end: SeekEnd) -> std::io::Result<Self> {
        let orig_pos = reader.stream_position()?;

        // this could be done in fewer seeks for particular start/end combinations
        let nbytes = match end {
            SeekEnd::NBytes(n) => n,
            SeekEnd::SeekFrom(sf) => {
                let pos = reader.seek(sf)?;
                reader.seek(SeekFrom::Start(orig_pos))?;
                pos
            }
        };

        let offset = reader.seek(start)?;

        Ok(Self {
            offset,
            nbytes,
            reader,
        })
    }

    pub fn end_offset(&self) -> u64 {
        self.offset + self.nbytes
    }
}

impl<'r, R: Read + Seek> Read for SubReader<'r, R> {
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

impl<'r, R: Read + Seek> Seek for SubReader<'r, R> {
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
