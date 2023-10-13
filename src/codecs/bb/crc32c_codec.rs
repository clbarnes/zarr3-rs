use super::BBCodec;
use crate::codecs::fwrite::FinalWrite;
use byteorder::{LittleEndian, WriteBytesExt};
use crc32c::{crc32c, crc32c_append};
use serde::{Deserialize, Serialize};
use std::io::{self, Cursor};
use std::io::{Read, Write};

/// Read the entire stream as if it were a payload with a u32le crc32c checksum suffix.
///
/// Return the payload, or an error if the checksum does not match.
pub fn validate_crc32c<'a, R: Read>(mut r: R) -> io::Result<Vec<u8>> {
    let mut buf = Vec::default();
    let end = r.read_to_end(&mut buf)?;
    if end < 4 {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "Expected CRC32C payload was too short",
        ));
    }
    let suff_be = [
        buf.pop().unwrap(),
        buf.pop().unwrap(),
        buf.pop().unwrap(),
        buf.pop().unwrap(),
    ];
    let expected = u32::from_be_bytes(suff_be);
    let actual = crc32c(&buf);
    if expected == actual {
        Ok(buf)
    } else {
        Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Failed CRC32C checksum",
        ))
    }
}

pub struct Crc32cReader<R: Read> {
    r: R,
    buf: Option<Cursor<Vec<u8>>>,
}

impl<R: Read> Crc32cReader<R> {
    pub fn new(r: R) -> Self {
        Self { r, buf: None }
    }
}

impl<R: Read> Read for Crc32cReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.buf.is_none() {
            let buf = validate_crc32c(&mut self.r)?;
            self.buf = Some(Cursor::new(buf));
        }

        self.buf.as_mut().unwrap().read(buf)
    }
}

pub struct Crc32cWriter<W: Write> {
    w: W,
    checksum: u32,
}

impl<W: Write> Crc32cWriter<W> {
    pub fn new(w: W) -> Self {
        Self { w, checksum: 0 }
    }

    pub fn into_inner(self) -> (W, u32) {
        (self.w, self.checksum)
    }
}

impl<W: Write> Write for Crc32cWriter<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let n = self.w.write(buf)?;
        self.checksum = crc32c_append(self.checksum, &buf[..n]);
        Ok(n)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.w.flush()
    }
}

impl<W: Write> FinalWrite for Crc32cWriter<W> {
    fn finalize(&mut self) -> io::Result<usize> {
        self.w.write_u32::<LittleEndian>(self.checksum).map(|_| 4)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Debug, Default)]
pub struct Crc32cCodec {}

impl BBCodec for Crc32cCodec {
    fn decoder<'a, R: Read + 'a>(&self, r: R) -> Box<dyn Read + 'a> {
        Box::new(Crc32cReader::new(r))
    }

    fn encoder<'a, W: Write + 'a>(&self, w: W) -> Box<dyn FinalWrite + 'a> {
        Box::new(Crc32cWriter::new(w))
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    const TEST_STRING: &[u8] =
        b"This is a very long string which is used to test the CRC-32-Castagnoli function.";
    const CHECKSUM: [u8; 4] = 0x20_CB_1E_59_u32.to_le_bytes();

    #[test]
    fn can_validate() {
        let mut buf = Vec::default();
        buf.extend_from_slice(TEST_STRING);
        buf.extend_from_slice(&CHECKSUM);

        let out = validate_crc32c(buf.as_slice()).unwrap();

        assert_eq!(out, TEST_STRING);

        let last = buf.pop().unwrap();
        buf.push(last.wrapping_add(1));
        assert!(validate_crc32c(buf.as_slice()).is_err())
    }

    #[test]
    fn can_write() {
        let mut buf = Vec::<u8>::default();

        let mut writer = Crc32cWriter::new(Cursor::new(&mut buf));
        writer.write_all(TEST_STRING).unwrap();
        writer.finalize().unwrap();

        assert_eq!(buf.len(), TEST_STRING.len() + 4);
        assert_eq!(&buf[..TEST_STRING.len()], TEST_STRING);
        assert_eq!(&buf[buf.len() - 4..], CHECKSUM)
    }
}
