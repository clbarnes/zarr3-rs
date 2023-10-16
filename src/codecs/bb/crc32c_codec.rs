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
fn validate_crc32c<'a, R: Read>(mut r: R) -> io::Result<Vec<u8>> {
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
    // big-endian because the pops above guarantee reverse order
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

/// [Read]er wrapper which, on first call to `.read()`, reads the entire wrapped [Read]er,
/// interprets the last 4 bytes as a u32le CRC32C checksum, and checks that it matches the rest of the content.
///
/// The payload is cached so that subsequent calls to `.read()` seem to progress through the wrapped reader normally.
/// In practice, these subsequent reads are infallible.
struct Crc32cReader<R: Read> {
    r: R,
    content: Option<Cursor<Vec<u8>>>,
}

impl<R: Read> Crc32cReader<R> {
    pub fn new(r: R) -> Self {
        Self { r, content: None }
    }
}

impl<R: Read> Read for Crc32cReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.content.is_none() {
            // no reads have happened yet
            let content = validate_crc32c(&mut self.r)?;
            self.content = Some(Cursor::new(content));
        }

        self.content.as_mut().unwrap().read(buf)
    }
}

struct Crc32cWriter<W: Write> {
    w: W,
    checksum: u32,
}

impl<W: Write> Crc32cWriter<W> {
    pub fn new(w: W) -> Self {
        Self { w, checksum: 0 }
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
        self.w
            .write_u32::<LittleEndian>(self.checksum)
            .map(|_| std::mem::size_of::<u32>())
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Debug, Default)]
/// Codec which appends a little-endian CRC32C checksum to an encoded payload,
/// and allows checking that hash when decoding.
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

        // change the checksum value
        let last = buf.pop().unwrap();
        buf.push(last.wrapping_add(1));
        assert!(validate_crc32c(buf.as_slice()).is_err())
    }

    #[test]
    fn can_read() {
        let mut buf = Vec::default();
        buf.extend_from_slice(TEST_STRING);
        buf.extend_from_slice(&CHECKSUM);

        let mut r = Crc32cReader::new(&buf[..]);
        let mut out = Vec::default();
        r.read_to_end(&mut out).unwrap();
        assert_eq!(out, TEST_STRING);
    }

    #[test]
    fn can_fail_read() {
        let mut buf = Vec::default();
        buf.extend_from_slice(TEST_STRING);
        buf.extend_from_slice(&CHECKSUM);
        // change the checksum value
        let last = buf.pop().unwrap();
        buf.push(last.wrapping_add(1));

        let mut r = Crc32cReader::new(&buf[..]);
        let mut out = Vec::default();
        let res = r.read_to_end(&mut out);
        assert!(res.is_err())
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
