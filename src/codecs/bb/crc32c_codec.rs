use crate::codecs::bb::BBCodec;
use byteorder::{ByteOrder, LittleEndian};
use crc32c::crc32c;
use serde::{Deserialize, Serialize};
use std::io::{Cursor, Read, Write};

#[derive(Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Debug)]
pub struct Crc32cCodec {}

impl BBCodec for Crc32cCodec {
    fn decoder<'a, R: Read + 'a>(&self, mut r: R) -> Box<dyn std::io::Read + 'a> {
        let mut buf = Vec::default();
        r.read_to_end(&mut buf)
            .expect("Reading CRC32c'd object failed");
        let suffix = buf.split_off(buf.len() - 4);
        let expected_hash = LittleEndian::read_u32(suffix.as_slice());
        let actual_hash = crc32c(buf.as_slice());

        // todo
        assert_eq!(expected_hash, actual_hash, "CRC32c checksum failed");

        Box::new(Cursor::new(buf))
    }

    fn encoder<'a, W: Write + 'a>(&self, w: W) -> Box<dyn std::io::Write + 'a> {
        todo!()
    }
}
