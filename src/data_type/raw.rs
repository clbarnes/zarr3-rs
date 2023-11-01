use super::{DataType, Endian, ReflectedType};

macro_rules! reflected_raw {
    ($($nbytes:expr), *) => {
        $(
        impl ReflectedType for [u8; $nbytes] {
            const ZARR_TYPE: DataType = DataType::Raw($nbytes * 8);

            /// Endianness is ignored for raw types.
            fn encoder(_endian: Endian) -> Box<dyn Fn(Self, &mut [u8])> {
                Box::new(|v: Self, buf: &mut[u8]| {
                    buf.copy_from_slice(&v);
                })
            }

            /// Produce a routine which reads a self-typed value from
            /// the given byte buffer.
            ///
            /// Endianness is ignored for raw types.
            fn decoder(_endian: Endian) -> Box<dyn Fn(&mut [u8]) -> Self> {
                Box::new(|buf: &mut[u8]| {
                    let mut out = [0; $nbytes];
                    out.as_mut().copy_from_slice(buf);
                    out
                })
            }
        }
    )*
    }
}

reflected_raw!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
