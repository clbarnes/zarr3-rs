use crate::{
    codecs::{aa::Order, ab::endian::Endian},
    data_types::{DataType, NBytes},
    CoordVec,
};
use ndarray::ArrayD;

#[derive(Clone, Debug)]
struct NDArr {
    indices: ArrayD<usize>,
    buffer: Vec<u8>,
    endian: Endian,
    data_type: DataType,
}

impl NDArr {
    pub fn new(
        shape: CoordVec<usize>,
        data_type: DataType,
        endian: Endian,
        buffer: Vec<u8>,
        order: Order,
    ) -> Result<Self, &'static str> {
        match order {
            Order::Permutation(_) => {
                return Err("Only C and F orders are permitted when constructing an array")
            }
            _ => (),
        }
        data_type.nbytes();
        todo!()
    }

    pub fn set_endian(mut self, endian: Endian) -> Self {
        if endian == self.endian {
            return self;
        }
        self.endian = endian;
        self
    }
}
