use thiserror::Error;

/// For an enum where some variants contain data and some do not,
/// serializes a given no-data variant as a string of its name.
///
/// From <https://github.com/serde-rs/serde/issues/1560#issuecomment-506915291>
#[macro_export]
macro_rules! named_unit_variant {
    ($variant:ident) => {
        pub mod $variant {
            pub fn serialize<S>(serializer: S) -> Result<S::Ok, S::Error>
            where
                S: serde::Serializer,
            {
                serializer.serialize_str(stringify!($variant))
            }

            pub fn deserialize<'de, D>(deserializer: D) -> Result<(), D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct V;
                impl<'de> serde::de::Visitor<'de> for V {
                    type Value = ();
                    fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                        f.write_str(concat!("\"", stringify!($variant), "\""))
                    }
                    fn visit_str<E: serde::de::Error>(self, value: &str) -> Result<Self::Value, E> {
                        if value == stringify!($variant) {
                            Ok(())
                        } else {
                            Err(E::invalid_value(serde::de::Unexpected::Str(value), &self))
                        }
                    }
                }
                deserializer.deserialize_str(V)
            }
        }
    };
}

/// adds `From<D>` for an enum with a variant containing D
///
/// N.B. this is also handled by enum_delegate::implement
#[macro_export]
macro_rules! variant_from_data {
    ($enum:ty, $variant:ident, $data_type:ty) => {
        impl std::convert::From<$data_type> for $enum {
            fn from(c: $data_type) -> Self {
                <$enum>::$variant(c)
            }
        }
    };
}

#[derive(Error, Debug)]
#[error("Got {other_ndim} dimensions when expecting {ref_ndim}")]
pub struct DimensionMismatch {
    ref_ndim: usize,
    other_ndim: usize,
}

impl DimensionMismatch {
    pub fn check_coords(coord_ndim: usize, array_ndim: usize) -> Result<(), Self> {
        if coord_ndim == array_ndim {
            Ok(())
        } else {
            Err(Self {
                ref_ndim: coord_ndim,
                other_ndim: array_ndim,
            })
        }
    }

    pub fn check_many(reference: usize, others: &[usize]) -> Result<(), Self> {
        for o in others.iter() {
            if o != &reference {
                return Err(Self {
                    ref_ndim: reference,
                    other_ndim: *o,
                });
            }
        }
        Ok(())
    }
}

/// Panic if any dimensions mismatch.
pub fn dimpanic(reference: usize, others: &[usize]) {
    DimensionMismatch::check_many(reference, others).unwrap()
}
