use std::fmt::Display;

use serde::{Deserialize, Serialize};
use smallvec::smallvec;

use crate::store::KEY_SEP;
use crate::{store::NodeKey, CoordVec};

#[enum_delegate::register]
pub trait ChunkKeyEncoder {
    fn components(&self, coord: &[u64]) -> CoordVec<String>;

    // probably unnecessary
    fn encode(&self, coord: &[u64]) -> String {
        let components = self.components(coord);
        components.join(KEY_SEP)
    }

    fn chunk_key(&self, node: &NodeKey, coord: &[u64]) -> NodeKey {
        let mut n = node.clone();
        for c in self.components(coord) {
            n.push_unchecked(&c);
        }
        n
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub enum Separator {
    #[serde(rename = "/")]
    Slash,
    #[serde(rename = ".")]
    Dot,
}

impl Display for Separator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Separator::Slash => write!(f, "/"),
            Separator::Dot => write!(f, "."),
        }
    }
}

fn slash() -> Separator {
    Separator::Slash
}

fn dot() -> Separator {
    Separator::Dot
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct DefaultChunkKeyEncoding {
    #[serde(default = "slash")]
    separator: Separator,
}

impl ChunkKeyEncoder for DefaultChunkKeyEncoding {
    fn components(&self, coord: &[u64]) -> CoordVec<String> {
        let mut out = CoordVec::default();
        match self.separator {
            Separator::Slash => {
                out.push("c".to_owned());
                for n in coord.iter() {
                    out.push(n.to_string());
                }
            }
            Separator::Dot => {
                let sep = self.separator.to_string();
                let s = coord
                    .iter()
                    .map(|n| n.to_string())
                    .fold(String::from("c"), |a, b| a + &sep + &b);
                out.push(s);
            }
        }
        out
    }
}

impl Default for DefaultChunkKeyEncoding {
    fn default() -> Self {
        Self {
            separator: Separator::Slash,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct V2ChunkKeyEncoding {
    #[serde(default = "dot")]
    separator: Separator,
}

impl ChunkKeyEncoder for V2ChunkKeyEncoding {
    fn components(&self, coord: &[u64]) -> CoordVec<String> {
        if coord.is_empty() {
            return smallvec!["0".to_owned()];
        }
        let mut out = CoordVec::default();
        match self.separator {
            Separator::Slash => {
                for n in coord.iter() {
                    out.push(n.to_string());
                }
            }
            Separator::Dot => {
                let sep = self.separator.to_string();
                let s = coord
                    .iter()
                    .map(|n| n.to_string())
                    .reduce(|a, b| a + &sep + &b)
                    .unwrap();
                out.push(s);
            }
        }
        out
    }
}

impl Default for V2ChunkKeyEncoding {
    fn default() -> Self {
        Self {
            separator: Separator::Dot,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(tag = "name", content = "configuration", rename_all = "lowercase")]
#[enum_delegate::implement(ChunkKeyEncoder)]
pub enum ChunkKeyEncoding {
    Default(DefaultChunkKeyEncoding),
    V2(V2ChunkKeyEncoding),
}

// todo: what to do when "configuration" is undefined?

impl Default for ChunkKeyEncoding {
    fn default() -> Self {
        Self::Default(DefaultChunkKeyEncoding::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json;

    #[test]
    fn roundtrip_chunk_key_encoding() {
        let to_deser = vec![
            r#"{"name":"default","configuration":{"separator":"/"}}"#,
            r#"{"name":"default","configuration":{"separator":"."}}"#,
            r#"{"name":"v2","configuration":{"separator":"/"}}"#,
            r#"{"name":"v2","configuration":{"separator":"."}}"#,
        ];

        for s in to_deser.into_iter() {
            let c: ChunkKeyEncoding =
                serde_json::from_str(s).expect(&format!("Could not deser {s}"));
            let s2 = serde_json::to_string(&c).expect(&format!("Could not ser {c:?}"));
            assert_eq!(s, &s2); // might depend on spaces
        }
    }

    #[test]
    fn serde_default_chunk_key_encoding() {
        let to_deser = vec![
            (
                r#"{"name":"default","configuration":{"separator":"/"}}"#,
                ChunkKeyEncoding::Default(DefaultChunkKeyEncoding {
                    separator: Separator::Slash,
                }),
            ),
            (
                r#"{"name":"default","configuration":{}}"#,
                ChunkKeyEncoding::Default(DefaultChunkKeyEncoding {
                    separator: Separator::Slash,
                }),
            ),
            // (r#"{"name":"default"}"#, ChunkKeyEncoding::Default(DefaultChunkKeyEncoding { separator: Separator::Slash })),
        ];

        for (s, expected) in to_deser.into_iter() {
            let c: ChunkKeyEncoding =
                serde_json::from_str(s).expect(&format!("Could not deser {s}"));
            assert_eq!(c, expected);
        }
    }

    #[test]
    fn default_chunk_key_encoding() {
        let cke = ChunkKeyEncoding::Default(DefaultChunkKeyEncoding::default());
        let s = cke.encode(&[1, 2, 3]);
        assert_eq!(&s, "c/1/2/3");
    }

    #[test]
    fn v2_chunk_key_encoding() {
        let cke = ChunkKeyEncoding::V2(V2ChunkKeyEncoding::default());
        let s = cke.encode(&[1, 2, 3]);
        assert_eq!(&s, "1.2.3");
    }
}
