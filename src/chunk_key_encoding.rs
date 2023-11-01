use std::fmt::Display;

use serde::{Deserialize, Serialize};
use smallvec::smallvec;

use crate::store::NodeName;
use crate::{store::NodeKey, CoordVec};

#[enum_delegate::register]
pub trait ChunkKeyEncoder {
    /// For a given coordinate slice, return the node name.
    ///
    /// If the encoder would insert path separators (e.g. `/`),
    /// the output will have multiple items in it.
    /// If the encoder would not (e.g. joining the coordinates with `.`),
    /// the output will have a single item.
    fn components(&self, coord: &[u64]) -> CoordVec<NodeName>;

    /// Get the key for a chunk below the given (array) node with the given coordinates.
    fn chunk_key(&self, node: &NodeKey, coord: &[u64]) -> NodeKey {
        let mut n = node.clone();
        for c in self.components(coord) {
            n.push(c);
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
    fn components(&self, coord: &[u64]) -> CoordVec<NodeName> {
        let mut out = CoordVec::default();
        match self.separator {
            Separator::Slash => {
                out.push("c".parse().unwrap());
                for n in coord.iter() {
                    out.push(NodeName::new_unchecked(n.to_string()));
                }
            }
            Separator::Dot => {
                let sep = self.separator.to_string();
                let s = coord
                    .iter()
                    .map(|n| n.to_string())
                    .fold(String::from("c"), |a, b| a + &sep + &b);
                out.push(NodeName::new_unchecked(s));
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
    fn components(&self, coord: &[u64]) -> CoordVec<NodeName> {
        if coord.is_empty() {
            return smallvec!["0".parse().unwrap()];
        }
        let mut out = CoordVec::default();
        match self.separator {
            Separator::Slash => {
                for n in coord.iter() {
                    out.push(NodeName::new_unchecked(n.to_string()));
                }
            }
            Separator::Dot => {
                let sep = self.separator.to_string();
                let s = coord
                    .iter()
                    .map(|n| n.to_string())
                    .reduce(|a, b| a + &sep + &b)
                    .unwrap();
                out.push(NodeName::new_unchecked(s));
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
                serde_json::from_str(s).unwrap_or_else(|_| panic!("Could not deser {s}"));
            let s2 = serde_json::to_string(&c).unwrap_or_else(|_| panic!("Could not ser {c:?}"));
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
                serde_json::from_str(s).unwrap_or_else(|_| panic!("Could not deser {s}"));
            assert_eq!(c, expected);
        }
    }

    #[test]
    fn default_chunk_key_encoding() {
        let cke = ChunkKeyEncoding::Default(DefaultChunkKeyEncoding::default());
        let s = cke.components(&[1, 2, 3]);
        let strs: Vec<_> = s.iter().map(|n| n.as_ref()).collect();
        let expected = vec!["c", "1", "2", "3"];
        assert_eq!(strs, expected);
    }

    #[test]
    fn v2_chunk_key_encoding() {
        let cke = ChunkKeyEncoding::V2(V2ChunkKeyEncoding::default());
        let s = cke.components(&[1, 2, 3]);
        let strs: Vec<_> = s.iter().map(|n| n.as_ref()).collect();
        let expected = vec!["1.2.3"];
        assert_eq!(strs, expected);
    }
}
