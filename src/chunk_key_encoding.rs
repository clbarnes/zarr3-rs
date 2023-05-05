use std::fmt::Display;

use serde::{Deserialize, Serialize};

use crate::ChunkCoord;

pub trait ChunkKeyEncoder {
    fn encode(&self, coord: ChunkCoord) -> String;
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
    fn encode(&self, coord: ChunkCoord) -> String {
        let s = String::from("c/");
        coord
            .iter()
            .map(|n| n.to_string())
            .fold(s, |a, b| a + &b + ",")
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
    fn encode(&self, coord: ChunkCoord) -> String {
        coord
            .iter()
            .map(|n| n.to_string())
            .fold(String::new(), |a, b| a + &b + ",")
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
    fn default_chunk_key_encoding() {
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
}
