use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::{
    chunk_key_encoding::ChunkKeyEncoding,
    codecs::{aa::AACodecType, ab::ABCodecType, bb::BBCodecType, CodecChain},
    data_type::{DataType, ReflectedType},
    CoordVec, GridCoord, MaybeNdim, Ndim, ZARR_FORMAT,
};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RegularChunkGrid {
    chunk_shape: GridCoord,
}

impl RegularChunkGrid {
    fn new<T: Into<GridCoord>>(chunk_shape: T) -> Self {
        let chunk_shape = chunk_shape.into();
        Self { chunk_shape }
    }
}

impl Ndim for RegularChunkGrid {
    fn ndim(&self) -> usize {
        self.chunk_shape.len()
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(tag = "name", content = "configuration", rename_all = "lowercase")]
#[enum_delegate::implement(MaybeNdim)]
pub enum ChunkGrid {
    Regular(RegularChunkGrid),
}

impl From<&[u64]> for ChunkGrid {
    fn from(value: &[u64]) -> Self {
        let cs: GridCoord = value.iter().cloned().collect();
        Self::Regular(RegularChunkGrid::new(cs))
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "name", content = "configuration")]
pub enum StorageTransformer {}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Extension(serde_json::Value);

impl Extension {
    pub fn try_understand(&self) -> Result<(), &'static str> {
        let mut map: HashMap<String, serde_json::Value> =
            serde_json::from_value(self.0.clone()).map_err(|_| "Extension is not an object")?;
        let mu_value = map
            .remove("must_understand")
            .ok_or("Extension does not define \"must_understand\"")?;
        let mu: bool = serde_json::from_value(mu_value)
            .map_err(|_| "Extension's \"must_understand\" is not a boolean")?;
        if mu {
            Err("Extension must be understood")
        } else {
            Ok(())
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ArrayMetadata {
    zarr_format: usize,
    shape: GridCoord,
    data_type: DataType,
    chunk_grid: ChunkGrid,
    chunk_key_encoding: ChunkKeyEncoding,
    fill_value: serde_json::Value,
    #[serde(default = "Vec::default")]
    storage_transformers: Vec<StorageTransformer>,
    #[serde(default = "CodecChain::default")]
    codecs: CodecChain,
    #[serde(default = "HashMap::default")]
    attributes: HashMap<String, serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dimension_names: Option<CoordVec<Option<String>>>,
    #[serde(flatten)]
    extensions: HashMap<String, Extension>,
}

impl Ndim for ArrayMetadata {
    fn ndim(&self) -> usize {
        self.shape.len()
    }
}

impl ArrayMetadata {
    pub fn new(
        zarr_format: usize,
        shape: GridCoord,
        data_type: DataType,
        chunk_grid: ChunkGrid,
        chunk_key_encoding: ChunkKeyEncoding,
        fill_value: serde_json::Value,
        storage_transformers: Vec<StorageTransformer>,
        codecs: CodecChain,
        attributes: HashMap<String, serde_json::Value>,
        dimension_names: Option<CoordVec<Option<String>>>,
        extensions: HashMap<String, Extension>,
    ) -> Self {
        Self {
            zarr_format,
            shape,
            data_type,
            chunk_grid,
            chunk_key_encoding,
            fill_value,
            storage_transformers,
            codecs,
            attributes,
            dimension_names,
            extensions,
        }
    }

    /// Ensures that all unknown extensions do not require understanding.
    pub fn try_understand_extensions(&self) -> Result<(), &'static str> {
        self.extensions
            .iter()
            .map(|(_name, config)| config.try_understand())
            .collect()
    }

    /// Ensure that all dimensioned metadata is consistent.
    pub fn validate_dimensions(&self) -> Result<(), &'static str> {
        self.union_ndim(&self.chunk_grid)?;
        if let Some(d) = &self.dimension_names {
            if d.len() != self.ndim() {
                return Err("Inconsistent dimensionality");
            }
        }
        self.codecs.validate_ndim()?;
        self.union_ndim(&self.codecs)?;

        Ok(())
    }

    pub fn get_effective_fill_value<T: ReflectedType>(&self) -> Result<T, &'static str> {
        serde_json::from_value(self.fill_value.clone())
            .map_err(|_| "Could not deserialize fill value")
    }
}

pub struct ArrayMetadataBuilder {
    shape: GridCoord,
    data_type: DataType,
    chunk_grid: Option<ChunkGrid>,
    chunk_key_encoding: Option<ChunkKeyEncoding>,
    fill_value: Option<serde_json::Value>,
    storage_transformers: Vec<StorageTransformer>,
    codecs: CodecChain,
    attributes: HashMap<String, serde_json::Value>,
    dimension_names: Option<CoordVec<Option<String>>>,
    extensions: HashMap<String, Extension>,
}

impl ArrayMetadataBuilder {
    pub fn new(shape: GridCoord, data_type: DataType) -> Self {
        Self {
            shape,
            data_type,
            chunk_grid: None,
            chunk_key_encoding: None,
            fill_value: None,
            storage_transformers: Vec::default(),
            codecs: CodecChain::default(),
            attributes: HashMap::default(),
            dimension_names: None,
            extensions: HashMap::default(),
        }
    }

    pub fn chunk_grid<T: Into<ChunkGrid>>(mut self, chunk_grid: T) -> Result<Self, &'static str> {
        let cg = chunk_grid.into();
        self.union_ndim(&cg)?;
        self.chunk_grid = Some(cg);
        Ok(self)
    }

    pub fn chunk_key_encoding<T: Into<ChunkKeyEncoding>>(mut self, chunk_key_encoding: T) -> Self {
        self.chunk_key_encoding = Some(chunk_key_encoding.into());
        self
    }

    pub fn fill_value<T: Serialize>(mut self, fill_value: T) -> Result<Self, &'static str> {
        let v = serde_json::to_value(fill_value).map_err(|_e| "Could not serialize fill value")?;
        self.data_type
            .validate_json_value(&v)
            // todo: more useful error
            .map_err(|_| "Not a valid fill value for this data type")?;
        self.fill_value = Some(v);
        Ok(self)
    }

    pub fn storage_transformers_mut(&mut self) -> &mut Vec<StorageTransformer> {
        &mut self.storage_transformers
    }

    pub fn push_storage_transformer<T: Into<StorageTransformer>>(
        mut self,
        storage_transformer: T,
    ) -> Self {
        self.storage_transformers.push(storage_transformer.into());
        self
    }

    pub fn ab_codec<T: Into<ABCodecType>>(mut self, codec: T) -> Result<Self, &'static str> {
        let c = codec.into();
        self.union_ndim(&c)?;
        self.codecs.replace_ab_codec(Some(c));
        Ok(self)
    }

    pub fn push_aa_codec<T: Into<AACodecType>>(mut self, codec: T) -> Result<Self, &'static str> {
        let c = codec.into();
        self.union_ndim(&c)?;
        self.codecs.aa_codecs_mut().push(c);
        Ok(self)
    }

    pub fn push_bb_codec<T: Into<BBCodecType>>(mut self, codec: T) -> Self {
        let c = codec.into();
        // todo: check blosc type size
        self.codecs.bb_codecs_mut().push(c);
        self
    }

    pub fn attributes_mut(&mut self) -> &mut HashMap<String, serde_json::Value> {
        &mut self.attributes
    }

    pub fn dimension_names(
        mut self,
        names: CoordVec<Option<String>>,
    ) -> Result<Self, &'static str> {
        if names.len() != self.shape.len() {
            return Err("Dimension names has wrong length");
        }
        self.dimension_names = Some(names);
        Ok(self)
    }

    pub fn extensions_mut(&mut self) -> &mut HashMap<String, Extension> {
        &mut self.extensions
    }

    pub fn build(self) -> ArrayMetadata {
        let chunk_grid = self
            .chunk_grid
            .unwrap_or_else(|| ChunkGrid::from(self.shape.as_slice()));
        let chunk_key_encoding = self.chunk_key_encoding.unwrap_or_default();
        let fill_value = self
            .fill_value
            .unwrap_or_else(|| self.data_type.default_fill_value());

        ArrayMetadata::new(
            ZARR_FORMAT,
            self.shape,
            self.data_type,
            chunk_grid,
            chunk_key_encoding,
            fill_value,
            self.storage_transformers,
            self.codecs,
            self.attributes,
            self.dimension_names,
            self.extensions,
        )
    }
}

impl Ndim for ArrayMetadataBuilder {
    fn ndim(&self) -> usize {
        self.shape.len()
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        chunk_key_encoding::V2ChunkKeyEncoding,
        codecs::{aa::TransposeCodec, ab::endian::EndianCodec, bb::gzip_codec::GzipCodec},
        data_type::{DataType, FloatSize},
    };

    use super::ArrayMetadataBuilder;
    use smallvec::smallvec;

    #[test]
    fn build_arraymeta() {
        let meta =
            ArrayMetadataBuilder::new(smallvec![100, 200, 300], DataType::Float(FloatSize::b32))
                .chunk_grid(vec![10, 10, 10].as_slice())
                .unwrap()
                .chunk_key_encoding(V2ChunkKeyEncoding::default())
                .fill_value(1.0)
                .unwrap()
                .push_aa_codec(TransposeCodec::new_f())
                .unwrap()
                .ab_codec(EndianCodec::new_little())
                .unwrap()
                .push_bb_codec(GzipCodec::default())
                .dimension_names(smallvec![
                    Some("x".to_string()),
                    None,
                    Some("z".to_string())
                ])
                .unwrap()
                .build();
    }
}
