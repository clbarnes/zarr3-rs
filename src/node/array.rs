use std::{
    collections::HashMap,
    io::{self, ErrorKind},
};

use serde::{Deserialize, Serialize};

use crate::{
    chunk_grid::{ArrayRegion, ChunkGrid, ChunkGridType},
    data_type::NBytes,
    to_usize,
    util::DimensionMismatch,
};
use crate::{
    chunk_key_encoding::{ChunkKeyEncoder, ChunkKeyEncoding},
    codecs::{
        aa::AACodecType,
        ab::{ABCodec, ABCodecType},
        bb::BBCodecType,
        ArrayRepr, CodecChain,
    },
    data_type::{DataType, ReflectedType},
    store::{ListableStore, NodeKey, ReadableStore, Store, WriteableStore},
    ArcArrayD, CoordVec, GridCoord, MaybeNdim, Ndim, ZARR_FORMAT,
};

use super::{JsonObject, ReadableMetadata, WriteableMetadata};

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "name", content = "configuration")]
pub enum StorageTransformer {}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Extension(serde_json::Value);

impl Extension {
    pub fn try_understand(&self) -> Result<(), &'static str> {
        let mut map: JsonObject =
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

/// Should be constructed using the [ArrayMetadataBuilder], which implements [Into]<[ArrayMetadata]>.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ArrayMetadata {
    zarr_format: usize,
    shape: GridCoord,
    data_type: DataType,
    chunk_grid: ChunkGridType,
    chunk_key_encoding: ChunkKeyEncoding,
    fill_value: serde_json::Value,
    #[serde(default = "Vec::default")]
    storage_transformers: Vec<StorageTransformer>,
    #[serde(default = "CodecChain::default")]
    codecs: CodecChain,
    #[serde(default = "HashMap::default")]
    attributes: JsonObject,
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

impl ReadableMetadata for ArrayMetadata {
    fn get_attributes(&self) -> &JsonObject {
        &self.attributes
    }

    fn get_zarr_format(&self) -> usize {
        self.zarr_format
    }

    fn is_array(&self) -> bool {
        true
    }
}

impl WriteableMetadata for ArrayMetadata {
    fn mutate_attributes<F, R>(&mut self, f: F) -> R
    where
        F: FnOnce(&mut JsonObject) -> R,
    {
        f(&mut self.attributes)
    }
}

impl ArrayMetadata {
    pub fn validate(&self) -> Result<(), &'static str> {
        self.try_understand_extensions()?;
        self.validate_dimensions()?;
        self.validate_codecs()?;
        Ok(())
    }

    /// Ensures that all unknown extensions do not require understanding.
    pub fn try_understand_extensions(&self) -> Result<(), &'static str> {
        self.extensions
            .values()
            .try_for_each(|config| config.try_understand())
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

    /// Check that codecs are consistent with other metadata.
    pub fn validate_codecs(&self) -> Result<(), &'static str> {
        self.data_type.valid_endian(self.codecs.endian())?;
        Ok(())
    }

    pub fn get_effective_fill_value<T: ReflectedType>(&self) -> Result<T, &'static str> {
        if T::ZARR_TYPE != self.data_type {
            return Err("Reflected type mismatches array data type");
        }
        serde_json::from_value(self.fill_value.clone())
            .map_err(|_| "Could not deserialize fill value")
    }

    /// Panics on dimension mismatch
    pub fn chunk_should_exist(&self, chunk: &GridCoord) -> bool {
        DimensionMismatch::check_coords(chunk.len(), self.ndim()).unwrap();
        self.chunk_should_exist_unchecked(chunk)
    }

    pub fn chunk_should_exist_unchecked(&self, chunk: &GridCoord) -> bool {
        let max_chunk = self
            .chunk_grid
            .voxel_chunk_unchecked(self.shape.as_slice())
            .0;
        max_chunk.iter().zip(chunk.iter()).all(|(ma, ch)| ch <= ma)
    }
}

pub struct ArrayMetadataBuilder<T: ReflectedType> {
    shape: GridCoord,
    data_type: DataType,
    chunk_grid: Option<ChunkGridType>,
    chunk_key_encoding: Option<ChunkKeyEncoding>,
    fill_value: Option<T>,
    storage_transformers: Vec<StorageTransformer>,
    codecs: CodecChain,
    attributes: JsonObject,
    dimension_names: Option<CoordVec<Option<String>>>,
    extensions: HashMap<String, Extension>,
}

impl<T: ReflectedType> ArrayMetadataBuilder<T> {
    /// Prepare metadata for a basic array with a shape and data type.
    ///
    /// At a minimum, [ArrayMetadataBuilder::chunk_grid()] should be called,
    /// as the default behaviour is to have a single chunk for the entire array.
    pub fn new(shape: &[u64]) -> Self {
        let s = shape.iter().cloned().collect();
        Self {
            shape: s,
            data_type: T::ZARR_TYPE,
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

    /// Set the chunk grid.
    ///
    /// By default, the entire array will be a single chunk.
    ///
    /// Fails if the chunk grid is incompatible with the array's dimensionality.
    pub fn chunk_grid<G: Into<ChunkGridType>>(
        mut self,
        chunk_grid: G,
    ) -> Result<Self, &'static str> {
        let cg = chunk_grid.into();
        self.union_ndim(&cg)?;
        self.chunk_grid = Some(cg);
        Ok(self)
    }

    /// Set the chunk key encoding.
    ///
    /// By default, uses the default chunk key encoding
    /// (`c/`-prefixed, `/`-separated).
    pub fn chunk_key_encoding<E: Into<ChunkKeyEncoding>>(mut self, chunk_key_encoding: E) -> Self {
        self.chunk_key_encoding = Some(chunk_key_encoding.into());
        self
    }

    /// Set the fill value.
    ///
    /// By default, uses the data type's default value, which is generally `false` or `0`.
    ///
    /// Fails if the value's JSON serialisation is not compatible
    /// with the array's data type.
    /// This means that types which have compatible JSON representations are interchangeable here:
    /// for example, a f32 fill value can be given for a f64 array,
    /// or a 2-length u8 array fill value can be given for a c128 array.
    pub fn fill_value(mut self, fill_value: T) -> Self {
        self.fill_value = Some(fill_value);
        self
    }

    /// Mutable access to the array's storage transformers.
    pub fn storage_transformers_mut(&mut self) -> &mut Vec<StorageTransformer> {
        &mut self.storage_transformers
    }

    /// Append a storage transformer to the list.
    ///
    /// N.B. this API is subject to change as there are no specified
    /// storage transformers at time of writing.
    // pub fn push_storage_transformer<T: Into<StorageTransformer>>(
    //     mut self,
    //     storage_transformer: T,
    // ) -> Self {
    //     self.storage_transformers.push(storage_transformer.into());
    //     self
    // }

    /// Set the array->bytes codec.
    ///
    /// By default, uses a little-endian [crate::codecs::ab::bytes_codec::BytesCodec].
    ///
    /// Replaces the existing AB codec.
    /// Fails if the dimensions are not compatible with the array's shape.
    pub fn ab_codec<C: Into<ABCodecType>>(mut self, codec: C) -> Result<Self, &'static str> {
        let c = codec.into();
        self.union_ndim(&c)?;
        c.valid_endian::<T>()?;
        self.codecs.replace_ab_codec(c);
        Ok(self)
    }

    /// Append an array->array codec.
    ///
    /// This will be the last AA encoder, or first AA decoder.
    ///
    /// Fails if the dimensions are not compatible with the array's shape.
    pub fn push_aa_codec<C: Into<AACodecType>>(mut self, codec: C) -> Result<Self, &'static str> {
        let c = codec.into();
        self.union_ndim(&c)?;
        self.codecs.aa_codecs_mut().push(c);
        Ok(self)
    }

    /// Append a bytes->bytes codec.
    ///
    /// This will be the last BB encoder, or first BB decoder.
    pub fn push_bb_codec<C: Into<BBCodecType>>(mut self, codec: C) -> Self {
        let c = codec.into();
        // todo: check blosc type size
        self.codecs.bb_codecs_mut().push(c);
        self
    }

    pub fn set_attribute<S: Serialize>(
        mut self,
        key: String,
        value: S,
    ) -> Result<Self, &'static str> {
        let v = serde_json::to_value(value).map_err(|_| "Could not serialize value")?;
        self.attributes.insert(key, v);
        Ok(self)
    }

    /// Set the dimension names.
    ///
    /// Fails if the number of dimension names do not match the array's dimensionality.
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

    /// Mutable access to the array's extensions.
    pub fn extensions_mut(&mut self) -> &mut HashMap<String, Extension> {
        &mut self.extensions
    }
}

impl<T: ReflectedType> From<ArrayMetadataBuilder<T>> for ArrayMetadata {
    fn from(value: ArrayMetadataBuilder<T>) -> Self {
        // todo: should this fail if there are must_understand extensions?
        let chunk_grid = value
            .chunk_grid
            .unwrap_or_else(|| ChunkGridType::from(value.shape.as_slice()));
        let chunk_key_encoding = value.chunk_key_encoding.unwrap_or_default();
        let fill_value = value.fill_value.unwrap_or_default();

        ArrayMetadata {
            zarr_format: ZARR_FORMAT,
            shape: value.shape,
            data_type: value.data_type,
            chunk_grid,
            chunk_key_encoding,
            fill_value: serde_json::to_value(fill_value).unwrap(),
            storage_transformers: value.storage_transformers,
            codecs: value.codecs,
            attributes: value.attributes,
            dimension_names: value.dimension_names,
            extensions: value.extensions,
        }
    }
}

impl<T: ReflectedType> Ndim for ArrayMetadataBuilder<T> {
    fn ndim(&self) -> usize {
        self.shape.len()
    }
}

pub struct Array<'s, S: Store, T: ReflectedType> {
    store: &'s S,
    key: NodeKey,
    meta_key: NodeKey,
    metadata: ArrayMetadata,
    fill_value: T,
}

impl<'s, S: Store, T: ReflectedType> Ndim for Array<'s, S, T> {
    fn ndim(&self) -> usize {
        self.metadata.ndim()
    }
}

impl<'s, S: Store, T: ReflectedType> ReadableMetadata for Array<'s, S, T> {
    fn get_zarr_format(&self) -> usize {
        self.metadata.get_zarr_format()
    }

    fn is_array(&self) -> bool {
        true
    }

    fn get_attributes(&self) -> &JsonObject {
        self.metadata.get_attributes()
    }
}

impl<'s, S: WriteableStore, T: ReflectedType> WriteableMetadata for Array<'s, S, T> {
    fn mutate_attributes<F, R>(&mut self, f: F) -> R
    where
        F: FnOnce(&mut JsonObject) -> R,
    {
        let result = self.metadata.mutate_attributes(f);
        // todo: fix this error
        self.write_meta().expect("metadata io error");
        result
    }
}

impl<'s, S: Store, T: ReflectedType> Array<'s, S, T> {
    /// Does not write metadata
    pub(crate) fn new(
        store: &'s S,
        key: NodeKey,
        metadata: ArrayMetadata,
    ) -> Result<Self, &'static str> {
        let mut meta_key = key.clone();
        meta_key.with_metadata();
        if T::ZARR_TYPE != metadata.data_type {
            return Err("Type annotation mismatches stored data type");
        }
        let fill_value = metadata.get_effective_fill_value()?;

        Ok(Self {
            store,
            key,
            meta_key,
            metadata,
            fill_value,
        })
    }

    pub fn key(&self) -> &NodeKey {
        &self.key
    }

    pub fn meta_key(&self) -> &NodeKey {
        &self.meta_key
    }

    pub fn store(&self) -> &'s S {
        self.store
    }

    fn chunk_repr(&self, chunk_idx: &GridCoord) -> ArrayRepr<T> {
        let shape = self.metadata.chunk_grid.chunk_shape(chunk_idx);
        ArrayRepr::new(shape.as_slice(), self.fill_value)
    }

    fn empty_chunk(&self, chunk_idx: &GridCoord) -> Result<ArcArrayD<T>, &'static str> {
        let shape = self.metadata.chunk_grid.chunk_shape(chunk_idx);

        let arr = ArcArrayD::from_elem(
            shape.into_iter().map(|s| s as usize).collect::<Vec<_>>(),
            self.fill_value,
        );
        Ok(arr)
    }
}

impl<'s, S: ReadableStore, T: ReflectedType> Array<'s, S, T> {
    pub fn from_store(store: &'s S, key: NodeKey) -> io::Result<Self> {
        let mut meta_key = key.clone();
        meta_key.with_metadata();
        if let Some(r) = store.get(&meta_key)? {
            let meta: ArrayMetadata = serde_json::from_reader(r).expect("deser error");
            Ok(Self::new(store, key, meta).unwrap())
        } else {
            Err(io::Error::new(
                ErrorKind::NotFound,
                "Group metadata not found",
            ))
        }
    }

    /// Read a chunk from the array.
    ///
    /// `Err` if IO problems; `Ok(None)` if out of bounds; panics if idx is the wrong dimensionality; `Ok(Some(array))` otherwise.
    /// Fills in empty chunks with the fill value.
    ///
    /// Includes padding values for chunks which overhang the array.
    pub fn read_chunk(&self, chunk_idx: &GridCoord) -> io::Result<Option<ArcArrayD<T>>> {
        if !(self.metadata.chunk_should_exist(chunk_idx)) {
            return Ok(None);
        }

        let key = self
            .metadata
            .chunk_key_encoding
            .chunk_key(&self.key, chunk_idx);
        if let Some(r) = self.store.get(&key)? {
            let arr = self.metadata.codecs.decode(r, self.chunk_repr(chunk_idx));
            Ok(Some(arr))
        } else {
            Ok(Some(self.empty_chunk(chunk_idx).expect("wrong data type")))
        }
    }

    fn read_partial_chunk(
        &self,
        chunk_idx: &GridCoord,
        chunk_region: &ArrayRegion,
    ) -> io::Result<Option<ArcArrayD<T>>> {
        // todo: check it fits in chunk?
        if let Some(sub_arr) = self.read_chunk(chunk_idx)? {
            let chunk_slice = chunk_region.slice_info();
            Ok(Some(sub_arr.slice_move(chunk_slice)))
        } else {
            Ok(None)
        }
    }

    pub fn read_region(&self, region: ArrayRegion) -> io::Result<Option<ArcArrayD<T>>> {
        if let Some(reg) = region.limit_extent(&self.metadata.shape) {
            let mut out =
                ArcArrayD::from_elem(to_usize(reg.shape().as_slice()).as_slice(), self.fill_value);
            let it = self.metadata.chunk_grid.chunks_in_region(&reg);
            for pc in it {
                if let Some(sub_chunk) = self.read_partial_chunk(&pc.chunk_idx, &pc.chunk_region)? {
                    let out_slice = pc.out_region.slice_info();
                    sub_chunk.assign_to(out.slice_mut(out_slice));
                }
            }
            Ok(Some(out))
        } else {
            Ok(None)
        }
    }
}

impl<'s, S: ListableStore, T: ReflectedType> Array<'s, S, T> {
    pub fn child_keys(&self) -> io::Result<Vec<NodeKey>> {
        let (_, keys) = self.store.list_dir(&self.key)?;
        Ok(keys)
    }
}

impl<'s, S: WriteableStore, T: ReflectedType> Array<'s, S, T> {
    pub(crate) fn write_meta(&self) -> io::Result<()> {
        self.store.set(&self.meta_key, |w| {
            serde_json::to_writer_pretty(w, &self.metadata).unwrap();
            Ok(())
        })?;
        Ok(())
    }

    pub fn write_chunk(&self, idx: &GridCoord, chunk: ArcArrayD<T>) -> Result<(), &'static str> {
        let shape = self.metadata.chunk_grid.chunk_shape(idx);
        if chunk
            .shape()
            .iter()
            .zip(shape.iter())
            .any(|(sh, exp)| *sh as u64 != *exp)
        {
            return Err("Chunk is the wrong shape");
        }
        let key = self.metadata.chunk_key_encoding.chunk_key(&self.key, idx);
        if chunk.iter().all(|v| v == &self.fill_value) {
            return self
                .store
                .erase(&key)
                .map(|_| ())
                .map_err(|_| "Could not erase chunk of fill value");
        }

        self.store
            .set(&key, move |w| {
                self.metadata.codecs.encode(chunk, w);
                Ok(())
            })
            .map_err(|_| "Could not get chunk writer")
    }

    fn write_partial_chunk(
        &self,
        chunk_idx: &GridCoord,
        chunk_region: &ArrayRegion,
        sub_chunk: ArcArrayD<T>,
    ) -> Result<(), &'static str> {
        let mut chunk = self
            .read_chunk(chunk_idx)
            .map_err(|_e| "IO error")?
            .unwrap();
        let chunk_slice = chunk_region.slice_info();
        sub_chunk.assign_to(chunk.slice_mut(chunk_slice));
        self.write_chunk(chunk_idx, chunk)?;
        Ok(())
    }

    pub fn write_region(
        &self,
        offset: &GridCoord,
        array: ArcArrayD<T>,
    ) -> Result<(), &'static str> {
        let shape: GridCoord = array.shape().iter().map(|n| *n as u64).collect();
        let region_opt = ArrayRegion::from_offset_shape(offset, shape.as_slice())
            .limit_extent_unchecked(&self.metadata.shape);

        if region_opt.is_none() {
            return Ok(());
        }
        let region = region_opt.unwrap();

        let slice_within = region.at_origin().slice_info();
        let array_within = array.slice(slice_within);

        let _it = self.metadata.chunk_grid.chunks_in_region_unchecked(&region);
        for pc in self.metadata.chunk_grid.chunks_in_region_unchecked(&region) {
            let arr_slice = pc.out_region.slice_info();
            let sub_arr = array_within.slice(arr_slice).to_shared();

            if pc.chunk_region.is_whole(
                &self
                    .metadata
                    .chunk_grid
                    .chunk_shape_unchecked(&pc.chunk_idx),
            ) {
                // whole chunk
                self.write_chunk(&pc.chunk_idx, sub_arr)?;
            } else {
                // partial chunk
                self.write_partial_chunk(&pc.chunk_idx, &pc.chunk_region, sub_arr)?;
            }
        }
        Ok(())
    }

    pub fn erase(self) -> io::Result<()> {
        self.store.erase_prefix(&self.key)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        chunk_key_encoding::V2ChunkKeyEncoding,
        codecs::{aa::TransposeCodec, ab::bytes_codec::BytesCodec, bb::gzip_codec::GzipCodec},
    };

    use super::{ArrayMetadata, ArrayMetadataBuilder};
    use smallvec::smallvec;

    #[test]
    fn build_arraymeta() {
        let _meta: ArrayMetadata = ArrayMetadataBuilder::new(&[100, 200, 300])
            .chunk_grid(vec![10, 10, 10].as_slice())
            .unwrap()
            .chunk_key_encoding(V2ChunkKeyEncoding::default())
            .fill_value(1.0)
            .push_aa_codec(TransposeCodec::new_transpose(3))
            .unwrap()
            .ab_codec(BytesCodec::new_little())
            .unwrap()
            .push_bb_codec(GzipCodec::default())
            .dimension_names(smallvec![
                Some("x".to_string()),
                None,
                Some("z".to_string())
            ])
            .unwrap()
            .into();
    }
}
