use zarr3::codecs::bb::gzip_codec::GzipCodec;
use zarr3::prelude::smallvec::smallvec;
use zarr3::prelude::{create_root_group, ArrayMetadataBuilder, ArrayRegion, GroupMetadata};
use zarr3::store::filesystem::FileSystemStore;
use zarr3::ArcArrayD;

fn main() -> anyhow::Result<()> {
    // Create a temporary directory for the zarr root to live in
    let tmp = tempdir::TempDir::new("zarr3-roundtrip")?;
    // Create a zarr store called in this directory called "root.zarr"
    let store = FileSystemStore::create(tmp.path().join("root.zarr"), true)?;

    // Create a group with appropriate metadata at the root of this store (could also be an array)
    let root_group = create_root_group(&store, GroupMetadata::default())?;

    // Build metadata for an array
    let arr_meta = ArrayMetadataBuilder::<i32>::new(&[20, 10])
        .chunk_grid(vec![10, 5].as_slice())
        .unwrap()
        .fill_value(-1)
        .push_bb_codec(GzipCodec::default())
        .into();

    // Create the (empty) array with this metadata below the group created above
    let arr = root_group.create_array::<i32>("my_array".parse()?, arr_meta)?;

    // Write some data into the middle of the array
    let data = ArcArrayD::from_shape_vec(vec![10, 6], (10..70).collect())?;

    let offset = smallvec![5, 2];
    arr.write_region(&offset, data).unwrap();

    // Read the whole array and print it to stdout
    let output = arr
        .read_region(ArrayRegion::from_offset_shape(&[0, 0], &[20, 10]))
        .unwrap()
        .unwrap();
    println!("{:?}", output);
    Ok(())
}
