use zarr3::prelude::smallvec::smallvec;
use zarr3::prelude::{create_root_group, ArrayMetadataBuilder, ArrayRegion, GroupMetadata};
use zarr3::store::filesystem::FileSystemStore;
use zarr3::ArcArrayD;

fn main() {
    let tmp = tempdir::TempDir::new("zarr3-roundtrip").unwrap();
    let store = FileSystemStore::create(tmp.path().join("root.zarr"), true).unwrap();

    let root_group = create_root_group(&store, GroupMetadata::default()).unwrap();

    let arr_meta = ArrayMetadataBuilder::<i32>::new(&[20, 10])
        .chunk_grid(vec![10, 5].as_slice())
        .unwrap()
        .fill_value(-1)
        .build();

    let arr = root_group
        .create_array::<i32>("my_array".parse().unwrap(), arr_meta)
        .unwrap();

    let data = ArcArrayD::from_shape_vec(vec![10, 6], (10..70).collect()).unwrap();

    let offset = smallvec![5, 2];
    arr.write_region(&offset, data).unwrap();

    let output = arr
        .read_region(ArrayRegion::from_offset_shape(&[0, 0], &[20, 10]))
        .unwrap()
        .unwrap();
    println!("{:?}", output);
}
