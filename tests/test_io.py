import numpy as np
import pytest
import xarray as xr

from cop_ard_app.io import (
    assign_attrs,
    check_write_arguments,
    copy,
    delete,
    dirname,
    drop_encodings,
    exists,
    generate_empty_dataset,
    get_crc,
    get_duration,
    get_id_product,
    get_polarization_code,
    glob,
    is_local,
    isdir,
    isfile,
    makedirs,
    move,
    normpath,
    path_to_dict,
    read,
    save_empty_dataset,
    touch,
    update_attrs,
    validate_encoding,
    write,
)

STAC = {
    "properties": {
        "created": "2024-06-05T12:00:00Z",
        "processing:datetime": "2024-06-06T00:00:00Z",
        "product:type": "S1",
        "start_datetime": "2024-05-27T04:57:02.981885Z",
        "end_datetime": "2024-05-27T05:03:29.839352Z",
        "platform": "sentinel-1",
        "sat:relative_orbit": 12,
        "eopf:datatake_id": "5062965948",
        "sar:polarizations": ["VV", "VH"],
    }
}
STAC_WITHOUT_DATATAKE = {"properties": {k: v for k, v in STAC["properties"].items() if k != "eopf:datatake_id"}}
BURST_ID = "iw1_254"
TILE_NAME = "t42"

EXPECTED_DURATION = "0387"
EXPECTED_PRODUCT_ID = "S1_20240527T045702_0387_1012_S9E0_12DC6BABC_DV"
EXPECTED_PRODUCT_ID_BURST = f"{EXPECTED_PRODUCT_ID}_01_000254"
EXPECTED_PRODUCT_ID_TILE = f"{EXPECTED_PRODUCT_ID_BURST}_T42"
EXPECTED_PRODUCT_ID_WITHOUT_DATATAKE = "S1_20240527T045702_0387_1012_SA95_000000_DV"

CRC_INPUT = {"a": 1, "b": {"c": 2}}
CRC_INPUT_STABLE = "A36"
CRC_STAC = "563"

DURATION_ROUNDING = {
    "properties": {
        "start_datetime": "2020-01-01T00:00:00Z",
        "end_datetime": "2020-01-01T00:00:00.6Z",
    }
}

LOCAL_PATH = "out/../product.zarr"
FILE_URL = "file:///a/b/product.zarr"
S3_PATH = "s3://bucket/product.zarr"
HTTP_URL = "https://host/a/../b.zarr"

NESTED = {"a": {"b": {"c": 42}}, "x": 7}

ENCODING_INPUT = {
    "v": {"chunks": (2, 2), "bogus_rule": 1},
    "missing_var": {"chunks": (4,)},
}

EMPTY_DATA_VARS = ["v"]
EMPTY_DIMS = {"v": ["y", "x"]}
EMPTY_SHAPE = {"v": [4, 6]}
EMPTY_CHUNKS = {"y": 2, "x": 3}
EMPTY_DTYPES = {"v": "float64"}
EMPTY_COORDS = {"y": [10, 11, 12, 13], "x": list(range(6))}
EMPTY_ATTRS = {"title": "empty store"}
EMPTY_ATTRS_DATA_VARS = {"v": {"units": "m"}}

ROUNDTRIP_VALUES = [[0, 1, 2, 3], [4, 5, 6, 7]]
ROUNDTRIP_ATTRS = {"note": "round-trip"}


def _empty_spec():
    """Fresh copy of the structure specification shared by generate/save_empty_dataset."""
    return {
        "data_vars": list(EMPTY_DATA_VARS),
        "dims": {k: list(v) for k, v in EMPTY_DIMS.items()},
        "shape": {k: list(v) for k, v in EMPTY_SHAPE.items()},
        "chunks": dict(EMPTY_CHUNKS),
        "dtype": dict(EMPTY_DTYPES),
        "coords": {k: list(v) for k, v in EMPTY_COORDS.items()},
        "attrs": dict(EMPTY_ATTRS),
        "attrs_data_vars": {k: dict(v) for k, v in EMPTY_ATTRS_DATA_VARS.items()},
    }


def _make_roundtrip_dataset():
    """A small, fully materialized Dataset used for write/read round-trips."""
    return xr.Dataset(
        {
            "v": xr.DataArray(
                np.arange(8, dtype=np.int32).reshape(2, 4),
                dims=("y", "x"),
                coords={"y": [10, 11], "x": [0, 1, 2, 3]},
            )
        },
        attrs=dict(ROUNDTRIP_ATTRS),
    )


def _make_encoded_dataset():
    """A small Dataset with non-empty encodings at the dataset, variable and coord level."""
    ds = xr.Dataset(
        {
            "v": xr.DataArray(
                [1, 2, 3],
                dims=("x",),
                coords={"x": [1, 2, 3]},
            )
        }
    )
    ds.encoding["compressor"] = "zlib"
    ds["v"].encoding["chunks"] = (2, 2)
    ds["x"].encoding["_FillValue"] = -1
    return ds


class TestAssignAttrs:
    """Unit tests for cop_ard_app.io.assign_attrs."""

    @pytest.mark.unit
    def test_assigns_attributes_and_returns_same_object(self):
        attrs = {"a": 1, "b": {"c": 2}}
        ds = xr.Dataset({"v": xr.DataArray([1, 2], dims=("x",))})
        ds.attrs["a"] = "to-be-overwritten"
        out = assign_attrs(ds, attrs)
        assert out is ds
        assert dict(out.attrs) == attrs

    @pytest.mark.unit
    def test_assigns_attributes_on_data_tree_root(self):
        tree = xr.DataTree.from_dict({"a": xr.Dataset()})
        out = assign_attrs(tree, {"z": 9})
        assert out is tree
        assert dict(out.attrs) == {"z": 9}


class TestCheckWriteArguments:
    """Unit tests for cop_ard_app.io.check_write_arguments."""

    @pytest.mark.unit
    def test_rejects_non_xarray_data(self):
        with pytest.raises(TypeError, match="Data type must be"):
            check_write_arguments({"not": "xarray"}, "out.zarr")

    @pytest.mark.unit
    def test_passes_dataset_through_with_normalized_path(self):
        ds = xr.Dataset()
        mode, file_path, data, data_type = check_write_arguments(ds, LOCAL_PATH)
        assert mode == "w"
        assert file_path == "product.zarr"
        assert data is ds
        assert data_type == "xr.Dataset"

    @pytest.mark.unit
    def test_data_array_promoted_to_named_dataset(self):
        arr = xr.DataArray([1, 2], name="sig")
        mode, _, data, data_type = check_write_arguments(arr, S3_PATH, mode="a", region={"x": slice(0, 1)})
        assert mode == "r+"
        assert list(data.variables) == ["sig"]
        assert data_type == "xr.DataArray"

    @pytest.mark.unit
    def test_unnamed_data_array_uses_default_variable_name(self):
        arr = xr.DataArray([1, 2])
        _, _, data, _ = check_write_arguments(arr, "out.zarr")
        assert list(data.variables) == ["v"]

    @pytest.mark.unit
    def test_data_tree_passes_through(self):
        tree = xr.DataTree()
        mode, file_path, data, data_type = check_write_arguments(tree, HTTP_URL)
        assert mode == "w"
        assert file_path == "https://host/b.zarr"
        assert data is tree
        assert data_type == "xr.DataTree"


class TestValidateEncoding:
    """Unit tests for cop_ard_app.io.validate_encoding."""

    @pytest.mark.unit
    def test_none_passes_through(self):
        assert validate_encoding(_make_encoded_dataset(), None) is None

    @pytest.mark.unit
    def test_strips_unknown_variables_and_rules(self):
        result = validate_encoding(_make_encoded_dataset(), ENCODING_INPUT)
        assert result == {"v": {"chunks": (2, 2)}}

    @pytest.mark.unit
    def test_drops_variable_when_all_rules_invalid(self):
        assert validate_encoding(_make_encoded_dataset(), {"v": {"bogus_rule": 1}}) == {}

    @pytest.mark.unit
    def test_accepts_variables_of_data_tree_descendants(self):
        tree = xr.DataTree.from_dict({"a": _make_encoded_dataset()})
        assert validate_encoding(tree, ENCODING_INPUT) == {"v": {"chunks": (2, 2)}}


class TestIsLocal:
    """Unit tests for cop_ard_app.io.is_local."""

    @pytest.mark.unit
    def test_local_paths_drop_storage_options(self):
        assert is_local(LOCAL_PATH, {"storage": "opt"}) is None
        assert is_local(FILE_URL, {"storage": "opt"}) is None

    @pytest.mark.unit
    def test_remote_paths_override_listings_cache_without_mutation(self):
        options = {"token": "secret"}
        assert is_local(S3_PATH, options) == {"token": "secret", "use_listings_cache": False}
        assert options == {"token": "secret"}

    @pytest.mark.unit
    def test_remote_path_without_options_returns_none(self):
        assert is_local(HTTP_URL, None) is None


class TestPathToDict:
    """Unit tests for cop_ard_app.io.path_to_dict."""

    @pytest.mark.unit
    def test_reaches_nested_value(self):
        assert path_to_dict(NESTED, "a.b.c") == 42
        assert path_to_dict(NESTED, "a", pattern=".") == {"b": {"c": 42}}

    @pytest.mark.unit
    def test_missing_keys_return_none(self):
        assert path_to_dict(NESTED, "missing") is None
        assert path_to_dict(NESTED, "a.b.z") is None

    @pytest.mark.unit
    def test_traversing_past_a_leaf_returns_none(self):
        assert path_to_dict(NESTED, "x.y") is None

    @pytest.mark.unit
    def test_custom_delimiter(self):
        assert path_to_dict(NESTED, "a/b/c", pattern="/") == 42


class TestGetPolarizationCode:
    """Unit tests for cop_ard_app.io.get_polarization_code."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "polarizations,product_level,expected",
        [
            (["VV", "VH"], True, "DV"),
            (["vh", "vv"], True, "DV"),
            (["HH", "HV"], True, "DH"),
            (["HH"], True, "SH"),
            (["HH"], False, "HH"),
            (["VV"], True, "SV"),
            (["VV"], False, "VV"),
            (["HV"], True, "HV"),
            (["VH"], False, "VH"),
        ],
    )
    def test_maps_polarization_sets_to_s1_codes(self, polarizations, product_level, expected):
        assert get_polarization_code(polarizations, product_level) == expected

    @pytest.mark.unit
    def test_unrecognized_combination_raises(self):
        with pytest.raises(ValueError, match="Unrecognized polarization combination"):
            get_polarization_code(["VV", "HH"])


class TestGetDuration:
    """Unit tests for cop_ard_app.io.get_duration."""

    @pytest.mark.unit
    def test_duration_padded_to_four_digits(self):
        # 386.857 s between the start and end dates of STAC, rounded to 387
        assert get_duration(STAC) == EXPECTED_DURATION

    @pytest.mark.unit
    def test_sub_second_durations_round(self):
        assert get_duration(DURATION_ROUNDING) == "0001"


class TestGetIdProduct:
    """Unit tests for cop_ard_app.io.get_id_product."""

    @pytest.mark.unit
    def test_stac_maps_to_exact_product_id(self):
        assert get_id_product(STAC) == EXPECTED_PRODUCT_ID

    @pytest.mark.unit
    def test_burst_and_tile_suffixes_are_appended(self):
        assert get_id_product(STAC, burst_id=BURST_ID) == EXPECTED_PRODUCT_ID_BURST
        assert get_id_product(STAC, burst_id=BURST_ID, tile_name=TILE_NAME) == EXPECTED_PRODUCT_ID_TILE

    @pytest.mark.unit
    def test_burst_id_data_keeps_the_base_id(self):
        assert get_id_product(STAC, burst_id="data") == EXPECTED_PRODUCT_ID

    @pytest.mark.unit
    def test_missing_datatake_id_falls_back_to_zeros(self):
        assert get_id_product(STAC_WITHOUT_DATATAKE) == EXPECTED_PRODUCT_ID_WITHOUT_DATATAKE

    @pytest.mark.unit
    def test_input_stac_is_not_mutated(self):
        get_id_product(STAC)
        assert "created" in STAC["properties"]
        assert "processing:datetime" in STAC["properties"]


class TestGetCrc:
    """Unit tests for cop_ard_app.io.get_crc."""

    @pytest.mark.unit
    def test_known_attrs_map_to_stable_digest(self):
        assert get_crc(CRC_INPUT) == CRC_INPUT_STABLE
        assert get_crc(STAC) == CRC_STAC

    @pytest.mark.unit
    def test_key_order_is_irrelevant(self):
        assert get_crc({"b": {"c": 2}, "a": 1}) == CRC_INPUT_STABLE

    @pytest.mark.unit
    def test_content_change_changes_digest(self):
        assert get_crc(STAC) != get_crc(STAC_WITHOUT_DATATAKE)


class TestDropEncodings:
    """Unit tests for cop_ard_app.io.drop_encodings."""

    @pytest.mark.unit
    def test_clears_dataset_variable_and_coord_encodings(self):
        ds = _make_encoded_dataset()
        drop_encodings(ds)
        assert ds.encoding == {}
        assert dict(ds["v"].encoding) == {}
        assert dict(ds["x"].encoding) == {}

    @pytest.mark.unit
    def test_clears_data_array_encodings_including_coordinates(self):
        arr = _make_encoded_dataset()["v"]
        arr.encoding["bogus"] = 1
        arr.variable.encoding["bogus"] = 1
        drop_encodings(arr)
        assert dict(arr.encoding) == {}
        assert dict(arr.variable.encoding) == {}
        assert dict(arr["x"].encoding) == {}

    @pytest.mark.unit
    def test_recurses_over_a_data_tree(self):
        tree = xr.DataTree.from_dict({"a": _make_encoded_dataset(), "b/c": _make_encoded_dataset()})
        drop_encodings(tree)
        for node in tree.subtree:
            if node.ds is not None:
                assert node.ds.encoding == {}
                for var in node.ds.variables.values():
                    assert dict(var.encoding) == {}


class TestGenerateEmptyDataset:
    """Unit tests for cop_ard_app.io.generate_empty_dataset."""

    @pytest.mark.unit
    def test_generates_dask_backed_structure_from_spec(self):
        ds = generate_empty_dataset(**_empty_spec())
        assert list(ds.data_vars) == EMPTY_DATA_VARS
        assert ds["v"].shape == (4, 6)
        assert str(ds["v"].dtype) == EMPTY_DTYPES["v"]
        assert ds["v"].chunks == ((2, 2), (3, 3))
        assert list(ds.coords["x"].values) == EMPTY_COORDS["x"]
        assert dict(ds.attrs) == EMPTY_ATTRS
        assert dict(ds["v"].attrs) == EMPTY_ATTRS_DATA_VARS["v"]


class TestSaveEmptyDataset:
    """Unit tests for cop_ard_app.io.save_empty_dataset."""

    @pytest.mark.unit
    def test_creates_zarr_store_with_structure_but_no_data(self, tmp_path):
        store = str(tmp_path / "empty.zarr")
        returned = save_empty_dataset(**_empty_spec(), data_path=store)
        assert isinstance(returned, xr.Dataset)
        assert (tmp_path / "empty.zarr").is_dir()
        back = read(store, as_type="xr.Dataset")
        assert dict(back.attrs) == {**EMPTY_ATTRS, "store_path": store}
        assert dict(back["v"].attrs) == EMPTY_ATTRS_DATA_VARS["v"]
        assert back["v"].shape == (4, 6)
        assert np.isnan(back["v"].values).all()


class TestUpdateAttrs:
    """Unit tests for cop_ard_app.io.update_attrs."""

    @pytest.mark.unit
    def test_updates_root_group_attributes_in_place(self, tmp_path):
        store = str(tmp_path / "store.zarr")
        write(_make_roundtrip_dataset(), store, skip_read=True)
        update_attrs({"updated": 42}, store)
        back = read(store, as_type="xr.Dataset")
        assert back.attrs["updated"] == 42
        assert back.attrs["note"] == ROUNDTRIP_ATTRS["note"]

    @pytest.mark.unit
    def test_updates_attributes_of_a_group(self, tmp_path):
        store = str(tmp_path / "tree.zarr")
        tree = xr.DataTree.from_dict({"g": _make_roundtrip_dataset()})
        write(tree, store, skip_read=True)
        update_attrs({"group_attr": 7}, store, group="g")
        node = read(store, group="g")
        assert node.attrs["group_attr"] == 7
        assert node["v"].values.tolist() == ROUNDTRIP_VALUES


class TestExists:
    """Unit tests for cop_ard_app.io.exists."""

    @pytest.mark.unit
    def test_reports_existing_and_missing_paths(self, tmp_path):
        present = tmp_path / "present.txt"
        present.write_text("x")
        assert exists(str(present))
        assert exists(str(tmp_path))
        assert not exists(str(tmp_path / "ghost.txt"))


class TestIsdir:
    """Unit tests for cop_ard_app.io.isdir."""

    @pytest.mark.unit
    def test_distinguishes_directories_from_files(self, tmp_path):
        present = tmp_path / "present.txt"
        present.write_text("x")
        assert isdir(str(tmp_path))
        assert not isdir(str(present))
        assert not isdir(str(tmp_path / "ghost"))


class TestIsfile:
    """Unit tests for cop_ard_app.io.isfile."""

    @pytest.mark.unit
    def test_distinguishes_files_from_directories(self, tmp_path):
        present = tmp_path / "present.txt"
        present.write_text("x")
        assert isfile(str(present))
        assert not isfile(str(tmp_path))
        assert not isfile(str(tmp_path / "ghost"))


class TestTouch:
    """Unit tests for cop_ard_app.io.touch."""

    @pytest.mark.unit
    def test_creates_an_empty_file(self, tmp_path):
        target = tmp_path / "touched.txt"
        touch(str(target))
        assert exists(str(target))
        assert target.stat().st_size == 0


class TestDelete:
    """Unit tests for cop_ard_app.io.delete."""

    @pytest.mark.unit
    def test_deletes_an_existing_file(self, tmp_path):
        target = tmp_path / "gone.txt"
        target.write_text("x")
        delete(str(target))
        assert not exists(str(target))

    @pytest.mark.unit
    def test_recursive_directory_delete(self, tmp_path):
        nested = tmp_path / "d1/d2"
        makedirs(str(nested))
        assert isdir(str(nested))
        delete(str(tmp_path / "d1"), recursive=True)
        assert not exists(str(tmp_path / "d1"))

    @pytest.mark.unit
    def test_missing_path_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            delete(str(tmp_path / "ghost.txt"))

    @pytest.mark.unit
    def test_allow_non_existent_swallows_missing_paths(self, tmp_path):
        delete(str(tmp_path / "ghost.txt"), allow_non_existent=True)


class TestMakedirs:
    """Unit tests for cop_ard_app.io.makedirs."""

    @pytest.mark.unit
    def test_creates_nested_directories(self, tmp_path):
        nested = tmp_path / "d1/d2"
        makedirs(str(nested))
        assert isdir(str(nested))

    @pytest.mark.unit
    def test_exist_ok_controls_recreation_errors(self, tmp_path):
        nested = tmp_path / "d1/d2"
        makedirs(str(nested))
        makedirs(str(nested), exist_ok=True)
        with pytest.raises(FileExistsError):
            makedirs(str(nested), exist_ok=False)


class TestGlob:
    """Unit tests for cop_ard_app.io.glob."""

    @pytest.mark.unit
    def test_returns_sorted_matches_of_a_pattern(self, tmp_path):
        for name in ["b.txt", "a.txt", "c.md"]:
            (tmp_path / name).write_text("x")
        expected = [str(tmp_path / "a.txt"), str(tmp_path / "b.txt")]
        assert glob(str(tmp_path / "*.txt"), sort=True) == expected


class TestNormpath:
    """Unit tests for cop_ard_app.io.normpath."""

    @pytest.mark.unit
    def test_normalizes_local_paths(self):
        assert normpath("a/../b//c") == "b/c"
        assert normpath("./out/product.zarr") == "out/product.zarr"

    @pytest.mark.unit
    def test_preserves_url_prefixes(self):
        assert normpath("s3://a/../b") == "s3://b"
        assert normpath("https://a/x//y") == "https://a/x/y"


class TestDirname:
    """Unit tests for cop_ard_app.io.dirname."""

    @pytest.mark.unit
    def test_returns_parent_of_normalized_path(self):
        assert dirname("a/b/c") == "a/b"
        assert dirname("s3://bucket/a/b.zarr") == "s3://bucket/a"


class TestCopy:
    """Unit tests for cop_ard_app.io.copy."""

    @pytest.mark.unit
    def test_copies_local_file_content_to_destination(self, tmp_path):
        src = tmp_path / "src.bin"
        dst = tmp_path / "dst.bin"
        src.write_text("payload")
        result = copy(str(src), str(dst))
        assert result == str(dst)
        assert dst.read_text() == "payload"
        assert src.read_text() == "payload"

    @pytest.mark.unit
    def test_missing_source_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            copy(str(tmp_path / "ghost.bin"), str(tmp_path / "dst.bin"))

    @pytest.mark.unit
    def test_only_if_different_fs_skips_local_to_local(self, tmp_path):
        src = tmp_path / "src.bin"
        src.write_text("payload")
        result = copy(str(src), str(tmp_path / "nope.bin"), only_if_different_fs=True)
        assert result == str(src)
        assert not exists(str(tmp_path / "nope.bin"))


class TestMove:
    """Unit tests for cop_ard_app.io.move."""

    @pytest.mark.unit
    def test_moves_local_file_content_to_destination(self, tmp_path):
        src = tmp_path / "src.bin"
        dst = tmp_path / "dst.bin"
        src.write_text("payload")
        result = move(str(src), str(dst))
        assert result == str(dst)
        assert not exists(str(src))
        assert dst.read_text() == "payload"

    @pytest.mark.unit
    def test_missing_source_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            move(str(tmp_path / "ghost.bin"), str(tmp_path / "dst.bin"))


class TestWrite:
    """Unit tests for cop_ard_app.io.write."""

    @pytest.mark.unit
    def test_dataset_round_trips_through_zarr_store(self, tmp_path):
        store = str(tmp_path / "store.zarr")
        result = write(_make_roundtrip_dataset(), store)
        assert result is not None
        assert result["v"].values.tolist() == ROUNDTRIP_VALUES
        assert dict(result.attrs) == {**ROUNDTRIP_ATTRS, "store_path": store}

    @pytest.mark.unit
    def test_skip_read_returns_none_after_writing(self, tmp_path):
        store = str(tmp_path / "skipped.zarr")
        assert write(_make_roundtrip_dataset(), store, skip_read=True) is None
        assert (tmp_path / "skipped.zarr").is_dir()

    @pytest.mark.unit
    def test_invalid_encoding_keys_are_ignored_during_write(self, tmp_path):
        store = str(tmp_path / "enc.zarr")
        result = write(_make_roundtrip_dataset(), store, encoding=ENCODING_INPUT)
        assert result["v"].values.tolist() == ROUNDTRIP_VALUES


class TestRead:
    """Unit tests for cop_ard_app.io.read."""

    @pytest.mark.unit
    def test_reads_data_tree_with_injected_store_paths(self, tmp_path):
        store = str(tmp_path / "tree.zarr")
        tree = xr.DataTree.from_dict({"g": _make_roundtrip_dataset()})
        write(tree, store, skip_read=True)
        dt = read(store)
        assert [node.path for node in dt.subtree] == ["/", "/g"]
        assert dt["g"]["v"].values.tolist() == ROUNDTRIP_VALUES
        assert dt["g"].attrs["store_path"] == f"{store}/g"

    @pytest.mark.unit
    def test_single_variable_store_reads_as_data_array(self, tmp_path):
        store = str(tmp_path / "single.zarr")
        write(xr.Dataset({"only": xr.DataArray([1, 2], dims=("x",))}), store, skip_read=True)
        arr = read(store, as_type="xr.DataArray")
        assert isinstance(arr, xr.DataArray)
        assert arr.values.tolist() == [1, 2]
        assert arr.attrs["store_path"] == f"{store}/only"

    @pytest.mark.unit
    def test_multi_variable_store_cannot_cast_to_data_array(self, tmp_path):
        store = str(tmp_path / "multi.zarr")
        write(
            xr.Dataset({"a": xr.DataArray([1, 2], dims=("x",)), "b": xr.DataArray([3, 4], dims=("x",))}),
            store,
            skip_read=True,
        )
        with pytest.raises(ValueError, match="Cannot cast to DataArray"):
            read(store, as_type="xr.DataArray")
