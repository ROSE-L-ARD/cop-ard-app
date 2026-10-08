import numpy as np
import pytest
import xarray as xr

from cop_ard_app import computation
from cop_ard_app.computation import (
    DummyClient,
    DummyFuture,
    apply_overlap,
    check_inputs,
    client_manager,
    conditional_delayed,
    get_chunks,
    get_params,
    init_zarr_groups,
    map,
    map_blocks,
    retrieve_client,
    setup,
    try_to_combine,
    try_to_store,
)

SHAPE = (6, 4)
DIMS = ("x", "y")
X_CHUNKSIZE = 3
Y_CHUNKSIZE = 2
RESERVED_KWARGS = ("job_id", "chunk", "block_id", "chunk_ov")
GROUPS = ("g1", "g2")
OFFSET = 100


def _zeros():
    return xr.DataArray(np.zeros(SHAPE), dims=DIMS)


def _column(values, start):
    return xr.DataArray(
        np.asarray(values),
        dims=("x",),
        coords={"x": [start + i for i in range(len(values))]},
    )


def _stub_function(alpha, beta=2):
    return alpha


def _grid():
    return xr.DataArray(
        np.arange(SHAPE[0] * SHAPE[1]).reshape(SHAPE),
        dims=DIMS,
        coords={"x": np.arange(SHAPE[0]), "y": np.arange(SHAPE[1])},
        name="d",
    )


def _pass_through(obj):
    return obj


def _offset_stub(obj, offset):
    return obj + offset


def _block_sizes(obj, obj_ov):
    return (obj.sizes["x"], obj_ov.sizes["x"])


def _plus_one(value):
    return value + 1


def _named_column(index):
    return _column([10 * index], 10 * index).rename("q")


_module_map = computation.map


def _map_without_dask(func, args):
    return _module_map(func, args, enable_dask=False)


class TestConditionalDelayed:
    """Unit tests for cop_ard_app.computation.conditional_delayed (enable_dask=False)."""

    @pytest.mark.unit
    def test_disabled_decorator_returns_the_function_unchanged(self):
        def stub(value):
            return value + 1

        wrapped = conditional_delayed(False, "combined")(stub)
        assert wrapped is stub
        assert wrapped(1) == 2


class TestGetChunks:
    """Unit tests for cop_ard_app.computation.get_chunks with plain (non-dask) payloads."""

    @pytest.mark.unit
    def test_explicit_chunks_are_returned_verbatim(self):
        chunks = {"x": ((0, 5), (5, 6)), "y": ((0, 2), (2, 4))}
        out_chunks, out_indices, num_chunks = get_chunks(_zeros(), chunks=chunks)
        assert out_chunks == chunks
        assert out_indices == {"x": (0, 0), "y": (0, 0)}
        assert num_chunks == 2

    @pytest.mark.unit
    def test_chunk_sizes_are_expanded_over_the_dimension_product(self):
        out_chunks, out_indices, num_chunks = get_chunks(_zeros(), chunksizes={"x": X_CHUNKSIZE, "y": Y_CHUNKSIZE})
        assert out_chunks == {
            "x": ((0, 3), (0, 3), (3, 6), (3, 6)),
            "y": ((0, 2), (2, 4), (0, 2), (2, 4)),
        }
        assert out_indices == {"x": (0, 0, 1, 1), "y": (0, 1, 0, 1)}
        assert num_chunks == 4

    @pytest.mark.unit
    def test_unlisted_dimension_becomes_a_single_repeated_chunk(self):
        out_chunks, out_indices, num_chunks = get_chunks(_zeros(), chunksizes={"x": X_CHUNKSIZE})
        assert out_chunks == {"x": ((0, 3), (3, 6)), "y": ((0, 4), (0, 4))}
        assert out_indices == {"x": (0, 1), "y": (0, 0)}
        assert num_chunks == 2

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "invalid_kwargs, message",
        [
            ({"chunks": {"x": ((0, 3), (3, 6))}}, "chunks must contain all the data dimensions"),
            ({"chunks": {"x": ((0, 3),), "y": ((0, 1), (1, 4))}}, "different number of elements"),
            ({"chunks": {"x": ((3, 1),), "y": ((0, 4),)}}, "lower than or equal"),
            ({"chunks": {"x": ((0, 9),), "y": ((0, 4),)}}, "lower than the size"),
            ({"chunksizes": {"z": 2}}, "must be a subset of the data"),
        ],
    )
    def test_invalid_chunking_arguments_are_rejected(self, invalid_kwargs, message):
        with pytest.raises(Exception, match=message):
            get_chunks(_zeros(), **invalid_kwargs)


class TestApplyOverlap:
    """Unit tests for cop_ard_app.computation.apply_overlap."""

    @pytest.mark.unit
    def test_overlap_is_clipped_at_zero_and_added_at_the_end(self):
        chunks = {"x": ((0, 4), (4, 8)), "y": ((0, 2),)}
        out = apply_overlap(chunks, {"x": 3, "z": 1})
        assert out == {"x": [[0, 7], [1, 11]], "y": [[0, 2]]}
        # overlap sizes for dimensions absent from chunks are ignored
        assert "z" not in out


class TestTryToCombine:
    """Unit tests for cop_ard_app.computation.try_to_combine."""

    @pytest.mark.unit
    def test_empty_results_yield_none(self):
        assert try_to_combine([]) is None

    @pytest.mark.unit
    def test_same_dimension_blocks_are_concatenated(self):
        first = _column([1, 2, 3], 0)
        second = _column([4, 5], 3)
        out = try_to_combine([first, second])
        assert isinstance(out, xr.DataArray)
        assert np.array_equal(out.values, np.arange(1, 6))
        assert np.array_equal(out["x"].values, np.arange(5))

    @pytest.mark.unit
    def test_per_block_lists_are_zipped_before_being_combined(self):
        first = _column([1, 2, 3], 0)
        second = _column([4, 5], 3)
        out = try_to_combine([[first], [second]])
        assert isinstance(out, xr.DataArray)
        assert np.array_equal(out.values, np.arange(1, 6))

    @pytest.mark.unit
    def test_non_xarray_payloads_are_returned_unchanged(self):
        assert try_to_combine([1, 2, 3]) == [1, 2, 3]

    @pytest.mark.unit
    def test_conflicting_objects_fall_back_to_the_results_list(self):
        first = xr.DataArray(np.ones((2, 2)), dims=DIMS)
        second = xr.DataArray(np.zeros((2, 2)), dims=DIMS)
        results = [first, second]
        out = try_to_combine(results)
        assert isinstance(out, list)
        assert len(out) == 2
        assert np.array_equal(out[0].values, np.ones((2, 2)))


class TestDummyClient:
    """Unit tests for the public call surface of DummyClient and DummyFuture."""

    @pytest.mark.unit
    def test_scatter_returns_the_payload_unchanged(self):
        client = DummyClient()
        assert client.name == "dummy-client"
        payload = np.arange(4)
        assert client.scatter(payload, borrow=False) is payload

    @pytest.mark.unit
    def test_compute_wraps_each_result_in_a_dummy_future(self):
        futures = DummyClient().compute([1, "two", None])
        assert all(f.name == "dummy-client" for f in futures)
        assert [f.result() for f in futures] == [1, "two", None]

    @pytest.mark.unit
    def test_map_and_gather_apply_the_function_locally(self):
        def add(a, b):
            return a + b

        futures = DummyClient().map(add, [1, 2], [10, 100])
        assert [f.result() for f in futures] == [11, 102]
        assert DummyClient().gather(futures) == [11, 102]

    @pytest.mark.unit
    def test_cancel_and_restart_are_no_ops(self):
        client = DummyClient()
        assert client.cancel(1, 2, hard=True) is None
        assert client.restart(hard=True) is None

    @pytest.mark.unit
    def test_dummy_future_holds_its_result_directly(self):
        payload = {"k": [1]}
        future = DummyFuture(payload)
        assert future.name == "dummy-client"
        assert future.result() is payload


class TestGetParams:
    """Unit tests for cop_ard_app.computation.get_params."""

    @pytest.mark.unit
    def test_signature_parameters_and_valid_kwargs_are_returned(self):
        func_parameters, kwargs = get_params(_stub_function, foo="bar")
        assert list(func_parameters) == ["alpha", "beta"]
        assert kwargs == {"foo": "bar"}

    @pytest.mark.unit
    @pytest.mark.parametrize("reserved", RESERVED_KWARGS)
    def test_reserved_keyword_arguments_are_rejected(self, reserved):
        with pytest.raises(Exception, match="are reserved"):
            get_params(_stub_function, **{reserved: 1})


class TestCheckInputs:
    """Unit tests for cop_ard_app.computation.check_inputs."""

    @pytest.mark.unit
    def test_lists_and_tuples_are_copied_into_lists(self):
        objs, objs_ov, args = check_inputs(7, (1, 2), [3, 4])
        assert objs == [7]
        assert objs_ov == [1, 2]
        assert args == [3, 4]

    @pytest.mark.unit
    def test_any_other_object_is_wrapped_in_a_single_element_list(self):
        payload = np.arange(3)
        objs, objs_ov, args = check_inputs(payload, None, "a-string")
        assert objs == [payload]
        assert objs[0] is payload
        assert objs_ov == [None]
        assert args == ["a-string"]


class TestSetup:
    """Unit tests for cop_ard_app.computation.setup."""

    @pytest.mark.unit
    def test_a_list_reports_its_length_and_the_list_constructor(self):
        length, typef, results_l = setup([1, 2])
        assert length == 2
        assert typef is list
        assert results_l == [1, 2]

    @pytest.mark.unit
    def test_a_tuple_reports_its_length_and_the_tuple_constructor(self):
        length, typef, results_l = setup(("a", "b"))
        assert length == 2
        assert typef is tuple
        assert results_l == ["a", "b"]

    @pytest.mark.unit
    def test_a_single_result_is_wrapped_and_unwrapped_by_the_lambda(self):
        length, typef, results_l = setup(42)
        assert length == 1
        assert results_l == [42]
        assert typef([42]) == 42


class TestRetrieveClient:
    """Unit tests for cop_ard_app.computation.retrieve_client (DUMMY and validation paths)."""

    @pytest.mark.unit
    def test_dummy_client_options_short_circuit_to_a_dummy_client(self):
        client, enable_dask, restart, create, broadcast, jobs_chunksize = retrieve_client({"type": "DUMMY"})
        assert isinstance(client, DummyClient)
        assert (enable_dask, restart, create, broadcast, jobs_chunksize) == (False, False, None, False, 1)

    @pytest.mark.unit
    def test_restart_and_broadcast_options_are_reported_back(self):
        options = {"type": "DUMMY", "restart": True, "broadcast": True}
        _, enable_dask, restart, create, broadcast, jobs_chunksize = retrieve_client(options, enable_dask=False)
        assert (enable_dask, restart, create, broadcast, jobs_chunksize) == (False, True, None, True, 1)

    @pytest.mark.unit
    def test_unknown_cluster_type_is_rejected_before_any_instantiation(self):
        with pytest.raises(Exception, match="Invalid cluster type"):
            retrieve_client({"type": "BOGUS"})


class TestClientManager:
    """Unit tests for cop_ard_app.computation.client_manager (non-worker paths)."""

    @pytest.mark.unit
    def test_disabled_dask_hands_out_a_dummy_client(self):
        with client_manager({}, enable_dask=False) as (client, enable_dask, restart, create, broadcast, jobs_chunksize):
            assert isinstance(client, DummyClient)
            assert (enable_dask, restart, create, broadcast, jobs_chunksize) == (False, False, None, False, 1)

    @pytest.mark.unit
    def test_dummy_client_options_avoid_local_cluster_creation(self):
        with client_manager({"type": "DUMMY"}, enable_dask=True) as (client, enable_dask, *_):
            assert isinstance(client, DummyClient)
            assert enable_dask is False


class TestTryToStore:
    """Unit tests for cop_ard_app.computation.try_to_store."""

    @pytest.mark.unit
    def test_results_pass_through_untouched_without_storage_instructions(self):
        assert try_to_store([1, 2, 3], {}) == [1, 2, 3]

    @pytest.mark.unit
    def test_force_compute_materializes_xarray_results(self):
        payload = _column([1, 2, 3], 0)
        (computed,) = try_to_store([payload], {}, force_compute=True)
        assert isinstance(computed, xr.DataArray)
        assert np.array_equal(computed.values, np.asarray([1, 2, 3]))

    @pytest.mark.unit
    def test_store_mapping_indices_must_fit_the_number_of_results(self):
        with pytest.raises(RuntimeError, match="must be lower than the number of returned elements"):
            try_to_store([1, 2, 3], {}, store_mapping={3: {"data_path": "any"}})

    @pytest.mark.unit
    def test_store_mapping_configurations_require_a_data_path(self):
        with pytest.raises(RuntimeError, match="unable to find `data_path`"):
            try_to_store([1], {}, store_mapping={0: {}})

    @pytest.mark.unit
    def test_store_mapping_writes_a_slice_into_an_initialized_store(self, tmp_path):
        store = str(tmp_path / "store")
        full = _column(np.arange(10), 0).rename("q")
        try_to_store([full], {}, store_directory=store, save_options={"consolidated": True})
        results = try_to_store(
            [_column([7, 8], 2).rename("q")],
            {"x": slice(2, 4)},
            store_mapping={0: {"data_path": store, "skip_read": True, "save_options": {"consolidated": True}}},
        )
        assert results == [None]
        with xr.open_zarr(store) as written:
            assert list(written.data_vars) == ["q"]
            assert np.array_equal(written["q"].values, np.asarray([0, 1, 7, 8, 4, 5, 6, 7, 8, 9]))


class TestMapBlocks:
    """Unit tests for cop_ard_app.computation.map_blocks (enable_dask=False)."""

    @pytest.mark.unit
    def test_blocks_follow_the_chunk_product_order(self):
        grid = _grid()
        blocks = map_blocks(
            _pass_through,
            grid,
            chunksizes={"x": X_CHUNKSIZE, "y": Y_CHUNKSIZE},
            enable_dask=False,
            combine=False,
        )
        assert len(blocks) == 4
        expected = [
            grid.values[0:3, 0:2],
            grid.values[0:3, 2:4],
            grid.values[3:6, 0:2],
            grid.values[3:6, 2:4],
        ]
        for block, want in zip(blocks, expected):
            assert isinstance(block, xr.DataArray)
            assert np.array_equal(block.values, want)

    @pytest.mark.unit
    def test_combine_merges_blocks_back_into_a_dataset(self):
        grid = _grid()
        combined = map_blocks(
            _offset_stub,
            grid,
            args=(OFFSET,),
            chunksizes={"x": X_CHUNKSIZE, "y": Y_CHUNKSIZE},
            enable_dask=False,
        )
        assert isinstance(combined, xr.Dataset)
        assert list(combined.data_vars) == ["d"]
        assert np.array_equal(combined["d"].values, grid.values + OFFSET)

    @pytest.mark.unit
    def test_overlap_haloes_are_clipped_at_the_array_bounds(self):
        obj = _column(np.arange(SHAPE[0]), 0)
        sizes = map_blocks(
            _block_sizes,
            obj,
            obj_ov=[obj],
            chunksizes={"x": X_CHUNKSIZE},
            overlap_sizes={"x": 1},
            enable_dask=False,
            combine=False,
        )
        assert sizes == [(3, 4), (3, 4)]


class TestMap:
    """Unit tests for cop_ard_app.computation.map (enable_dask=False)."""

    @pytest.mark.unit
    def test_flat_argument_sequences_are_applied_in_order(self):
        assert map(_plus_one, [1, 2, 3], enable_dask=False) == [2, 3, 4]

    @pytest.mark.unit
    def test_tuple_of_sequences_is_mapped_elementwise(self):
        assert map(_stub_function, ([1, 2], [10, 20]), enable_dask=False) == [1, 2]

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "bad_args, message",
        [
            ("xyz", "args must be a list or a tuple"),
            (([1, 2], [3]), "args elements must have the same length"),
        ],
    )
    def test_inconsistent_arguments_are_rejected(self, bad_args, message):
        with pytest.raises(Exception, match=message):
            map(_stub_function, bad_args, enable_dask=False)

    @pytest.mark.unit
    def test_combine_concatenates_result_columns(self):
        combined = map(_named_column, [1, 2], enable_dask=False, combine=True)
        assert isinstance(combined, xr.DataArray)
        assert np.array_equal(combined.values, np.asarray([10, 20]))
        assert np.array_equal(combined.coords["x"].values, np.asarray([10, 20]))


class TestInitZarrGroups:
    """Unit tests for cop_ard_app.computation.init_zarr_groups."""

    @pytest.mark.unit
    def test_groups_and_measurements_subgroups_are_created(self, tmp_path, monkeypatch):
        monkeypatch.setattr(computation, "map", _map_without_dask)
        out_path = tmp_path / "out"
        init_zarr_groups(str(out_path), list(GROUPS), False)
        for group in GROUPS:
            group_path = out_path / group
            assert group_path.is_dir()
            assert (group_path / "zarr.json").is_file()
            measurements = group_path / "measurements"
            assert measurements.is_dir()
            assert (measurements / "zarr.json").is_file()
