# Copyright 2024 European Union
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import inspect
import itertools
from contextlib import contextmanager
from types import FunctionType
from typing import Any, Callable, Dict, KeysView, List, Literal, Mapping, Sequence, Tuple, Union, ValuesView, cast

import dask.distributed
import numpy as np
import psutil
import retry
import xarray as xr
from cop_ard_core.logger import logger

from cop_ard_app.io import Path, get_storage_options, isdir, read, write


class DummyClient(object):
    """A mock client that mimics a distributed computing client API.

    This class provides a synchronous, in-memory implementation of a client
    interface commonly found in libraries like Dask. It is designed for
    testing or for running workflows in a local, non-parallel environment
    without changing the core application logic.

    All methods execute immediately and do not involve any networking or
    parallel processing.
    """

    def __init__(self):
        """Initializes the DummyClient."""
        self.name = "dummy-client"

    def scatter(self, data, **kwargs):
        """Mock implementation of a submit operation.

        In a real client, this would submit a function for asynchronous
        execution. Here, it executes the function immediately and returns
        a `DummyFuture` that already holds the result.

        Parameters
        ----------
        func : Callable
            The function to execute.
        *args
            Positional arguments to pass to `func`.
        **kwargs
            Keyword arguments to pass to `func`.

        Returns
        -------
        DummyFuture
            A future-like object holding the result of `func(*args, **kwargs)`.
        """
        return data

    def compute(self, results, workers=None):
        """Mock implementation of a compute operation.
        Wraps each item in an iterable into a DummyFuture, simulating the
        asynchronous computation of results.

        Parameters
        ----------
        results : Iterable[Any]
            A collection of results to be "computed."
        workers : Any, optional
            Included for API compatibility. Defaults to None.

        Returns
        -------
        list
            A list of DummyFuture objects, each holding one of the results.
        """
        return [DummyFuture(e) for e in results]

    def cancel(self, *args, **kwargs):
        """Mock implementation of a cancel operation. This is a no-op."""
        pass

    def restart(self, **kwargs):
        """Mock implementation of a client restart. This is a no-op."""
        pass

    def map(self, func, *args, key=None, workers=None, **kwargs):
        """Mock implementation of a map operation.

        Applies a function to a series of arguments synchronously and locally.

        Parameters
        ----------
        func : Callable
            The function to apply to the arguments.
        *args : Iterable
            One or more iterables of arguments to be passed
            to the function.
        workers : Any, optional
            Included for API compatibility.
            Defaults to None.
        **kwargs
            Additional keyword arguments to pass to `func`.

        Returns
        -------
        list
            A list of DummyFuture objects containing the results of applying
            `func` to each set of arguments.
        """
        return [DummyFuture(func(*arg, **kwargs)) for arg in zip(*args)]

    def gather(self, results):
        """Mock implementation of a gather operation.

        Extracts the results from a collection of DummyFuture objects.

        Parameters
        ----------
        results : Iterable[DummyFuture]
            A collection of DummyFuture objects
            from which to retrieve results.

        Returns
        -------
        list
            A list containing the concrete results from the futures.
        """
        return [e.result() for e in results]


class DummyFuture(object):
    """A mock Future object that holds a result directly.

    This class mimics the behavior of a future object from a concurrent
    or distributed library, but all operations are synchronous. The result
    is stored upon creation and returned immediately when requested.
    """

    def __init__(self, obj):
        """Initializes the DummyFuture.

        Parameters
        ----------
        obj : Any
            The result that this future should hold.
        """
        self.name = "dummy-client"
        self._result = obj

    def result(self):
        """Returns the stored result immediately.

        Unlike a real future, this method does not block, as the result
        is already available.

        Returns
        -------
        Any
            The result provided during the initialization of the future.
        """
        return self._result


def conditional_delayed(enable_dask: bool, name: str):
    """Creates a decorator to conditionally apply `dask.delayed`.

    This function acts as a decorator factory. It allows a function to be
    optionally wrapped with `dask.delayed`, controlled by a boolean flag. This
    is useful for writing code that can seamlessly switch between serial
    execution (for debugging or simple cases) and parallel, lazy execution
    with Dask.

    Parameters
    ----------
    enable_dask : bool
        If True, the returned decorator will wrap functions
        with `dask.delayed`. If False, it will return the functions
        unchanged.

    Returns
    -------
    Callable
        A decorator that can be applied to other functions.
    """

    def wrapper(func):
        if enable_dask:
            func.__name__ = name
            return dask.delayed(func)
        else:
            return func

    return wrapper


def get_chunks(
    data: xr.DataArray,
    chunks: Dict[str, Tuple] | None = None,
    chunksizes: Dict[str, int] | None = None,
) -> Tuple[Dict[str, Tuple], Dict[str, Any], int]:
    """
    Calculates an optimal chunking scheme aligned with the data's existing chunks.

    This function takes a desired chunking pattern, specified by either
    explicit boundaries (`chunks`) or by size (`chunksizes`), and aligns it
    with the underlying Dask chunk grid of the input `xr.DataArray`. This
    alignment ensures efficient data access by preventing misaligned reads.

    Notes
    -----
    Exactly one of `chunks` or `chunksizes` must be provided.

    Parameters
    ----------
    data : xarray.DataArray
        The input xarray.DataArray, which must be backed by a
        Dask array with known chunks.
    chunks : dict
        A dictionary defining explicit chunk boundaries for each
        dimension. Mutually exclusive with `chunksizes`.
    chunksizes : dict
        A dictionary defining the desired chunk size for each
        dimension. Dimensions not specified will not be chunked.
        Mutually exclusive with `chunks`.

    Returns
    -------
    aligned_chunks : Dict[str, Tuple]
        A dictionary with the
        calculated chunk boundaries, aligned to the data's grid.
    output_chunksizes : Dict[str, int]
        The effective chunk
        sizes for each dimension corresponding to `aligned_chunks`.
    num_chunks : int
        The total number of chunks.
    """
    _chunks: dict[str, Any] = {}
    _indices: dict[str, Any] = {}
    if chunks is None:
        if chunksizes is None:
            k: Union[str, KeysView[str]]
            v: Union[
                np.typing.NDArray[Any], Tuple[int, ...], List[int], Tuple[Tuple[int, int]], List[Tuple[Tuple[int, int]]]
            ]

            for k, v in data.chunksizes.items():
                v = np.insert(np.cumsum(v), 0, 0)
                if not (isinstance(k, str)):
                    raise Exception(f"Expected str, got {type(k)}")
                _chunks[k], _indices[k] = zip(*tuple(((v[i], v[i + 1]), i) for i in range(len(v) - 1)))
        else:
            if ((set(chunksizes.keys()) - set(cast(Tuple[str], data.dims))) == set()) is False:
                raise Exception(
                    f"chunksizes element list {tuple(chunksizes.keys())} must be a subset of the data\
                    dimension set {data.dims}",
                )
            for k in cast(Tuple[str], data.dims):
                if k in chunksizes:
                    if chunksizes[k] >= data[k].size:
                        v = [0, chunksizes[k]]
                    else:
                        v = list(range(0, data[k].size, chunksizes[k])) + [data[k].size]
                    v, j = zip(*tuple(((v[i], v[i + 1]), i) for i in range(len(v) - 1)))
                else:
                    v, j = ((0, data[k].size),), (0,)
                _chunks[k], _indices[k] = v, j
        v = list(cast(tuple, zip(*itertools.product(*_chunks.values()))))
        _chunks = {k: v for k, v in zip(_chunks.keys(), v)}
        v = list(cast(tuple, zip(*itertools.product(*_indices.values()))))
        _indices = {k: v for k, v in zip(_indices.keys(), v)}
    else:
        k, v = chunks.keys(), cast(List[Tuple[Tuple[int, int]]], list(cast(ValuesView, chunks.values())))
        if (set(k) == set(data.dims)) is False:
            raise Exception("chunks must contain all the data dimensions")
        if (all(len(v[0]) == len(e) for e in v)) is False:
            raise Exception("the lists of values of chunks cannot have a different number of elements")
        if (all(i[0] <= i[1] for j in v for i in j)) is False:
            raise Exception("chunk start value must be lower than or equal to the corresponding end value")
        if (all(np.max(v) <= data[k].size for k, v in chunks.items())) is False:
            raise Exception("all chunks values must be lower than the size of the corresponding data dimension")
        _chunks = dict(chunks)
        _indices = {k: (0,) * len(v) for k, v in _chunks.items()}
    n_chunks = list(_chunks.values())[0]
    n_chunks = len(n_chunks) if isinstance(n_chunks, tuple) else n_chunks
    return _chunks, _indices, n_chunks


def apply_overlap(
    chunks: Dict[str, Tuple],
    overlap_sizes: Mapping[str, Any] = {},
) -> dict:
    """
    Expands chunk boundaries to include a specified overlap.

    This utility function adjusts the start and end coordinates of each chunk
    in a chunking scheme to include data from neighboring chunks. This is
    often required for block processing algorithms where a "halo" or "ghost"
    region is needed for calculations at chunk edges.

    The overlap is applied symmetrically, subtracting from the start and adding
    to the end of each chunk's interval. The start coordinate is clipped at
    zero to prevent negative indices.

    Parameters
    ----------
    chunks : dict
        A dictionary mapping dimension names to an iterable of
        `(start, end)` tuples representing the chunk boundaries along
        that dimension.
    overlap_sizes : dict
        A mapping from dimension names to the integer
        size of the overlap. If a dimension from `chunks` is not present
        in this mapping, an overlap of 0 is assumed for that dimension.

    Returns
    -------
    dict
        A new dictionary with the same keys as `chunks` but with expanded
        `(start, end)` boundaries for each chunk.
    """
    _chunks_ov = {}
    for k in chunks.keys():
        delta = overlap_sizes[k] if k in overlap_sizes else 0
        _chunks_ov[k] = [[s - delta if s > delta else 0, e + delta] for (s, e) in chunks[k]]
    return _chunks_ov


def try_to_combine(
    results: List[xr.DataArray | xr.Dataset | str],
) -> List[xr.DataArray | xr.Dataset] | xr.DataArray | xr.Dataset | None:
    """Applies a function to blocks of xarray objects, with parallel support via Dask.

    This function provides a high-level parallel map operation for large xarray
    datasets. It automates the process of defining a chunking scheme, applying
    a user function to each chunk, handling overlaps (halos), executing tasks
    in parallel, and combining the results back into a single xarray object.

    Notes
    -----
    This function is decorated with a retry mechanism, making it resilient
    to transient failures during execution.

    Parameters
    ----------
    func : callable
        The function to apply to each block of data.
    obj : xarray.DataArray or xarray.Dataset or sequence
        The primary xarray.DataArray, xarray.Dataset, or sequence of them
        to be processed.
    obj_ov : sequence, optional
        A sequence of objects to be processed using chunks with an
        overlap or halo. This is useful for stencil operations where a
        block needs access to neighboring data.
    combine : bool, default True
        If `True`, the function will attempt to combine the results
        from each block into a single xarray object. If `False`, it returns
        a list of the results.
    enable_dask : bool, default True
        If `True`, execute tasks in parallel on a Dask cluster.
        If `False`, tasks are executed sequentially in a loop.
    chunks : dict, optional
        A dictionary defining explicit chunk boundaries for each
        dimension. Mutually exclusive with `chunksizes`.
    chunksizes : dict, optional
        A dictionary defining the desired chunk size for each
        dimension. Mutually exclusive with `chunks`.
    overlap_sizes : dict or mapping, optional
        A dictionary mapping dimension names to the integer size
        of the overlap or halo to apply to the chunks for `obj_ov`.
    fix_memory_leak : bool or str, default False
        A strategy to mitigate potential memory leaks in long
        computations. If `True`, submits jobs in batches and restarts the
        client between batches. If 'auto', this behavior is enabled when
        the number of jobs is very large.
    scatter_strategy : str, optional
        The strategy for scattering data to workers. Can be
        'once-and-for-all' or 'inside-loop'. If `None`, a strategy is
        selected automatically based on data size.
    scheduler : str, default "synchronous"
        The Dask scheduler to use (e.g., 'threads', 'processes', 'synchronous').
    client_options : dict, optional
        Configuration for the Dask client. Key options include:

        - `name` (str): Name of an existing client to connect to.
        - `create` (bool): If `True`, create a new client.
        - `type` (str): Type of client to create (e.g., "LOCAL").
        - `restart` (bool): If `True`, restart the client after execution.
        - `broadcast` (bool): If `True`, scatter data using `client.broadcast`.
        - `n_jobs`, `job_memory`: Parameters for creating a new client.

    compute_on_host : bool, default False
        If `True`, forces computation to run on the host
        machine's scheduler, bypassing Dask workers.
    workers : list of str or str, optional
        A list of Dask worker addresses to restrict this computation to.
    args : sequence, optional
        Positional arguments to pass to `func` after the data chunks.
    kwargs : dict or mapping, optional
        Keyword arguments to pass to `func`.

    Returns
    -------
    xarray.DataArray or xarray.Dataset or list or None

        - A single, combined `xarray.DataArray` or `xarray.Dataset` if `combine`
          is `True` and the combination is successful.
        - A list of result objects if `combine` is `False` or if the
          combination fails.
        - `None` if the input object is empty.
    """
    _results: list[Any] = []
    if len(results) == 0:
        return None

    def try_to_combine_(result):

        try:
            # when func returns paths, I will try  #
            # reading and then concatenating them  #
            if all(isinstance(e, str) for e in result) and all(isdir(e) for e in result):
                _result = [read(e, consolidate_metadata=False, as_type="xr.Dataset") for e in result]
                _result = [e.drop_attrs() for e in _result]
            else:
                raise ValueError("result does not seem to be data file")
        except (FileNotFoundError, ValueError):
            _result = result
        _dims, _ok = [list(e.dims) for e in _result], False
        if (len(_dims[0]) == 1) and all(e == _dims[0] for e in _dims):
            # in such a case there is only one dimension and it is the same for all the #
            # result elements so I can try to use concatenate along this dimension     #
            try:
                _result = xr.concat(_result, dim=_dims[0][0])
                _ok = True
            except Exception as e:
                logger.warning(f"type of exception: {e.__class__.__name__}")
                logger.warning("concat failed results will be returned as a list")
        else:
            try:
                _result = xr.combine_by_coords(_result)
                _ok = True
            except Exception as e:
                logger.warning(f"type of exception: {e.__class__.__name__}")
                logger.warning("combine_by_coords failed results will be returned as a list")
        return _result if _ok else result

    result: Any  # mypy fix
    for result in zip(*results) if all(isinstance(e, (list, tuple)) for e in results) else (results,):
        _s = [getattr(e, "attrs", {}).get("store_path", None) for e in result]
        if (_s[0] is not None) and all(_s[0] == e for e in _s):
            # all the result elements have been saved into the same store_path. In such a case  #
            # case, results have been concatenated yet and I just have to return one of them    #
            result = result[0]
        else:
            if all(isinstance(e, (xr.DataArray, xr.Dataset, str)) for e in result):
                # when func returns a list of xr.DataArray or xr.Dataset I will try to concatenate them    #
                # On the contrary, when the output of the called function is not a xr.DataArray or a       #
                # xr.Dataset or a list of paths I simply return the list of the function output objects    #
                result = try_to_combine_(result)
        _results += (result,)
    return _results if len(_results) > 1 else _results[0]


def retrieve_client(
    client_options: dict = {},
    enable_dask: bool = True,
    **kwargs: dict,
) -> Tuple[Union[dask.distributed.Client, DummyClient], bool, bool, Union[bool, None], bool, int]:
    """Retrieves an existing Dask client or creates a new one based on options.

    This function manages the lifecycle of a distributed computing client. It
    parses a configuration dictionary to decide whether to get an existing Dask
    client, create a new one, or fall back to a local, synchronous `DummyClient`
    for non-parallel execution.

    It returns the client object along with several configuration flags derived
    from the input options, which can be used to control downstream processing.

    Parameters
    ----------
    client_options : dict
        A dictionary of configuration options, which can include:
        - `name` (str, optional): The name of a Dask client to get.
        - `type` (str, optional): The type of client to create if one is not
        found (e.g., "LOCAL", "DUMMY"). Defaults to "LOCAL".
        - `create` (bool, optional): A flag to force the creation of a new
        client, even if one exists.
        - `restart` (bool, optional): A flag indicating if the client
        should be restarted. Defaults to False.
        - `broadcast` (bool, optional): A flag indicating if data should
        be broadcast to workers. Defaults to False.
        - Other options are passed directly to the `create_client` function.
    enable_dask : bool
        A master switch to enable Dask. This will be automatically
        set to `False` if the `client_options['type']` is "DUMMY".
        Defaults to True.
    **kwargs
        Catches unused keyword arguments.

    Returns
    -------
    tuple
        A tuple containing six items:
        1.  `client`: The retrieved or created client object.
        2.  `enable_dask`: The effective boolean status of whether Dask is used.
        3.  `restart`: The parsed `restart` option.
        4.  `create`: The parsed `create` option.
        5.  `broadcast`: The parsed `broadcast` option.
        6.  `jobs_chunksize`: A suggested chunk size for processing, which is
        large for Dask (20,000) and 1 for the dummy client.
    """
    c_opt = copy.deepcopy(client_options)
    # client options retrieve #
    name = c_opt.pop("name", None)
    c_type = c_opt.pop("type", "LOCAL")
    create = c_opt.pop("create", None)
    restart = c_opt.pop("restart", False)
    broadcast = c_opt.pop("broadcast", False)
    enable_dask &= c_type != "DUMMY"

    if enable_dask:
        jobs_chunksize = 20_000
        try:
            # client retrieve if dask is enabled #
            client: Union[DummyClient, dask.distributed.client.Client] = dask.distributed.get_client(name)
        except (ValueError, OSError):
            create = True

        if create:
            n_jobs, job_memory = [c_opt.pop(e, None) for e in ["n_jobs", "job_memory"]]
            client = create_client(c_type, n_jobs, job_memory, **c_opt)
    else:
        jobs_chunksize = 1
        client = create_client("DUMMY")

    return client, enable_dask, restart, create, broadcast, jobs_chunksize


def select_scatter_strategy(
    objs: Sequence[xr.DataArray | xr.Dataset],
    scatter_strategy: str | None = None,
    memory_threshold: float = 512,
) -> Tuple[Union[None, str], bool]:
    """Selects an optimal Dask scattering strategy based on coordinate memory size.

    This function determines whether it's more efficient to scatter (i.e., send)
    a list of xarray objects to Dask workers all at once before computation
    begins ("once-and-for-all"), or to send them as needed within the task
    graph ("inside-loop").

    The automatic decision is based on the memory size of the **coordinate arrays**
    associated with the data, as these are often broadcast to many workers. If
    all coordinate arrays are smaller than the `memory_threshold`, scattering
    upfront is chosen. Otherwise, the more memory-conservative "inside-loop"
    strategy is selected.

    Parameters
    ----------
    objs : sequence
        A sequence of xarray.DataArray or xarray.Dataset objects
        that will be scattered to the cluster.
    scatter_strategy : str, optional
        If specified (e.g., "once-and-for-all" or
        "inside-loop"), this forces a specific strategy, and automatic
        selection is skipped. Defaults to None.
    memory_threshold : int or float, optional
        The maximum size in megabytes (MB) a single
        coordinate array can be for the "once-and-for-all" strategy
        to be chosen. Defaults to 512.

    Returns
    -------
    str
        The name of the selected strategy as a string.
    bool
        A boolean flag which is `True` if the strategy is
        "once-and-for-all", and `False` otherwise.
    """
    if scatter_strategy is None:
        to_m = 1024.0 * 1024.0
        v = [
            ((obj[dim].dtype.itemsize * obj[dim].size) / to_m) <= memory_threshold
            for obj in objs
            for dim in obj.dims
            if dim in obj.coords
        ]
        return ("once-and-for-all", True) if all(v) else ("inside-loop", False)
    else:
        return (scatter_strategy, scatter_strategy == "once-and-for-all")


@contextmanager
def client_manager(client_options: dict, enable_dask: bool = True):
    """Provides a computing client within a context, automatically handling worker-side execution.

    This context manager is the recommended way to obtain a client for computations.
    It intelligently detects if the code is running on a Dask worker. If so, it
    efficiently provides a `worker_client`. If not, or if Dask is disabled,
    it falls back to the more general `retrieve_client` logic to get or create
    an appropriate client.

    This allows functions to be written flexibly, working correctly whether they
    are executed on the client-side or as part of a Dask task graph on a worker.

    Parameters
    ----------
    client_options : dict
        A dictionary of configuration options. This is passed
        directly to `retrieve_client` only when the code is not running
        on a Dask worker.
    enable_dask : bool
        A flag to enable or disable Dask. If `False`, a
        `DummyClient` is always provided.

    Yields
    ------
    tuple
        A tuple containing six items consistent with `retrieve_client`'s output:
        1.  `client`: The client object (`dask.distributed.Client` or `DummyClient`).
        2.  `enable_dask`: The effective boolean status of whether Dask is used.
        3.  `restart`: A configuration flag.
        4.  `create`: A configuration flag.
        5.  `broadcast`: A configuration flag.
        6.  `jobs_chunksize`: A suggested chunk size for processing.
    """
    if not enable_dask:
        yield retrieve_client({"type": "DUMMY"}, enable_dask=False)
        return

    # 1. Safely check if we are on a worker
    try:
        dask.distributed.get_worker()
        is_worker = True
    except ValueError:
        is_worker = False

    if not is_worker:
        yield retrieve_client(client_options, enable_dask=True)
        return

    # --- 2. OPTIMISTIC EXECUTION (Zero Delay) ---
    yielded = False

    try:
        # Standard context block. Safe for the Dashboard Event Loop.
        with dask.distributed.worker_client() as client:
            yielded = True
            yield (client, True, False, False, False, 20_000)

    except Exception as e:
        if yielded:
            # The client connected perfectly, but your actual map/map_blocks
            # computation crashed.
            raise e

        # If yielded is STILL False, the error happened during worker_client setup
        # (the race condition). We swallow the error and prepare to retry.
        pass

    # --- 3. THE FALLBACK (Jitter Delay) ---
    if not yielded:
        # The optimistic try failed. Add a tiny random delay, then try again.

        with dask.distributed.worker_client() as client:
            yield (client, True, False, False, False, 20_000)


def get_params(func: Callable[..., Any], **kwargs):
    """Validates keyword arguments against a reserved list and inspects a function's signature.

    This utility function prepares arguments before calling a user-provided
    function. It ensures that the provided keyword arguments do not conflict
    with a set of internally reserved keywords used by the processing framework.

    Parameters
    ----------
    func
        The target function whose signature will be inspected.
    **kwargs
        The keyword arguments to validate against the reserved list.

    Returns
    -------
    tuple
        A tuple containing:
            - A mapping of the target function's parameter names to their
              `inspect.Parameter` objects.
            - The original `kwargs` dictionary, confirmed to be free of
              any reserved keys.

    Raises
    ------
    Exception
        If any of the keys in `kwargs` is a reserved keyword.
    """
    RESERVED_KWARGS = ("job_id", "chunk", "block_id", "chunk_ov")
    if (set(kwargs.keys()) == (set(kwargs.keys()) - set(RESERVED_KWARGS))) is False:
        raise Exception(f"{RESERVED_KWARGS} are reserved. Got {kwargs}")

    func_parameters = inspect.signature(func).parameters
    kwargs = {k: v for k, v in kwargs.items() if k not in RESERVED_KWARGS}
    return func_parameters, kwargs


def check_inputs(obj, obj_ov, args):
    """Ensures that a set of inputs are all returned as lists.

    This is a utility function for input normalization. It takes three
    arguments and guarantees that each is returned as a list, regardless
    of whether it was originally a single item, a list, or a tuple.
    This simplifies functions that need to iterate over inputs by allowing
    them to handle single items and collections uniformly.

    Parameters
    ----------
    obj : object or list or tuple
        An object or a list/tuple of objects.
    obj_ov : object or list or tuple
        A second object or a list/tuple of objects.
    args : object or list or tuple
        A third object or a list/tuple of objects.

    Returns
    -------
    tuple
        A tuple containing the three normalized inputs, each as a list.
    """
    _objs = [obj] if not isinstance(obj, (list, tuple)) else list(obj)
    _objs_ov = [obj_ov] if not isinstance(obj_ov, (list, tuple)) else list(obj_ov)
    _args = [args] if not isinstance(args, (list, tuple)) else list(args)
    return _objs, _objs_ov, _args


def setup(results: list | tuple | Any) -> tuple[int, Callable, list[Any]]:
    """

    Return the number of elements of results the typef function and results as a list

    """

    typef: list | tuple | Callable
    if isinstance(results, list):
        results_l = results
        typef = list
    elif isinstance(results, tuple):
        results_l = list(results)
        typef = tuple
    else:
        results_l, typef = [results], lambda x: list(x)[0]
    return len(results_l), typef, results_l


def try_to_store(
    results: list[Any],
    chunk: Mapping[str, slice],
    store_mapping: Mapping[int, dict[str, Any]] = {},
    store_directory: str | Path | None = None,
    force_compute: bool | None = None,
    region_callback: Mapping[int, Callable[[Mapping[str, slice], dict[str, Any]], Mapping[str, slice] | None]] = {},
    **kwargs,
) -> Any:
    """

    This function try to save the results (that can be a list or a single element) using pstools.io

    Parameters
    ----------
    chunk: dict
        It is the chunk of data that has been passed to the executed function to compute the results
    store_directory: pstools.path.Path or None
        It is the directory where results will be saved if store_mapping is not provided. If None is provided, results will not be saved
    force_compute: bool | None
        If True, the results will be computed even if they are already available
    store_mapping: dict
        It describes where and how each element of results should be saved. It must have the following structure:
            store_mapping = {
                id_1: {
                        "data_path": "path_1",
                        "dims": ["dim_0", "dim_1", "dim_2", ..., "dim_k"],
                        "drop_coords": True,
                        "drop_attrs": True,
                        "save_options": {"skip_read": True, ...},
                    },
                ...
                id_n: {
                        "data_path": "path_n",
                        "dims": ["dim_0", "dim_1", ..., "dim_m"],
                        "drop_coords": True,
                        "drop_attrs": True,
                        "save_options": {"skip_read": True, ...}
                    },
            }
        where id_j is the index of the element of results to be saved, path_j is the path where that element
        should be saved (this file has to be previously initialized). The `dims` key (optional) can be used to identify
        the subset of dimensions of chunk to create the region to pass to pstools.io.write (if not provided all the
        dimensions of chunk will be used). `drop_coords` (optional) can be used to drop coordinates from xarray objects
        just before saving it. The `save_options` key (optional) can be used to pass additional options to
        pstools.io.write when saving the j-th element of results (if not provided the default options will be used)

    """
    # Options to be used to save (i.e to be passed to pstools.io.write) xarray objects
    save_options: dict[str, Any] = {"consolidated": False, **kwargs.get("save_options", {})}

    def extract_store_info(
        j: int, config: Mapping[str, Any]
    ) -> tuple[str, Mapping[str, slice] | None, bool, bool, Mapping[str, Any]]:
        """Return the information inside options used to save results from executed function"""

        try:
            data_path = config["data_path"]
        except KeyError:
            raise RuntimeError(f"Provided invalid configuration for element {j}: unable to find `data_path` ({config})")

        # Region to be used with pstools.io.write
        region = {k: v for k, v in chunk.items() if k in config.get("dims", chunk.keys())}
        if j in region_callback:
            callback = region_callback[j]
            region = callback(region, kwargs)
            if region is not None:
                region = dict(region)

        return (
            data_path,
            region,
            config.get("drop_coords", True),
            config.get("drop_attrs", True),
            {**config.get("save_options", save_options), "skip_read": config.get("skip_read", True)},
        )

    n_results, typef, results_l = setup(results)

    if store_mapping:
        # If store_mapping exists it will be used and cache_directory if provided will be ignored
        logger.debug(f"Try to save results ({chunk}) using as configurations ({store_mapping})", tags=["store"])

        # Indices provided must be lower than the number of elements returned by executed function
        if max(list(store_mapping.keys())) >= n_results:
            raise RuntimeError(
                f"Invalid `store_mapping`: maximum index ({max(list(store_mapping.keys()))}) "
                f"must be lower than the number of returned elements ({n_results})"
            )

        # The dimensions names must be all keys of chunk
        _, regions, _, _, _ = zip(*[extract_store_info(k, v) for k, v in store_mapping.items()])
        dims = set([k for j in regions if j is not None for k in j.keys()])
        if any(j not in chunk for j in dims):
            raise RuntimeError(f"Invalid `store_mapping`: all dimensions ({dims}) must be a key of chunk ({chunk})")

        # If I am here it means store_mapping seems to be good so I try to save following that instructions
        for j in store_mapping.keys():
            # Result to be saved and required information (i.e path, region and options) extraction
            j_result, (j_data_path, j_region, j_drop_coords, j_drop_attrs, j_opts) = (
                results_l[j],
                extract_store_info(j, store_mapping[j]),
            )

            if isinstance(j_result, (xr.Dataset, xr.DataArray, xr.DataTree)):
                # When the result is saved results_l[j] will be replaced by the path to the file into the data has been saved
                # This replacement allows try_to_combine (that will be execute in the next steps) to do the right thing
                if j_drop_coords:
                    j_result = j_result.reset_coords(drop=True)
                if j_drop_attrs:
                    j_result = j_result.drop_attrs()
                logger.debug(
                    f"Saving {j}-th element ({j_result}) of results into {j_data_path} using as region {j_region}",
                    tags=["store"],
                    verbose=True,
                )
                results_l[j] = write(j_result, data_path=j_data_path, region=j_region, **j_opts)
            else:
                logger.warning(
                    f"The {j}-th element of results cannot be saved: only xr.Dataset, xr.DataArray or xr.DataTree can be saved by computation"
                )
        results = typef(results_l)
    elif store_directory is not None:
        logger.debug(
            f"Try to save results into cache directory ({store_directory}) using the following options: {save_options}",
            tags=["store"],
        )

        logger.debug(f"Results to be saved: {results_l}", verbose=True, tags=["store"])
        results = typef(
            (
                write(j, data_path=store_directory, **{**save_options, "skip_read": False})  # type: ignore
                if isinstance(j, (xr.Dataset, xr.DataArray, xr.DataTree))
                and all(s > 0 for s in getattr(j, "sizes", {}).values())
                else j
            )
            for j in results_l
        )
    elif force_compute:
        results = typef(
            [
                (
                    e.compute()
                    if isinstance(e, (xr.Dataset, xr.DataArray, xr.DataTree))
                    else e
                )
                for e in results_l
            ]
        )

    return results


@retry.retry(tries=5, delay=1, jitter=5, logger=logger)
def map_blocks(
    func: FunctionType,
    obj: Sequence[xr.DataArray | xr.Dataset] | xr.Dataset | xr.DataArray,
    obj_ov: Sequence[xr.DataArray] = [],
    combine: bool = True,
    enable_dask: bool = True,
    force_compute: bool = False,
    chunks: dict | None = None,
    chunksizes: dict[str, Any] | None = None,
    overlap_sizes: Mapping[str, Any] = {},
    fix_memory_leak: bool | Literal["auto"] = False,  # n_replicate: int = 0,
    scatter_strategy: str | None = None,
    scheduler: str | None = "synchronous",
    client_options: dict[str, Any] = {},
    compute_on_host: bool = False,
    workers: list[str] | str | None = None,
    store_directory: str | None = None,
    store_mapping: Mapping[int, Any] = {},
    region_callback: Mapping[int, Callable[[Mapping[str, slice], dict[str, Any]], Mapping[str, slice] | None]] = {},
    args: Sequence[Any] = (),
    kwargs: Mapping[str, Any] = {},
) -> List[Any] | xr.DataArray | xr.Dataset | None:
    """Applies a function to blocks of xarray objects, with parallel support via Dask.

    This function provides a high-level parallel map operation for large xarray
    datasets. It automates the process of defining a chunking scheme, applying
    a user function to each chunk, handling overlaps (halos), executing tasks
    in parallel, and combining the results back into a single xarray object.

    Notes
    -----
    This function is decorated with a retry mechanism, making it resilient
    to transient failures during execution.

    Parameters
    ----------
    func : callable
        The function to apply to each block of data.
    obj : xarray.DataArray or xarray.Dataset or sequence
        The primary xarray.DataArray, xarray.Dataset, or sequence of them
        to be processed.
    obj_ov : sequence, optional
        A sequence of objects to be processed using chunks with an
        overlap or halo. This is useful for stencil operations where a
        block needs access to neighboring data.
    combine : bool, default True
        If `True`, the function will attempt to combine the results
        from each block into a single xarray object. If `False`, it returns
        a list of the results.
    enable_dask : bool, default True
        If `True`, execute tasks in parallel on a Dask cluster.
        If `False`, tasks are executed sequentially in a loop.
    force_compute : bool
        If `True`, forces computation of the results.
    chunks : dict, optional
        A dictionary defining explicit chunk boundaries for each
        dimension. Mutually exclusive with `chunksizes`.
    chunksizes : dict, optional
        A dictionary defining the desired chunk size for each
        dimension. Mutually exclusive with `chunks`.
    overlap_sizes : dict or mapping, optional
        A dictionary mapping dimension names to the integer size
        of the overlap or halo to apply to the chunks for `obj_ov`.
    fix_memory_leak : bool or str, default False
        A strategy to mitigate potential memory leaks in long
        computations. If `True`, submits jobs in batches and restarts the
        client between batches. If 'auto', this behavior is enabled when
        the number of jobs is very large.
    scatter_strategy : str, optional
        The strategy for scattering data to workers. Can be
        'once-and-for-all' or 'inside-loop'. If `None`, a strategy is
        selected automatically based on data size.
    scheduler : str, default "synchronous"
        The Dask scheduler to use (e.g., 'threads', 'processes', 'synchronous').
    client_options : dict, optional
        Configuration for the Dask client. Key options include:

        - `name` (str): Name of an existing client to connect to.
        - `create` (bool): If `True`, create a new client.
        - `type` (str): Type of client to create (e.g., "LOCAL").
        - `restart` (bool): If `True`, restart the client after execution.
        - `broadcast` (bool): If `True`, scatter data using `client.broadcast`.
        - `n_jobs`, `job_memory`: Parameters for creating a new client.

    compute_on_host : bool, default False
        If `True`, forces computation to run on the host
        machine's scheduler, bypassing Dask workers.
    workers : list of str or str, optional
        A list of Dask worker addresses to restrict this computation to.
    args : sequence, optional
        Positional arguments to pass to `func` after the data chunks.
    kwargs : dict or mapping, optional
        Keyword arguments to pass to `func`.

    Returns
    -------
    xarray.DataArray or xarray.Dataset or list or None
        - A single, combined `xarray.DataArray` or `xarray.Dataset` if `combine`
          is `True` and the combination is successful.
        - A list of result objects if `combine` is `False` or if the
          combination fails.
        - `None` if the input object is empty.
    """
    with client_manager(client_options, enable_dask) as (
        client,
        enable_dask,
        restart,
        destroy,
        broadcast,
        jobs_chunksize,
    ):
        if compute_on_host and not isinstance(client, DummyClient):
            workers = get_worker_host()
        func_parameters, kwargs_ = get_params(func, **kwargs)
        _objs, _objs_ov, _args = check_inputs(obj, obj_ov, args)
        i_ov = len(_objs) + 1

        # optimal scattering strategy selection #
        scatter_strategy, apply_isel = select_scatter_strategy((_objs + _objs_ov), scatter_strategy)

        @conditional_delayed(enable_dask, func.__name__)
        def execute(objs, args, indices, block_id, job_id, apply_isel=apply_isel):
            # reserved keywords set #
            if "chunk" in func_parameters:
                kwargs_["chunk"] = indices[0]
            if "chunk_ov" in func_parameters:
                kwargs_["chunk_ov"] = indices[i_ov]
            if "job_id" in func_parameters:
                kwargs_["job_id"] = job_id
            if "block_id" in func_parameters:
                kwargs_["block_id"] = block_id
            # processing execution #
            if apply_isel is True:
                objs = [obj.isel(index) for (obj, index) in zip(objs, indices)]
            config = {"scheduler": scheduler} if scheduler is not None else {}
            with dask.config.set(config):
                result = func(*objs, *args, **kwargs_)

                return try_to_store(
                    result,
                    indices[0],
                    store_mapping=store_mapping,
                    store_directory=store_directory,
                    force_compute=force_compute,
                    region_callback=region_callback,
                    **{k: v for k, v in kwargs_.items() if k != "chunk"},
                )

        _chunks, _block_ids, _n_chunks = get_chunks(_objs[0], chunks, chunksizes)

        # when requested, the overlapped chunks will be computed #
        _chunks_ov = apply_overlap(_chunks, overlap_sizes)

        # dask bug: a dict cannot be scattered #
        _future_args = [_arg if isinstance(_arg, dict) else client.scatter(_arg, broadcast=broadcast) for _arg in _args]
        if scatter_strategy == "no-scatter":
            _future_objs = _objs + _objs_ov
        elif scatter_strategy == "once-and-for-all":
            _future_objs = [client.scatter(_obj, broadcast=broadcast) for _obj in (_objs + _objs_ov)]

        results: list[Any] = []
        job_array: list[Any] = []

        def compute_results(fix_memory_leak, enable_dask, restart, destroy, job_array):
            _future_args = []
            _future_objs = []
            results_ = client.compute(job_array)
            return results_, _future_args, _future_objs

        for j in range(_n_chunks):
            _block_id = {k: v[j] for k, v in _block_ids.items()}
            _chunk = {k: slice(v[j][0], v[j][1], None) for k, v in _chunks.items()}
            _chunk_ov = {k: slice(v[j][0], v[j][1], None) for k, v in _chunks_ov.items()}

            # chunk to slice indices #
            _i = [{k: v for k, v in _chunk.items() if k in _obj.dims} for _obj in _objs]
            _i += [{k: v for k, v in _chunk_ov.items() if k in _obj.dims} for _obj in _objs_ov]

            if scatter_strategy == "inside-loop":
                _objs_isel = [_obj.isel(index) for _obj, index in zip((_objs + _objs_ov), _i)]
                _future_objs = [client.scatter(_obj, broadcast=broadcast) for _obj in _objs_isel]
            job_array += [execute(_future_objs, _future_args, _i, _block_id, j)]
            if (j + 1) % jobs_chunksize == 0:
                r, fa, fo = compute_results(fix_memory_leak, enable_dask, restart, destroy, job_array)
                job_array = []
                results = results + r
                if fa:
                    _future_args = fa
                if fo:
                    _future_objs = fo
        if job_array:
            results += client.compute(job_array, workers=workers)
        results = [e.result() for e in results]

        def stop_client(destroy, restart):
            if destroy:
                destroy_client(client)
            elif restart:
                client.restart()
            else:
                [client.cancel(e) for e in _future_args + _future_objs]

        stop_client(destroy, restart)
    return try_to_combine(results) if combine else results


def map(
    func: FunctionType,
    args: Union[Sequence[List | Tuple], Tuple, List] = (),
    enable_dask: bool = True,
    combine: bool = False,
    force_compute: bool = False,
    scheduler: str | None = "synchronous",
    client_options: dict = {},
    map_equally: bool = False,
    compute_on_host: bool = False,
    workers: list[str] | str | None = None,
    kwargs: Mapping[str, Any] = {},
    Plugin: dask.distributed.WorkerPlugin | None = None,
    store_directory: Path | None = None,
    save_options: Mapping[str, Any] = {},
) -> List[Any] | xr.DataArray | xr.Dataset | None:
    """
    Applies a function to sequences of arguments, with parallel execution via Dask.

    This function provides a parallel implementation of the map pattern. It
    applies a given function to corresponding elements from one or more
    argument sequences. The execution can be distributed across a Dask cluster
    for performance or run locally for simplicity and debugging.

    Parameters
    ----------
    func : callable
        The function to be applied to the arguments.
    args : sequence or list of sequences
        A sequence or a list of sequences containing the arguments for `func`.
        - For a function with one argument, `func(x)`: `([x1, x2, ...])`
        - For a function with multiple arguments, `func(x, y)`: `([x1, x2, ...], [y1, y2, ...])`
        All argument sequences must have the same length.
    enable_dask : bool
        If `True`, execute tasks in parallel on a Dask cluster.
        If `False`, tasks are executed sequentially.
    combine : bool
        If `True`, the function will attempt to combine the list of
        results into a single xarray object.
    force_compute : bool
        If `True`, forces computation of the results.
    scheduler : str
        The Dask scheduler to use (e.g., 'threads', 'processes').
    client_options : dict
        Configuration for the Dask client. Key options include:
        - `name` (str): Name of an existing client to connect to.
        - `create` (bool): If `True`, create a new client.
        - `type` (str): Type of client to create (e.g., "LOCAL").
    map_equally : bool
        If `True`, attempts to distribute the arguments as evenly
        as possible across the available Dask workers for load balancing.
    compute_on_host : bool
        If `True`, forces tasks to run on the same worker as
        the main client process, which can be useful for data locality.
    workers : str or list
        A specific worker address or list of addresses to run tasks on.
    kwargs : dict
        A dictionary of keyword arguments to be passed to every
        call of `func`.

    Returns
    -------
    list or xarray.Dataset or xarray.DataArray
        A list containing the result of each function call. If `combine` is
        `True` and the results are compatible xarray objects, they will be
        combined into a single `xarray.Dataset` or `xarray.DataArray`.

    Raises
    ------
    Exception
        If the `args` parameter is not a list/tuple or if the
        inner argument sequences have inconsistent lengths.
    """
    # Ensure the main 'args' container is a list or tuple.
    if (isinstance(args, (list, tuple))) is False:
        raise Exception("args must be a list or a tuple")

    # Normalize 'args' to be a sequence of sequences. If it's a single flat list
    # (e.g., for a function with one argument), wrap it in a tuple.
    _args = (args,) if not isinstance(args[0], (list, tuple, np.ndarray)) else args

    # Validate that all argument sequences have the same length for consistent mapping.
    if (all(len(_args[0]) == len(e) for e in _args)) is False:
        raise Exception("args elements must have the same length")

    # Define a wrapper function to ensure each execution uses the specified Dask scheduler.
    def execute_map(*args, **kwargs):
        config = {"scheduler": scheduler} if scheduler is not None else {}
        with dask.config.set(config):
            result = func(*args, **kwargs)
            return try_to_store(result, {}, save_options=save_options, store_directory=store_directory, force_compute=force_compute)

    # Use a context manager to retrieve/create a client (Dask or Dummy)
    # and handle its lifecycle, along with other configuration flags.
    with client_manager(client_options, enable_dask) as (client, _, restart, destroy, _, _):
        if Plugin and not isinstance(client, DummyClient):
            _ = client.register_plugin(Plugin)
        # If 'map_equally' is enabled, distribute arguments evenly across worker hosts.
        if map_equally and not isinstance(client, DummyClient):
            host_ips = get_host_ips(client=client)
            workers = map_args_to_workers(args=args, host_ips=host_ips)

        # If 'compute_on_host' is enabled, restrict tasks to the worker on the same host as the client.
        if compute_on_host and not isinstance(client, DummyClient):
            workers = get_worker_host(client=client)

        future_args: tuple[Any, ...]
        scattered_kwargs: dict[str, Any] = {}

        # 1. Scatter heavy argument sequences
        _scattered_args = []
        for idx, arg_seq in enumerate(_args):
            if len(arg_seq) > 0 and isinstance(arg_seq[0], (xr.DataTree, xr.Dataset, xr.DataArray)):
                _scattered_args.append(client.scatter(arg_seq))
            else:
                _scattered_args.append(arg_seq)
        future_args = tuple(_scattered_args)

        # 2. Scatter heavy kwargs
        for k, v in kwargs.items():
            if isinstance(v, (xr.DataTree, xr.Dataset, xr.DataArray)):
                # Wrap the single heavy object in a list, scatter it, and extract the Future
                scattered_kwargs[k] = client.scatter([v])[0]
            else:
                # Keep lightweight kwargs as they are
                scattered_kwargs[k] = v

        # Submit the 'execute' function to be mapped over the arguments using the client.
        # The '*' unpacks the sequences in 'future_args'. 'gather' blocks and collects the results.
        results = client.map(
            execute_map,
            *future_args,
            **scattered_kwargs,
            workers=workers,
            key=f"{func.__name__}",
        )
        results = client.gather(results)

        # Handle client teardown based on the 'destroy' or 'restart' flags.
        if destroy:
            destroy_client(client)
        elif restart:
            client.restart()

    # If 'combine' is True, attempt to merge the list of results into a single object;
    # otherwise, return the list of results as is.
    return try_to_combine(results) if combine else results


def create_cluster(
    type: Literal["LOCAL", "DUMMY"] = "LOCAL",
    n_jobs: int | None = None,
    job_memory: Union[str, int] | None = None,
    **kwargs: Any,
):
    """Creates a Dask LocalCluster or a dummy placeholder for serial execution.

    This function is a factory for setting up a local computation environment.
    It can either configure and start a `dask.distributed.LocalCluster` for
    parallel processing on the local machine, or return a "DUMMY" placeholder
    for workflows where Dask is disabled.

    It intelligently calculates default values for the number of workers (`n_jobs`)
    and memory per worker based on the system's hardware if they are not
    explicitly provided.

    Parameters
    ----------
    type : str
        The type of cluster to create.
        - "LOCAL": Creates a `dask.distributed.LocalCluster`.
        - "DUMMY": Returns a placeholder string for non-parallel execution.
    n_jobs : int or None
        The number of Dask workers to start. If `None`, defaults to the
        number of logical CPU cores on the machine.
    job_memory : str or int or None
        The memory limit per worker (e.g., "4GB" or an integer in
        bytes). If `None`, it is automatically calculated by distributing
        95% of the total system RAM across the workers.
    **kwargs
        Additional keyword arguments to be passed directly to the
        `dask.distributed.LocalCluster` constructor (e.g., `processes=False`).

    Returns
    -------
    dask.distributed.LocalCluster or str
        The created Dask `LocalCluster` object if `type` is "LOCAL",
        or the string "DUMMY" if `type` is "DUMMY".

    Raises
    ------
    Exception
        If an invalid cluster `type` is provided.
    """
    # Ensure the requested cluster type is valid.
    if (type in ["LOCAL", "DUMMY"]) is False:
        raise Exception("Invalid cluster type. Allowed types are LOCAL and DUMMY")

    # If the number of jobs isn't specified, default to the number of logical CPU cores.
    if n_jobs is None:
        n_jobs = psutil.cpu_count()

    # If job memory isn't specified, calculate a sensible default.
    if job_memory is None:
        if n_jobs is None:
            raise Exception("n_jobs is None, expected int")
        # Distribute 95% of total system RAM evenly among the jobs.
        job_memory = int(psutil.virtual_memory().total / n_jobs * 0.95)
        # If using threads instead of processes (processes=False), the memory
        # limit applies to the single parent process, so allocate the total
        # calculated memory to it.
        if kwargs.get("processes", True) is False:
            job_memory *= n_jobs

    # Log the final cluster configuration for debugging and record-keeping.
    logger.info(f"<Cluster info: type={type}, n_jobs={n_jobs}, job_memory={job_memory}>")

    # For a 'LOCAL' cluster, instantiate a Dask LocalCluster with the specified configuration.
    if type == "LOCAL":
        cluster: Union[str, dask.distributed.LocalCluster] = dask.distributed.LocalCluster(
            n_workers=n_jobs,
            memory_limit=job_memory,
            processes=kwargs.pop("processes", True),
            threads_per_worker=kwargs.pop("threads_per_worker", 1),
            **kwargs,
        )
    # For a 'DUMMY' cluster, simply return a placeholder string.
    elif type == "DUMMY":
        cluster = "DUMMY"
    return cluster


def create_client(
    type: Literal["LOCAL", "DUMMY"] = "LOCAL",
    n_jobs: int | None = None,
    job_memory: str | int | None = None,
    **kwargs: dict,
) -> Union[dask.distributed.client.Client, DummyClient]:
    """Creates a Dask client connected to a new cluster or a dummy client.

    This function is a high-level factory that wraps the `create_cluster`
    utility. It handles the final step of instantiating the client object
    (`dask.distributed.Client` or `DummyClient`) that will be used to submit tasks.

    Parameters
    ----------
    type : str
        The type of client to create.
        - "LOCAL": Creates a client connected to a new `dask.distributed.LocalCluster`.
        - "DUMMY": Creates a `DummyClient` for serial, non-parallel execution.
    n_jobs : int
        The number of Dask workers to start. Passed to `create_cluster`.
    job_memory : str or int
        The memory limit per worker. Passed to `create_cluster`.
    **kwargs
        Additional keyword arguments passed to `create_cluster`.

    Returns
    -------
    dask.distributed.Client or DummyClient
        An instantiated `dask.distributed.Client` or `DummyClient` object.
    """
    # First, create the underlying cluster (either a Dask cluster object or a 'DUMMY' placeholder).
    cluster = create_cluster(type, n_jobs, job_memory, **kwargs)

    # If the cluster is a 'DUMMY' placeholder, return a non-parallel DummyClient.
    # Otherwise, connect a Dask client to the newly created cluster.
    return DummyClient() if cluster == "DUMMY" else dask.distributed.Client(cluster)


def destroy_client(
    client: Union[dask.distributed.client.Client, DummyClient],
):
    """Safely closes a Dask client and its associated cluster.

    This utility handles the teardown process for both real Dask clients and
    dummy clients. For Dask clients, it attempts a graceful shutdown of the
    client and the underlying cluster with a timeout to prevent hanging.

    Parameters
    ----------
    client : object
        The Dask or Dummy client object to be destroyed.
    """
    # Check if the provided client is a real Dask client with resources to release.
    if isinstance(client, dask.distributed.client.Client):
        try:
            # Attempt to gracefully close both the cluster and the client connection.
            client.cluster.close(timeout=30)
            client.close(timeout=30)
        except dask.distributed.TimeoutError:
            # If shutdown takes too long, log a warning instead of crashing.
            logger.warning("Unable to gracefully destroy the client")
    # DummyClients have no external resources, so no action is needed.
    elif isinstance(client, DummyClient):
        pass
    # Remove the client variable from the local scope.
    del client


def get_host_ips(client: dask.distributed.client.Client) -> dict[str, list[str]]:
    """Groups Dask worker addresses by their host IP.

    This function queries the Dask scheduler to get a list of all workers and
    organizes them into a dictionary. Each key in the dictionary is a unique
    host machine IP, and the value is a list of all worker addresses
    (e.g., 'tcp://192.168.1.10:12345') running on that host.

    Parameters
    ----------
    client : dask.distributed.Client
        An active Dask client connected to a scheduler.

    Returns
    -------
    dict
        A dictionary mapping host IPs (str) to lists of worker addresses (str).
    """
    # Get detailed information about all workers from the scheduler.
    workers_info = client.scheduler_info(n_workers=-1)["workers"]
    host_ips: dict[Any, Any] = {}
    # Iterate through each worker's address (w_ip) and its information dict (w_if).
    for w_ip, w_if in workers_info.items():
        # Extract the host IP address for the current worker.
        host = w_if["host"]
        # Group the worker addresses by their host IP.
        if host not in host_ips.keys():
            host_ips[host] = [w_ip]
        else:
            host_ips[host].append(w_ip)
    return host_ips


def map_args_to_workers(args: Sequence, host_ips: dict[str, list[str]]) -> list[str]:
    """Assigns a target worker to each argument set in a round-robin fashion across hosts.

    This function implements a load-balancing strategy to distribute tasks.
    It ensures that work is spread as evenly as possible first across different
    physical machines (hosts), and then across the multiple workers that may be
    running on each machine.

    Parameters
    ----------
    args : sequence
        The sequences of arguments that define the number of jobs to dispatch.
    host_ips : dict
        The dictionary mapping hosts to workers from `get_host_ips`.

    Returns
    -------
    list
        A list of worker addresses, with one address for each job to be run.
    """
    # Normalize input 'args' to ensure it's a sequence of sequences.
    _args = (args,) if not isinstance(args[0], (list, tuple, np.ndarray)) else args

    # Prepare lists for round-robin assignment across hosts.
    hosts = list(host_ips)
    counters = [0] * len(hosts)
    assigned_ips = []

    # Loop through each job (the length of the first arg list determines the number of jobs).
    for i, _ in enumerate(_args[0]):
        # Select the next host in a round-robin fashion.
        host_index = i % len(hosts)
        host = hosts[host_index]
        # Get the list of workers available on the selected host.
        ip_list = host_ips[host]

        # Assign the next available worker on that host to the current job.
        assigned_ips.append(ip_list[counters[host_index]])
        # Move to the next worker on that host for the next time it's selected (inner round-robin).
        counters[host_index] = (counters[host_index] + 1) % len(ip_list)

    return assigned_ips


def get_worker_host(client: dask.distributed.client.Client | None = None) -> str:
    """Gets the host IP of the current Dask worker or a default host from the cluster.

    This function first attempts to identify the host of the worker currently
    executing the code. If the code is not running on a Dask worker (e.g., it's
    running on the main client process), it falls back to querying the client for
    a list of all hosts and returns the first one as a default.

    Parameters
    ----------
    client : dask.distributed.Client, optional
        An active Dask client. This is only required if the function
        is called from a process that is not a Dask worker.

    Returns
    -------
    str
        The IP address (str) of a host.

    Raises
    ------
    Exception
        If the code is not on a worker and no `client` is provided.
    """
    try:
        # First, try to get the current worker directly. This succeeds only if run on a worker.
        host = dask.distributed.get_worker().host
    except ValueError:
        # This error means the code is not running on a Dask worker.
        # A client is required to find other hosts in the cluster.
        if client is None:
            raise Exception(f"Expected client, got {type(client)}")
        # As a fallback, get all hosts from the cluster and pick the first one.
        host = list(get_host_ips(client))[0]
    return host


def init_zarr_groups(out_path, groups, multiburst, **kwargs):
    """
    Initialize the necessary Zarr groups on disk before writing data.
    """
    import os

    # Create and save an empty dataset in each group path to initialize it.
    def write_dt(dt_group):
        dt = xr.DataTree.from_dict({kk: xr.Dataset() for kk in ["measurements"]})
        write(
            dt,
            os.path.join(out_path, dt_group),
            mode="w",
            storage_options=get_storage_options(kwargs),
        )

    map(
        write_dt,
        args=list(groups),
    )
