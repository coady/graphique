"""
Core utilities that add pandas-esque features to arrow arrays and tables.

Arrow forbids subclassing, so the classes are for logical grouping.
Their methods are called as functions.
"""
import bisect
import contextlib
import functools
import inspect
import itertools
import json
from concurrent import futures
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, Optional, Sequence, Union, get_type_hints
import numpy as np  # type: ignore
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds

threader = futures.ThreadPoolExecutor(pa.cpu_count())


class Agg:
    """Aggregation options."""

    option_map = {
        'all': pc.ScalarAggregateOptions,
        'any': pc.ScalarAggregateOptions,
        'approximate_median': pc.ScalarAggregateOptions,
        'count': pc.CountOptions,
        'count_distinct': pc.CountOptions,
        'distinct': pc.CountOptions,
        'list': pc.ElementWiseAggregateOptions,
        'max': pc.ScalarAggregateOptions,
        'mean': pc.ScalarAggregateOptions,
        'min': pc.ScalarAggregateOptions,
        'min_max': pc.ScalarAggregateOptions,
        'one': pc.ScalarAggregateOptions,  # no options
        'product': pc.ScalarAggregateOptions,
        'stddev': pc.VarianceOptions,
        'sum': pc.ScalarAggregateOptions,
        'tdigest': pc.TDigestOptions,
        'variance': pc.VarianceOptions,
    }

    associatives = {'all', 'any', 'first', 'last', 'max', 'min', 'one', 'product', 'sum'}

    def __init__(self, name: str, alias: str = '', **options):
        self.name = name
        self.alias = alias or name
        self.options = options

    def astuple(self, func: str) -> tuple:
        return f'hash_{func}', self.option_map[func](**self.options)

    @classmethod
    def getfunc(cls, name: str) -> Callable:
        """Return callable with named parameters for keyword arguments."""
        if name == 'element':
            return lambda array, index: array[index]
        if name == 'slice':
            return lambda array, start, stop, step: array[start:stop:step]
        return getattr(pc, name)


@dataclass(frozen=True)
class Compare:
    """Comparable wrapper for bisection search."""

    value: object
    __slots__ = ('value',)  # slots keyword in Python >=3.10

    def __lt__(self, other):
        return self.value < other.as_py()

    def __gt__(self, other):
        return self.value > other.as_py()


def sort_key(name: str) -> tuple:
    """Parse sort order."""
    return name.lstrip('-'), ('descending' if name.startswith('-') else 'ascending')


def decode(array: pa.Array) -> pa.Array:
    """Decode dictionary array."""
    return array.dictionary_decode() if isinstance(array, pa.DictionaryArray) else array


def register(func: Callable) -> pc.Function:
    """Register user defined scalar function."""
    doc = inspect.getdoc(func)
    doc = {'summary': doc.splitlines()[0], 'description': doc}  # type: ignore
    annotations = dict(get_type_hints(func))
    pc.register_scalar_function(func, func.__name__, doc, annotations, annotations.pop('return'))
    return pc.get_function(func.__name__)


@register
def digitize(
    ctx, array: pa.float64(), bins: pa.list_(pa.float64()), right: pa.bool_()  # type: ignore
) -> pa.int64():  # type: ignore
    """Return the indices of the bins to which each value in input array belongs."""
    return pa.array(np.digitize(array, bins.values, right.as_py()))


class ListChunk(pa.lib.BaseListArray):
    def from_counts(counts: pa.IntegerArray, values: pa.Array) -> pa.LargeListArray:
        """Return list array by converting counts into offsets."""
        offsets = pa.concat_arrays([pa.array([0], counts.type), pc.cumulative_sum_checked(counts)])
        cls = pa.LargeListArray if offsets.type == 'int64' else pa.ListArray
        return cls.from_arrays(offsets, values)

    def from_scalars(values: Iterable) -> pa.LargeListArray:
        """Return list array from array scalars."""
        return ListChunk.from_counts(pa.array(map(len, values)), pa.concat_arrays(values))

    def element(self, index: int) -> pa.Array:
        """element at index of each list scalar; defaults to null"""
        with contextlib.suppress(ValueError):
            return pc.list_element(self, index)
        size = -index if index < 0 else index + 1
        if isinstance(self, pa.ChunkedArray):
            self = self.combine_chunks()
        mask = np.asarray(Column.fill_null(pc.list_value_length(self), 0)) < size
        offsets = np.asarray(self.offsets[1:] if index < 0 else self.offsets[:-1])
        return pc.list_flatten(self).take(pa.array(offsets + index, mask=mask))

    def first(self) -> pa.Array:
        """first value of each list scalar"""
        return ListChunk.element(self, 0)

    def last(self) -> pa.Array:
        """last value of each list scalar"""
        return ListChunk.element(self, -1)

    def scalars(self) -> Iterable:
        empty = pa.array([], self.type.value_type)
        return (scalar.values or empty for scalar in self)

    def map_list(self, func: Callable, **kwargs) -> pa.lib.BaseListArray:
        """Return list array by mapping function across scalars, with null handling."""
        values = [func(value, **kwargs) for value in ListChunk.scalars(self)]
        return ListChunk.from_scalars(values)

    def aggregate(self, **funcs: Optional[pc.FunctionOptions]) -> pa.StructArray:
        """Return aggregated scalars by grouping each hash function on the parent indices.

        If there are empty or null scalars, then the result must be padded with null defaults and
        reordered. If the function is a `count`, then the default is 0.
        """
        items = {f'hash_{name}': funcs[name] for name in funcs}.items()
        indices = pc.list_parent_indices(self)
        groups = pc._group_by([pc.list_flatten(self)] * len(funcs), [indices], items)
        if len(groups) == len(self):  # no empty or null scalars
            return groups
        mask = pc.equal(pc.list_value_length(self), 0)
        empties = pc.indices_nonzero(Column.fill_null(mask, True))
        indices = pa.concat_arrays([groups.field(-1).cast('uint64'), empties])
        counters = [field.name for field in groups.type if 'count' in field.name]
        empties = pa.repeat(pa.scalar(dict.fromkeys(counters, 0), groups.type), len(empties))
        return pa.concat_arrays([groups, empties]).take(pc.sort_indices(indices))

    def min_max(self, **options) -> pa.Array:
        if pa.types.is_dictionary(self.type.value_type):
            self = ListChunk.aggregate(self, distinct=None).field(0)
            self = type(self).from_arrays(self.offsets, self.values.dictionary_decode())
        return ListChunk.aggregate(self, min_max=pc.ScalarAggregateOptions(**options)).field(0)

    def min(self, **options) -> pa.Array:
        """min value of each list scalar"""
        return ListChunk.min_max(self, **options).field('min')

    def max(self, **options) -> pa.Array:
        """max value of each list scalar"""
        return ListChunk.min_max(self, **options).field('max')

    def mode(self, **options) -> pa.Array:
        """modes of each list scalar"""
        return ListChunk.map_list(self, pc.mode, **options)

    def quantile(self, **options) -> pa.Array:
        """quantiles of each list scalar"""
        return ListChunk.map_list(self, pc.quantile, **options)

    def index(self, **options) -> pa.Array:
        """index for first occurrence of each list scalar"""
        values = [pc.index(value, **options) for value in ListChunk.scalars(self)]
        return Column.from_scalars(values)

    @register
    def list_all(ctx, self: pa.list_(pa.bool_())) -> pa.bool_():  # type: ignore
        """Test whether all elements in a boolean array evaluate to true."""
        return ListChunk.aggregate(self, all=None).field(0)

    @register
    def list_any(ctx, self: pa.list_(pa.bool_())) -> pa.bool_():  # type: ignore
        """Test whether any element in a boolean array evaluates to true."""
        return ListChunk.aggregate(self, any=None).field(0)


class Column(pa.ChunkedArray):
    """Chunked array interface as a namespace of functions."""

    def from_scalars(values: Sequence) -> pa.Array:
        """Return array from arrow scalars."""
        return pa.array((value.as_py() for value in values), values[0].type)

    def scalar_type(self):
        return self.type.value_type if pa.types.is_dictionary(self.type) else self.type

    def is_list_type(self):
        funcs = pa.types.is_list, pa.types.is_large_list, pa.types.is_fixed_size_list
        return any(func(self.type) for func in funcs)

    def combine_dictionaries(self) -> tuple:
        if not pa.types.is_dictionary(self.type):
            return self, None
        if not self or len(self) <= max(len(chunk.dictionary) for chunk in self.iterchunks()):
            return self.cast(self.type.value_type), None
        self = self.unify_dictionaries()
        indices = pa.chunked_array(chunk.indices for chunk in self.iterchunks())
        return self.chunk(0).dictionary, indices

    def indices(self) -> Union[pa.Array, pa.ChunkedArray]:
        """Return chunked indices suitable for aggregation."""
        if isinstance(self, pa.Array):
            return pa.array(np.arange(len(self)))
        offsets = list(itertools.accumulate(map(len, self.iterchunks())))
        return pa.chunked_array(map(np.arange, [0] + offsets, offsets))

    def call_indices(self, func: Callable) -> pa.ChunkedArray:
        dictionary, indices = Column.combine_dictionaries(self)
        if indices is None:
            return func(dictionary)
        return pa.chunked_array(
            pa.DictionaryArray.from_arrays(chunk, dictionary) for chunk in func(indices).chunks
        )

    def fill_null_backward(self) -> pa.ChunkedArray:
        """`fill_null_backward` with dictionary support."""
        return Column.call_indices(self, pc.fill_null_backward)

    def fill_null_forward(self) -> pa.ChunkedArray:
        """`fill_null_forward` with dictionary support."""
        return Column.call_indices(self, pc.fill_null_forward)

    def fill_null(self, value) -> pa.ChunkedArray:
        """Optimized `fill_null` to check `null_count`."""
        return self.fill_null(value) if self.null_count else self

    def sort_values(self) -> pa.Array:
        dictionary, indices = Column.combine_dictionaries(self)
        if indices is None:
            return dictionary
        return pc.sort_indices(pc.sort_indices(dictionary)).take(indices)

    def diff(self, func: Callable = pc.subtract) -> pa.ChunkedArray:
        """Return discrete differences between adjacent values."""
        return func(self[1:], self[:-1])

    def partition_offsets(self, predicate: Callable = pc.not_equal, *args) -> pa.IntegerArray:
        """Return list array offsets by partitioning on discrete differences.

        Args:
            predicate: binary function applied to adjacent values
            *args: apply binary function to scalar, using `subtract` as the difference function
        """
        ends = [pa.array([True])]
        mask = predicate(Column.diff(self), *args) if args else Column.diff(self, predicate)
        return pc.indices_nonzero(pa.chunked_array(ends + mask.chunks + ends))

    def min_max(self, **options):
        if pa.types.is_dictionary(self.type):
            self = self.unique().dictionary_decode()
        return pc.min_max(self, **options).as_py()

    def min(self, **options):
        """Return min of the values."""
        return Column.min_max(self, **options)['min']

    def max(self, **options):
        """Return max of the values."""
        return Column.min_max(self, **options)['max']

    def index(self, value, start=0, end=None) -> int:
        """Return the first index of a value."""
        with contextlib.suppress(NotImplementedError):
            return self.index(value, start, end).as_py()  # type: ignore
        offset = start
        for chunk in self[start:end].iterchunks():
            index = chunk.dictionary.index(value).as_py()
            if index >= 0:
                index = chunk.indices.index(index).as_py()
            if index >= 0:
                return offset + index
            offset += len(chunk)
        return -1

    def range(self, lower=None, upper=None, include_lower=True, include_upper=False) -> slice:
        """Return slice within range from a sorted array, by default a half-open interval."""
        method = bisect.bisect_left if include_lower else bisect.bisect_right
        start = 0 if lower is None else method(self, Compare(lower))
        method = bisect.bisect_right if include_upper else bisect.bisect_left
        stop = None if upper is None else method(self, Compare(upper), start)
        return slice(start, stop)

    def find(self, *values) -> Iterator[slice]:
        """Generate slices of matching rows from a sorted array."""
        stop = 0
        for value in map(Compare, sorted(values)):
            start = bisect.bisect_left(self, value, stop)
            stop = bisect.bisect_right(self, value, start)
            yield slice(start, stop)


class Table(pa.Table):
    """Table interface as a namespace of functions."""

    def map_batch(self, func: Callable, *rargs, **kwargs) -> pa.Table:
        return pa.Table.from_batches(
            threader.map(lambda batch: func(batch, *rargs, **kwargs), self.to_batches())
        )

    def union(*tables: pa.Table) -> pa.Table:
        """Return table with union of columns."""
        columns: dict = {}
        for table in tables:
            columns.update(zip(table.column_names, table))
        return pa.table(columns)

    def range(self, name: str, lower=None, upper=None, **includes) -> pa.Table:
        """Return rows within range, by default a half-open interval.

        Assumes the table is sorted by the column name, i.e., indexed.
        """
        return self[Column.range(self[name], lower, upper, **includes)]

    def is_in(self, name: str, *values) -> pa.Table:
        """Return rows which matches one of the values.

        Assumes the table is sorted by the column name, i.e., indexed.
        """
        slices = list(Column.find(self[name], *values)) or [slice(0)]
        return pa.concat_tables(self[slc] for slc in slices)

    def not_equal(self, name: str, value) -> pa.Table:
        """Return rows which don't match the value.

        Assumes the table is sorted by the column name, i.e., indexed.
        """
        (slc,) = Column.find(self[name], value)
        return pa.concat_tables([self[: slc.start], self[slc.stop :]])  # noqa: E203

    def from_offsets(self, offsets: pa.IntegerArray) -> pa.Table:
        """Return table with columns converted into list columns."""
        cls = pa.LargeListArray if offsets.type == 'int64' else pa.ListArray
        arrays = [col.combine_chunks() if col else pa.array([], col.type) for col in self]
        return pa.table([cls.from_arrays(offsets, array) for array in arrays], self.column_names)

    def from_counts(self, counts: pa.IntegerArray) -> pa.Table:
        """Return table with columns converted into list columns."""
        offsets = pa.concat_arrays([pa.array([0], counts.type), pc.cumulative_sum_checked(counts)])
        return Table.from_offsets(self, offsets)

    def partition(self, *names: str, **predicates: tuple) -> tuple:
        """Return table partitioned by discrete differences and corresponding counts.

        Args:
            *names: columns to partition by `not_equal` which will return scalars
            **predicates: inequality predicates with optional args which will return list arrays;
                if the predicate has args, it will be called on the differences
        """
        offsets = pa.chunked_array(
            Column.partition_offsets(self[name], *predicates.get(name, (pc.not_equal,)))
            for name in names + tuple(predicates)
        ).unique()
        offsets = offsets.take(pc.sort_indices(offsets))
        scalars = self.select(names).take(offsets[:-1])
        lists = self.select(set(self.column_names) - set(names))
        table = Table.union(scalars, Table.from_offsets(lists, offsets))
        return table, Column.diff(offsets)

    def group(
        self,
        *names: str,
        counts: str = '',
        first: Sequence[Agg] = (),
        last: Sequence[Agg] = (),
        **funcs: Sequence[Agg],
    ) -> Union[pa.Table, pa.RecordBatch]:
        """Group by and aggregate.

        Args:
            *names: columns to group by
            counts: alias for optional row counts
            first: columns to take first value
            last: columns to take last value
            **funcs: aggregate funcs with columns options
        """
        column = self[names[0]]
        args = [(column, 'hash_count', pc.CountOptions(mode='all'))] if counts else []
        if first or last:
            args.append((Column.indices(column), 'hash_min_max', None))
        lists: Sequence[Agg] = []
        list_cols = [self[agg.name] for agg in funcs.get('list', [])]
        if any(pa.types.is_dictionary(col.type) or Column.is_list_type(col) for col in list_cols):
            args.append((Column.indices(column), 'hash_list', None))
            lists = funcs.pop('list')
        for func in funcs:
            args += [(self[agg.name], *agg.astuple(func)) for agg in funcs[func]]  # type: ignore
        values, hashes, options = zip(*args) if args else [()] * 3
        keys = map(self.column, names)
        if isinstance(self, pa.Table):
            keys = (key.unify_dictionaries() for key in keys)  # type: ignore
            values = (
                value.unify_dictionaries() if 'distinct' in func else value
                for value, func in zip(values, hashes)
            )
        arrays = iter(pc._group_by(values, keys, zip(hashes, options)).flatten())
        columns = {counts: next(arrays)} if counts else {}
        if first or last:
            for aggs, indices in zip([first, last], next(arrays).flatten()):
                columns.update({agg.alias: self[agg.name].take(indices) for agg in aggs})
        if lists:
            indices = next(arrays)
            table = self.select({agg.name for agg in lists}).take(indices.values)
            table = Table.from_offsets(table, indices.offsets)
            columns.update({agg.alias: table[agg.name] for agg in lists})
        for aggs in funcs.values():
            columns.update(zip([agg.alias for agg in aggs], arrays))
        columns.update(zip(names, map(decode, arrays)))
        return type(self).from_pydict(columns)

    def aggregate(self, counts: str = '', **funcs: Sequence[Agg]) -> dict:
        """Return aggregated scalars as a row of data."""
        row = {name: self[name] for name in self.column_names}
        if counts:
            row[counts] = len(self)
        for key in funcs:
            func = Agg.getfunc(key)
            row.update({agg.alias: func(self[agg.name], **agg.options) for agg in funcs[key]})
        for name, value in row.items():
            if isinstance(value, pa.ChunkedArray):
                row[name] = value.combine_chunks()
        return row

    def list_value_length(self) -> pa.Array:
        lists = {name for name in self.column_names if Column.is_list_type(self[name])}
        if not lists:
            raise ValueError(f"no list columns available: {self.column_names}")
        counts, *others = (pc.list_value_length(self[name]) for name in lists)
        if any(counts != other for other in others):
            raise ValueError(f"list columns have different value lengths: {lists}")
        return counts.chunk(0)

    def sort_list(self, *names: str, length: Optional[int] = None) -> pa.Table:
        """Return table with list columns sorted within scalars."""
        keys = dict(map(sort_key, names))  # type: ignore
        columns = {name: Column.sort_values(pc.list_flatten(self[name])) for name in keys}
        columns[''] = pc.list_parent_indices(self[next(iter(keys))])
        keys = dict({'': 'ascending'}, **keys)
        indices = pc.sort_indices(pa.table(columns), sort_keys=keys.items())
        counts = Table.list_value_length(self)
        if length is not None:
            indices = pa.concat_arrays(
                scalar.values[:length] for scalar in ListChunk.from_counts(counts, indices)
            )
            counts = pc.min_element_wise(counts, length)
        table = Table.select_list(self, pc.list_flatten).take(indices)
        return Table.union(self, Table.from_counts(table, counts))

    def select_list(self, apply: Callable = lambda c: c) -> pa.Table:
        """Return table with only the list columns."""
        names = [name for name in self.column_names if Column.is_list_type(self[name])]
        return pa.table(list(map(apply, self.select(names))), names)

    def sort_indices(self, *names: str, length: Optional[int] = None) -> pa.Table:
        """Return indices which would sort the table by columns, optimized for fixed length."""
        func = pc.sort_indices
        if length is not None:
            func = functools.partial(pc.select_k_unstable, k=length)
        keys = dict(map(sort_key, names))  # type: ignore
        table = pa.table({name: Column.sort_values(self[name]) for name in keys})
        return func(table, sort_keys=keys.items())

    def sort(self, *names: str, length: Optional[int] = None, indices: str = '') -> pa.Table:
        """Return table sorted by columns, optimized for fixed length.

        Args:
            *names: columns to sort by
            length: maximum number of rows to return
            indices: include original indices in the table
        """
        indices_ = Table.sort_indices(self, *names, length=length)
        table = self.take(indices_)
        if indices:
            table = table.append_column(indices, indices_)
        func = lambda name: not name.startswith('-') and not self[name].null_count  # noqa: E731
        metadata = {'index_columns': list(itertools.takewhile(func, names))}
        return table.replace_schema_metadata({'pandas': json.dumps(metadata)})

    def filter_list(self, expr: ds.Expression) -> 'Table':
        """Return table with list columns filtered within scalars."""
        table = Table.select_list(self)
        counts = Table.list_value_length(table)
        first = table[0].combine_chunks()
        indices = pc.list_parent_indices(first)
        columns = [
            pc.list_flatten(column) if Column.is_list_type(column) else column.take(indices)
            for column in self
        ]
        flattened = pa.table(columns, self.column_names)
        mask = ds.dataset(flattened).to_table(columns={'mask': expr})['mask'].combine_chunks()
        masks = type(first).from_arrays(first.offsets, mask)
        counts = Column.fill_null(ListChunk.aggregate(masks, sum=None).field(0), 0)
        flattened = flattened.select(table.column_names).filter(mask)
        return Table.union(self, Table.from_counts(flattened, counts))

    def matched(self, func: Callable, *names: str) -> pa.Table:
        for name in names:
            if Column.is_list_type(self[name]):
                self = self.append_column('', getattr(ListChunk, func.__name__)(self[name]))
                self = Table.filter_list(self, pc.field(name) == pc.field('')).drop([''])
            else:
                self = self.filter(pc.field(name) == func(self[name]))
        return self
