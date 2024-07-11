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
import operator
import json
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Optional, Union, get_type_hints
import numpy as np
import pyarrow as pa
import pyarrow.acero as ac
import pyarrow.compute as pc
import pyarrow.dataset as ds
from typing_extensions import Self

Array = Union[pa.Array, pa.ChunkedArray]
Batch = Union[pa.RecordBatch, pa.Table]


class Agg:
    """Aggregation options."""

    option_map = {
        'all': pc.ScalarAggregateOptions,
        'any': pc.ScalarAggregateOptions,
        'approximate_median': pc.ScalarAggregateOptions,
        'count': pc.CountOptions,
        'count_distinct': pc.CountOptions,
        'distinct': pc.CountOptions,
        'first': pc.ScalarAggregateOptions,
        'first_last': pc.ScalarAggregateOptions,
        'last': pc.ScalarAggregateOptions,
        'list': type(None),
        'max': pc.ScalarAggregateOptions,
        'mean': pc.ScalarAggregateOptions,
        'min': pc.ScalarAggregateOptions,
        'min_max': pc.ScalarAggregateOptions,
        'one': type(None),
        'product': pc.ScalarAggregateOptions,
        'stddev': pc.VarianceOptions,
        'sum': pc.ScalarAggregateOptions,
        'tdigest': pc.TDigestOptions,
        'variance': pc.VarianceOptions,
    }

    associatives = {'all', 'any', 'first', 'last', 'max', 'min', 'one', 'product', 'sum'}
    associatives |= {'count'}  # transformed to be associative

    def __init__(self, name: str, alias: str = '', **options):
        self.name = name
        self.alias = alias or name
        self.options = options

    def astuple(self, func: str) -> tuple:
        options = self.option_map[func.rpartition('hash_')[-1]](**self.options)
        return self.name, func, options, self.alias

    one = operator.itemgetter(0)
    list = staticmethod(lambda array: array)

    @staticmethod
    def distinct(array: Array, mode: str = 'only_valid') -> pa.Array:
        values = pc.unique(array)
        if not values.null_count or mode == 'all':
            return values
        return values.drop_null() if mode == 'only_valid' else pa.array([None], array.type)


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


def register(func: Callable, kind: str = 'scalar') -> pc.Function:
    """Register user defined function by kind."""
    doc = inspect.getdoc(func)
    doc = {'summary': doc.splitlines()[0], 'description': doc}  # type: ignore
    annotations = dict(get_type_hints(func))
    result = annotations.pop('return')
    with contextlib.suppress(pa.ArrowKeyError):  # apache/arrow#{31611,31612}
        getattr(pc, f'register_{kind}_function')(func, func.__name__, doc, annotations, result)
    return pc.get_function(func.__name__)


@register
def digitize(
    ctx,
    array: pa.float64(),  # type: ignore
    bins: pa.list_(pa.float64()),  # type: ignore
    right: pa.bool_(),  # type: ignore
) -> pa.int64():  # type: ignore
    """Return the indices of the bins to which each value in input array belongs."""
    return pa.array(np.digitize(array, bins.values, right.as_py()))


class ListChunk(pa.lib.BaseListArray):
    def from_counts(counts: pa.IntegerArray, values: pa.Array) -> pa.LargeListArray:
        """Return list array by converting counts into offsets."""
        mask = None
        if counts.null_count:
            mask, counts = counts.is_null(), counts.fill_null(0)
        offsets = pa.concat_arrays([pa.array([0], counts.type), pc.cumulative_sum_checked(counts)])
        cls = pa.LargeListArray if offsets.type == 'int64' else pa.ListArray
        return cls.from_arrays(offsets, values, mask=mask)

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

    def inner_flatten(self) -> pa.lib.BaseListArray:
        """Return flattened inner lists from a nested list array."""
        offsets = self.values.offsets.take(self.offsets)
        return type(self).from_arrays(offsets, self.values.values)

    def aggregate(self, **funcs: Optional[pc.FunctionOptions]) -> pa.RecordBatch:
        """Return aggregated scalars by grouping each hash function on the parent indices.

        If there are empty or null scalars, then the result must be padded with null defaults and
        reordered. If the function is a `count`, then the default is 0.
        """
        columns = {'key': pc.list_parent_indices(self), '': pc.list_flatten(self)}
        items = [('', name, funcs[name]) for name in funcs]
        table = pa.table(columns).group_by(['key']).aggregate(items)
        indices, table = table['key'], table.remove_column(table.schema.get_field_index('key'))
        (batch,) = table.to_batches()
        if len(batch) == len(self):  # no empty or null scalars
            return batch
        mask = pc.equal(pc.list_value_length(self), 0)
        empties = pc.indices_nonzero(Column.fill_null(mask, True))
        indices = pa.chunked_array(indices.chunks + [empties.cast(indices.type)])
        columns = {}
        for field in batch.schema:
            scalar = pa.scalar(0 if 'count' in field.name else None, field.type)
            columns[field.name] = pa.repeat(scalar, len(empties))
        table = pa.concat_tables([table, pa.table(columns)]).combine_chunks()
        return table.to_batches()[0].take(pc.sort_indices(indices))

    def min_max(self, **options) -> pa.Array:
        if pa.types.is_dictionary(self.type.value_type):
            (self,) = ListChunk.aggregate(self, distinct=None)
            self = type(self).from_arrays(self.offsets, self.values.dictionary_decode())
        return ListChunk.aggregate(self, min_max=pc.ScalarAggregateOptions(**options))[0]

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
        return ListChunk.aggregate(self, all=None)[0]

    @register
    def list_any(ctx, self: pa.list_(pa.bool_())) -> pa.bool_():  # type: ignore
        """Test whether any element in a boolean array evaluates to true."""
        return ListChunk.aggregate(self, any=None)[0]


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

    def call_indices(self, func: Callable) -> Array:
        if not pa.types.is_dictionary(self.type):
            return func(self)
        array = self.combine_chunks()
        return pa.DictionaryArray.from_arrays(func(array.indices), array.dictionary)

    def fill_null_backward(self) -> Array:
        """`fill_null_backward` with dictionary support."""
        return Column.call_indices(self, pc.fill_null_backward)

    def fill_null_forward(self) -> Array:
        """`fill_null_forward` with dictionary support."""
        return Column.call_indices(self, pc.fill_null_forward)

    def fill_null(self, value) -> pa.ChunkedArray:
        """Optimized `fill_null` to check `null_count`."""
        return self.fill_null(value) if self.null_count else self

    def sort_values(self) -> Array:
        if not pa.types.is_dictionary(self.type):
            return self
        array = self if isinstance(self, pa.Array) else self.combine_chunks()
        return pc.rank(array.dictionary, 'ascending').take(array.indices)

    def pairwise_diff(self, period: int = 1) -> Array:
        """`pairwise_diff` with chunked array support."""
        return pc.pairwise_diff(self.combine_chunks(), period)

    def diff(self, func: Callable = pc.subtract, period: int = 1) -> Array:
        """Compute first order difference of an array.

        Unlike `pairwise_diff`, does not return leading nulls.
        """
        return func(self[period:], self[:-period])

    def run_offsets(self, predicate: Callable = pc.not_equal, *args) -> pa.IntegerArray:
        """Run-end encode array with leading zero, suitable for list offsets.

        Args:
            predicate: binary function applied to adjacent values
            *args: apply binary function to scalar, using `subtract` as the difference function
        """
        ends = [pa.array([True])]
        mask = predicate(Column.diff(self), *args) if args else Column.diff(self, predicate)
        return pc.indices_nonzero(pa.chunked_array(ends + mask.chunks + ends))

    def first_last(self, **options):
        if not pa.types.is_dictionary(self.type):
            return pc.first_last(self, **options).as_py()
        self = self.combine_chunks()
        indices = pc.first_last(self.indices, **options).as_py().items()
        return {key: None if index is None else self.dictionary[index] for key, index in indices}

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

    def map_batch(scanner: ds.Scanner, func: Callable, *rargs, **kwargs) -> pa.Table:
        # TODO(apache/arrow#31612): replace with user defined function for multiple kernels
        batches = [func(batch, *rargs, **kwargs) for batch in scanner.to_batches() if batch]
        return pa.Table.from_batches(batches, None if batches else scanner.projected_schema)

    def columns(self) -> dict:
        """Return columns as a dictionary."""
        return dict(zip(self.schema.names, self))

    def union(*tables: Batch) -> Batch:
        """Return table with union of columns."""
        columns: dict = {}
        for table in tables:
            columns |= Table.columns(table)
        return type(tables[0]).from_pydict(columns)

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
        return pa.concat_tables([self[: slc.start], self[slc.stop :]])

    def from_offsets(self, offsets: pa.IntegerArray, mask=None) -> pa.RecordBatch:
        """Return record batch with columns converted into list columns."""
        cls = pa.LargeListArray if offsets.type == 'int64' else pa.ListArray
        if isinstance(self, pa.Table):
            (self,) = self.combine_chunks().to_batches() or [pa.record_batch([], self.schema)]
        arrays = [cls.from_arrays(offsets, array, mask=mask) for array in self]
        return pa.RecordBatch.from_arrays(arrays, self.schema.names)

    def from_counts(self, counts: pa.IntegerArray) -> pa.RecordBatch:
        """Return record batch with columns converted into list columns."""
        mask = None
        if counts.null_count:
            mask, counts = counts.is_null(), counts.fill_null(0)
        offsets = pa.concat_arrays([pa.array([0], counts.type), pc.cumulative_sum_checked(counts)])
        return Table.from_offsets(self, offsets, mask=mask)

    def runs(self, *names: str, **predicates: tuple) -> tuple:
        """Return table grouped by pairwise differences, and corresponding counts.

        Args:
            *names: columns to partition by `not_equal` which will return scalars
            **predicates: pairwise predicates with optional args which will return list arrays;
                if the predicate has args, it will be called on the differences
        """
        offsets = pa.chunked_array(
            Column.run_offsets(self[name], *predicates.get(name, ()))
            for name in names + tuple(predicates)
        )
        offsets = offsets.unique().sort()
        scalars = self.select(names).take(offsets[:-1])
        lists = self.select(set(self.schema.names) - set(names))
        table = Table.union(scalars, Table.from_offsets(lists, offsets))
        return table, Column.diff(offsets)

    def group(self, *names: str, counts: str = '', **funcs: Sequence[Agg]) -> pa.Table:
        """Group by and aggregate.

        Args:
            *names: columns to group by
            counts: alias for optional row counts
            **funcs: aggregate funcs with columns options
        """
        prefix = 'hash_' if names else ''
        aggs = [agg.astuple(prefix + func) for func in funcs for agg in funcs[func]]
        columns = list(names) + [agg[0] for agg in aggs]
        if counts:
            aggs.append(([], 'hash_count_all', None, counts))
        if isinstance(self, pa.Table):
            decl = Declaration('table_source', self.unify_dictionaries())
        else:
            decl = Declaration.scan(self, columns=columns)
        return decl.aggregate(aggs, names).to_table(use_threads=False)

    def aggregate(self, counts: str = '', **funcs: Sequence[Agg]) -> dict:
        """Return aggregated scalars as a row of data."""
        row = {counts: len(self)} if counts else {}
        for key in ('one', 'list', 'distinct'):  # hash only functions
            func, aggs = getattr(Agg, key), funcs.pop(key, [])
            row |= {agg.alias: func(self[agg.name], **agg.options) for agg in aggs}
        if funcs:
            table = Table.group(self, **funcs)  # type: ignore
            row |= {name: table[name][0] for name in table.schema.names}
        for name, value in row.items():
            if isinstance(value, pa.ChunkedArray):
                row[name] = value.combine_chunks()
        return row

    def list_fields(self) -> set:
        return {field.name for field in self.schema if Column.is_list_type(field)}

    def list_value_length(self) -> pa.Array:
        lists = Table.list_fields(self)
        if not lists:
            raise ValueError(f"no list columns available: {self.schema.names}")
        counts, *others = (pc.list_value_length(self[name]) for name in lists)
        if any(counts != other for other in others):
            raise ValueError(f"list columns have different value lengths: {lists}")
        return counts if isinstance(counts, pa.Array) else counts.chunk(0)

    def map_list(self, func: Callable, *args, **kwargs) -> Batch:
        """Return table with function mapped across list scalars."""
        batches: Iterable = Table.split(self.select(Table.list_fields(self)))
        batches = [None if batch is None else func(batch, *args, **kwargs) for batch in batches]
        counts = pa.array(None if batch is None else len(batch) for batch in batches)
        table = pa.Table.from_batches(batch for batch in batches if batch is not None)
        return Table.union(self, Table.from_counts(table, counts))

    def sort_indices(
        self, *names: str, length: Optional[int] = None, null_placement: str = 'at_end'
    ) -> pa.Array:
        """Return indices which would sort the table by columns, optimized for fixed length."""
        func = functools.partial(pc.sort_indices, null_placement=null_placement)
        if length is not None and length < len(self):
            func = functools.partial(pc.select_k_unstable, k=length)
        keys = dict(map(sort_key, names))
        table = pa.table({name: Column.sort_values(self[name]) for name in keys})
        return func(table, sort_keys=keys.items()) if table else pa.array([], 'int64')

    def sort(
        self,
        *names: str,
        length: Optional[int] = None,
        indices: str = '',
        null_placement: str = 'at_end',
    ) -> Batch:
        """Return table sorted by columns, optimized for fixed length.

        Args:
            *names: columns to sort by
            length: maximum number of rows to return
            indices: include original indices in the table
        """
        if length == 1 and not indices:
            return Table.ranked(self, 1, *names, null_placement=null_placement)[:1]
        indices_ = Table.sort_indices(self, *names, length=length, null_placement=null_placement)
        table = self.take(indices_)
        if indices:
            table = table.append_column(indices, indices_)
        func = lambda name: not name.startswith('-') and not self[name].null_count  # noqa: E731
        metadata = {'index_columns': list(itertools.takewhile(func, names))}
        return table.replace_schema_metadata({'pandas': json.dumps(metadata)})

    def filter_list(self, expr: ds.Expression) -> Batch:
        """Return table with list columns filtered within scalars."""
        fields = Table.list_fields(self)
        tables = [
            None if batch is None else pa.Table.from_batches([batch]).filter(expr).select(fields)
            for batch in Table.split(self)
        ]
        counts = pa.array(None if table is None else len(table) for table in tables)
        table = pa.concat_tables(table for table in tables if table is not None)
        return Table.union(self, Table.from_counts(table, counts))

    def ranked(self, k: int, *names: str, null_placement: str = 'at_end') -> Batch:
        """Return table filtered by values within dense rank, similar to `select_k_unstable`."""
        if k == 1:  # optimized for min_max
            expr = isinstance(self, pa.Table)
            for name, order in map(sort_key, names):
                func = Column.min if order == 'ascending' else Column.max
                value = func(self[name], skip_nulls=null_placement == 'at_end')
                if value is None:
                    mask = (pc.field(name) if expr else self[name]).is_null()
                else:
                    mask = pc.field(name) == value if expr else pc.equal(self[name], value)
                self = self.filter(mask)
            return self
        values = None
        kwargs = dict(tiebreaker='dense', null_placement=null_placement)
        for name, order in map(sort_key, names):
            if len(self) <= k:
                break
            ranks = pc.rank(Column.sort_values(self[name]), order, **kwargs)
            if values is None:  # optimized to sort only once on first iteration
                values = ranks
            else:
                values = pc.add_checked(pc.multiply_checked(values, pc.max(ranks)), ranks)
                values = pc.rank(values, 'ascending', tiebreaker='dense')
            mask = pc.less_equal(values, k)
            self = self.filter(mask)
            values = values.filter(mask)
        return self

    def get_fragments(self) -> Iterator[ds.Fragment]:
        """Support filtered datasets if it only references partition keys."""
        expr = self._scan_options.get('filter')
        if expr is not None:  # raise ValueError if filter references other fields
            ds.dataset([], schema=self.partitioning.schema).scanner(filter=expr)
        return self._get_fragments(expr)

    def fragment_keys(self) -> list:
        """Filtered partitioned datasets may not have fragments."""
        try:
            Table.get_fragments(self)
        except (AttributeError, ValueError):
            return []
        return self.partitioning.schema.names

    def rank_keys(self, k: int, *names: str, dense: bool = True) -> tuple:
        """Return expression and unmatched fields for partitioned dataset which filters by rank.

        Args:
            k: max dense rank or length
            *names: columns to rank by
            dense: use dense rank; false indicates sorting
        """
        schema = set(Table.fragment_keys(self))
        keys = dict(itertools.takewhile(lambda key: key[0] in schema, map(sort_key, names)))
        if not keys:
            return None, names
        parts = []
        for fragment in Table.get_fragments(self):
            parts.append(ds.get_partition_keys(fragment.partition_expression))
            if not dense:
                parts[-1][''] = fragment.count_rows()
        offset, table = len(keys), pa.Table.from_pylist(parts)
        if dense:
            table = Table.ranked(table, k, *names[:offset])
        else:
            table = Table.sort(table, *names[:offset])
            index = pc.index(pc.greater_equal(pc.cumulative_sum(table['']), k), True).as_py()
            table = table[: index + 1 if index >= 0 else None]
        remaining = names[offset:]
        if remaining or not dense:  # fields with a single value are no longer needed
            selectors = [len(table[name].unique()) > 1 for name in keys]
            remaining = tuple(itertools.compress(names, selectors)) + remaining
        fields, prefix = [], []  # type: ignore
        for index, (name, order) in enumerate(keys.items()):
            expr, field, values = None, ds.field(name), table[name].unique()
            value = pc.max(values) if order == 'ascending' else pc.min(values)
            if index == len(keys) - 1:  # last one
                expr = field <= value if order == 'ascending' else field >= value
            elif len(values) > 1:
                expr = field < value if order == 'ascending' else field > value
            if expr is not None:
                fields.append(functools.reduce(operator.and_, prefix + [expr]))
            prefix.append(field == value)
            table = table.filter(prefix[-1])
        return functools.reduce(operator.or_, fields), remaining

    def flatten(self, indices: str = '') -> Iterator[pa.RecordBatch]:
        """Generate batches with list arrays flattened, optionally with parent indices."""
        offset = 0
        for batch in self.to_batches():
            _ = Table.list_value_length(batch)
            indices_ = pc.list_parent_indices(batch[Table.list_fields(batch).pop()])
            arrays = [
                pc.list_flatten(array) if Column.is_list_type(array) else array.take(indices_)
                for array in batch
            ]
            columns = dict(zip(batch.schema.names, arrays))
            if indices:
                columns[indices] = pc.add(indices_, offset)
            offset += len(batch)
            yield pa.RecordBatch.from_pydict(columns)

    def split(self) -> Iterator[Optional[pa.RecordBatch]]:
        """Generate tables from splitting list scalars."""
        lists = Table.list_fields(self)
        scalars = set(self.schema.names) - lists
        for index, count in enumerate(Table.list_value_length(self).to_pylist()):
            if count is None:
                yield None
            else:
                row = {name: pa.repeat(self[name][index], count) for name in scalars}
                row |= {name: self[name][index].values for name in lists}
                yield pa.RecordBatch.from_pydict(row)

    def size(self) -> str:
        """Return buffer size in readable units."""
        size, prefix = self.nbytes, ''
        for prefix in itertools.takewhile(lambda _: size >= 1e3, 'kMGT'):
            size /= 1e3
        return f'{size:n} {prefix}B'


class Declaration(ac.Declaration):
    """[Acero](https://arrow.apache.org/docs/python/api/acero.html) engine declaration."""

    option_map = {
        'table_source': ac.TableSourceNodeOptions,
        'scan': ac.ScanNodeOptions,
        'filter': ac.FilterNodeOptions,
        'project': ac.ProjectNodeOptions,
        'aggregate': ac.AggregateNodeOptions,
        'order_by': ac.OrderByNodeOptions,
        'hashjoin': ac.HashJoinNodeOptions,
    }

    def __init__(self, name, *args, inputs=None, **options):
        super().__init__(name, self.option_map[name](*args, **options), inputs)

    @classmethod
    def scan(cls, dataset: ds.Dataset, columns=None) -> Self:
        """Return source node from a dataset."""
        expr = dataset._scan_options.get('filter')
        self = cls('scan', dataset, filter=expr, columns=columns)
        if expr is not None:
            self = self.filter(expr)
        if isinstance(columns, Mapping):
            self = self.project(columns.values(), columns)
        elif columns is not None:
            self = self.project(map(pc.field, columns))
        return self

    def apply(self, name: str, *args, **options) -> Self:
        return type(self)(name, *args, inputs=[self], **options)

    def __getattr__(self, name: str) -> Callable:
        return functools.partial(self.apply, name)
