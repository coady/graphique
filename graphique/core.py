"""
Core utilities that add pandas-esque features to arrow arrays and tables.

Arrow forbids subclassing, so the classes are for logical grouping.
Their methods are called as functions.
"""
import bisect
import contextlib
import functools
import itertools
import operator
from concurrent import futures
from dataclasses import dataclass
from datetime import time
from typing import Callable, Iterable, Iterator, Optional, Sequence, Union
import numpy as np  # type: ignore
import pyarrow as pa
import pyarrow.compute as pc

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
        'one': pc.ScalarAggregateOptions,  # no options
        'product': pc.ScalarAggregateOptions,
        'stddev': pc.VarianceOptions,
        'sum': pc.ScalarAggregateOptions,
        'tdigest': pc.TDigestOptions,
        'variance': pc.VarianceOptions,
    }

    associatives = {'all', 'any', 'count', 'first', 'last', 'max', 'min', 'one', 'product', 'sum'}

    def __init__(self, name: str, alias: str = '', **options):
        self.name = name
        self.alias = alias or name
        self.options = options

    def astuple(self, func: str) -> tuple:
        if func == 'tdigest' and set(self.options) <= {'skip_nulls', 'min_count'}:
            func = 'approximate_median'
        return f'hash_{func}', self.option_map[func](**self.options)


@dataclass(frozen=True)
class Compare:
    """Comparable wrapper for bisection search."""

    value: object
    __slots__ = ('value',)  # slots keyword in Python >=3.10

    def __lt__(self, other):
        return self.value < other.as_py()

    def __gt__(self, other):
        return self.value > other.as_py()


class ListChunk(pa.lib.BaseListArray):
    value_length = pc.list_value_length

    def from_counts(counts: pa.IntegerArray, values: pa.Array) -> pa.LargeListArray:
        """Return list array by converting counts into offsets."""
        offsets = np.concatenate([[0], np.cumsum(counts)])  # type: ignore
        return pa.LargeListArray.from_arrays(offsets, values)

    def element(self, index: int) -> pa.Array:
        """element at index of each list scalar; defaults to null"""
        with contextlib.suppress(ValueError):
            return pc.list_element(self, index)
        size = -index if index < 0 else index + 1
        mask = np.asarray(self.value_lengths().fill_null(0)) < size
        offsets = np.asarray(self.offsets[1:] if index < 0 else self.offsets[:-1])
        return self.values.take(pa.array(offsets + index, mask=mask))

    def count(self, **options) -> pa.IntegerArray:
        """non-null count of each list scalar"""
        return ListChunk.reduce(self, pc.count, 'int64', pc.CountOptions(**options))

    def count_distinct(self, **options) -> pa.IntegerArray:
        """non-null distinct count of each list scalar"""
        if pa.types.is_dictionary(self.values.type):
            self = type(self).from_arrays(self.offsets, self.values.indices)
        return ListChunk.reduce(self, pc.count_distinct, 'int64', pc.CountOptions(**options))

    def first(self) -> pa.Array:
        """first value of each list scalar"""
        return ListChunk.element(self, 0)

    def last(self) -> pa.Array:
        """last value of each list scalar"""
        return ListChunk.element(self, -1)

    def one(self) -> pa.Array:
        """one arbitrary value of each list scalar (pyarrow >=8 only)"""
        return ListChunk.aggregate(self, one=None).field(0)

    def unique(self) -> pa.lib.BaseListArray:
        """unique values within each scalar"""
        return ListChunk.map_list(self, lambda arr: Column.decode(arr.unique()))

    def distinct(self, **options) -> pa.lib.BaseListArray:
        """non-null distinct values within each scalar"""
        return ListChunk.aggregate(self, distinct=pc.CountOptions(**options)).field(0)

    def scalars(self) -> Iterable:
        empty = pa.array([], self.type.value_type)
        return (scalar.values or empty for scalar in self)

    def map_list(self, func: Callable, **kwargs) -> pa.lib.BaseListArray:
        """Return list array by mapping function across scalars, with null handling."""
        values = [func(value, **kwargs) for value in ListChunk.scalars(self)]
        return ListChunk.from_counts(pa.array(map(len, values)), pa.concat_arrays(values))

    def filter_list(self, mask: pa.BooleanArray) -> pa.lib.BaseListArray:
        """Return list array by selecting true values."""
        masks = type(self).from_arrays(self.offsets, mask)
        counts = pa.array(scalar.values.true_count for scalar in masks)
        return ListChunk.from_counts(counts, self.values.filter(mask))

    def aggregate(self, **funcs) -> pa.Array:
        items = {f'hash_{name}': funcs[name] for name in funcs}.items()
        return pc._group_by([self.values] * len(funcs), [self.value_parent_indices()], items)

    def reduce(self, func: Callable, tp=None, options=None) -> pa.Array:
        with contextlib.suppress(ValueError):
            groups = ListChunk.aggregate(self, **{func.__name__: options})
            if len(groups) == len(self):  # empty scalars cause index collision
                return groups.field(0)
        values = (
            None if scalar.values is None else func(scalar.values, options=options).as_py()
            for scalar in self
        )
        return pa.array(values, tp or self.type.value_type)

    def min(self, **options) -> pa.Array:
        """min value of each list scalar"""
        if pa.types.is_dictionary(self.values.type):
            self = ListChunk.unique(self)
        return ListChunk.reduce(self, pc.min, options=pc.ScalarAggregateOptions(**options))

    def max(self, **options) -> pa.Array:
        """max value of each list scalar"""
        if pa.types.is_dictionary(self.values.type):
            self = ListChunk.unique(self)
        return ListChunk.reduce(self, pc.max, options=pc.ScalarAggregateOptions(**options))

    def sum(self, **options) -> pa.Array:
        """sum of each list scalar"""
        return ListChunk.reduce(self, pc.sum, options=pc.ScalarAggregateOptions(**options))

    def product(self, **options) -> pa.Array:
        """product of each list scalar"""
        return ListChunk.reduce(self, pc.product, options=pc.ScalarAggregateOptions(**options))

    def mean(self, **options) -> pa.FloatingPointArray:
        """mean of each list scalar"""
        return ListChunk.reduce(self, pc.mean, 'float64', pc.ScalarAggregateOptions(**options))

    def mode(self, **options) -> pa.Array:
        """modes of each list scalar"""
        array = ListChunk.map_list(self, lambda arr: pc.mode(arr, **options).field(0))
        return array if 'n' in options else ListChunk.first(array)

    def quantile(self, **options) -> pa.Array:
        """quantiles of each list scalar"""
        array = ListChunk.map_list(self, pc.quantile, options=pc.QuantileOptions(**options))
        return array if 'q' in options else ListChunk.first(array)

    def tdigest(self, **options) -> pa.Array:
        """approximate quantiles of each list scalar"""
        if set(options) <= {'skip_nulls', 'min_count'}:
            return ListChunk.reduce(
                self, pc.approximate_median, 'float64', pc.ScalarAggregateOptions(**options)
            )
        return ListChunk.map_list(self, pc.tdigest, options=pc.TDigestOptions(**options))

    def stddev(self, **options) -> pa.FloatingPointArray:
        """stddev of each list scalar"""
        return ListChunk.reduce(self, pc.stddev, 'float64', pc.VarianceOptions(**options))

    def variance(self, **options) -> pa.FloatingPointArray:
        """variance of each list scalar"""
        return ListChunk.reduce(self, pc.variance, 'float64', pc.VarianceOptions(**options))

    def mask(self) -> pa.ListArray:
        return pa.ListArray.from_arrays(self.offsets, Column.mask(self.values))

    def any(self, **options) -> pa.BooleanArray:
        """any true of each list scalar"""
        return ListChunk.reduce(
            ListChunk.mask(self), pc.any, options=pc.ScalarAggregateOptions(**options)
        )

    def all(self, **options) -> pa.BooleanArray:
        """all true of each list scalar"""
        return ListChunk.reduce(
            ListChunk.mask(self), pc.all, options=pc.ScalarAggregateOptions(**options)
        )


class Column(pa.ChunkedArray):
    """Chunked array interface as a namespace of functions."""

    is_in = pc.is_in_meta_binary

    def map(self, func: Callable, **kwargs) -> pa.ChunkedArray:
        map_ = threader.map if self.num_chunks > 1 else map
        return pa.chunked_array(map_(functools.partial(func, **kwargs), self.iterchunks()))  # type: ignore

    def scalar_type(self):
        return self.type.value_type if pa.types.is_dictionary(self.type) else self.type

    def is_list_type(self):
        return pa.types.is_list(self.type) or pa.types.is_large_list(self.type)

    def decode(self) -> pa.ChunkedArray:
        """Native `dictionary_decode` is only on `DictionaryArray`."""
        return self.cast(self.type.value_type) if pa.types.is_dictionary(self.type) else self

    def unify_dictionaries(self) -> pa.ChunkedArray:
        """Native `unify_dictionaries` is inefficient if the dictionary is too large."""
        if not pa.types.is_dictionary(self.type):
            return self
        if not self or len(self) <= max(len(chunk.dictionary) for chunk in self.iterchunks()):
            return self.cast(self.type.value_type)
        return self.unify_dictionaries()

    def dict_flatten(self):
        indices = pa.chunked_array(chunk.indices for chunk in self.iterchunks())
        return self.chunk(0).dictionary, indices

    def unique_indices(self, counts=False) -> tuple:
        """Return index array of first occurrences, optionally with counts.

        Relies on `unique` having stable ordering.
        """
        self = Column.unify_dictionaries(self)
        if pa.types.is_dictionary(self.type):
            _, self = Column.dict_flatten(self)
        values, counts = self.value_counts().flatten() if counts else (self.unique(), None)
        return pc.index_in(values, value_set=self), counts

    def indices(self) -> Union[pa.Array, pa.ChunkedArray]:
        """Return chunked indices suitable for aggregation."""
        if isinstance(self, pa.Array):
            return pa.array(np.arange(len(self)))
        offsets = list(itertools.accumulate(map(len, self.iterchunks())))
        return pa.chunked_array(map(np.arange, [0] + offsets, offsets))

    def mask(self, func='and', ignore_case=False, regex=False, **query) -> pa.ChunkedArray:
        """Return boolean mask array which matches query predicates."""
        masks = []
        if Column.is_list_type(self):
            self = pc.list_flatten(self)
        options = {'ignore_case': True} if ignore_case else {}
        for op, value in query.items():
            if hasattr(Column, op):
                masks.append(getattr(Column, op)(self, value))
            elif 'is_' not in op:
                op += '_regex' * regex
                masks.append(Column.call(self, getattr(pc, op), value, **options))
            elif value:
                masks.append(Column.call(self, getattr(pc, op)))
        if masks:
            return functools.reduce(getattr(pc, func), masks)
        with contextlib.suppress(NotImplementedError):
            self = Column.call(self, pc.binary_length)
        return self.cast('bool')

    def call(self, func: Callable, *args, **kwargs) -> pa.ChunkedArray:
        """Call compute function on array with support for dictionaries."""
        scalar = Column.scalar_type(self)
        args = tuple(pa.scalar(arg, scalar) if isinstance(arg, time) else arg for arg in args)
        with contextlib.suppress(NotImplementedError):
            return func(self, *args, **kwargs)
        self = Column.unify_dictionaries(self)
        if not pa.types.is_dictionary(self.type):
            return func(self, *args, **kwargs)
        dictionary, indices = Column.dict_flatten(self)
        return func(dictionary, *args, **kwargs).take(indices)

    def equal(self, value) -> pa.ChunkedArray:
        """Return boolean mask array which matches scalar value."""
        if value is None:
            return pc.is_null(self)
        return Column.call(self, pc.equal, value)

    def not_equal(self, value) -> pa.ChunkedArray:
        """Return boolean mask array which doesn't match scalar value."""
        if value is None:
            return pc.is_valid(self)
        return Column.call(self, pc.not_equal, value)

    def fill_null(self, value) -> pa.ChunkedArray:
        """Optimized `fill_null` to check `null_count`."""
        return self.fill_null(value) if self.null_count else self

    def sort_values(self) -> pa.Array:
        self = Column.unify_dictionaries(self)
        if not pa.types.is_dictionary(self.type):
            return self
        dictionary, indices = Column.dict_flatten(self)
        return pc.sort_indices(pc.sort_indices(dictionary)).take(indices)

    def sort(self, reverse=False, length: int = None) -> pa.Array:
        """Return sorted values, optimized for fixed length."""
        func = pc.sort_indices
        if length is not None:
            func = functools.partial(pc.select_k_unstable, k=length)
        keys = {'': 'descending' if reverse else 'ascending'}
        return self.take(func(Column.sort_values(self), sort_keys=keys.items()))

    def diff(self, func: Callable = pc.subtract) -> pa.ChunkedArray:
        """Return discrete differences between adjacent values."""
        return func(self[1:], self[:-1])

    def partition_offsets(self, predicate: Callable = pc.not_equal, *args) -> pa.IntegerArray:
        """Return list array offsets by partitioning on discrete differences.

        Args:
            predicate: binary function applied to adjacent values
            args: apply binary function to scalar, using `subtract` as the difference function
        """
        ends = [pa.array([True])]
        if args:
            mask = Column.call(Column.diff(self), predicate, *args)
        else:
            mask = Column.diff(self, predicate)
        return pc.indices_nonzero(pa.concat_arrays(ends + mask.chunks + ends))

    def min_max(self):
        if pa.types.is_dictionary(self.type):
            self = self.unique().dictionary_decode()
        return pc.min_max(self).as_py()

    def min(self):
        """Return min of the values."""
        return Column.min_max(self)['min']

    def max(self):
        """Return max of the values."""
        return Column.min_max(self)['max']

    def digitize(self, bins: Iterable, right=False) -> pa.ChunkedArray:
        """Return the indices of the bins to which each value in input array belongs."""
        if not isinstance(bins, (pa.Array, np.ndarray)):
            bins = pa.array(bins, self.type)
        return Column.map(self, np.digitize, bins=bins, right=bool(right))

    def count(self, value) -> int:
        """Return number of occurrences of value."""
        if value is None:
            return self.null_count
        if not isinstance(value, bool):
            self, value = Column.equal(self, value), True
        getter = operator.attrgetter('true_count' if value else 'false_count')
        return sum(map(getter, Column.mask(self).iterchunks()))

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

    def any(self) -> Optional[bool]:
        """Return whether any values evaluate to true."""
        return pc.any(Column.mask(self)).as_py()

    def all(self) -> Optional[bool]:
        """Return whether all values evaluate to true."""
        return pc.all(Column.mask(self)).as_py()

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

    applied = {'fill_null', 'digitize'}
    projected = {
        'coalesce',
        'add',
        'subtract',
        'multiply',
        'divide',
        'power',
        'min_element_wise',
        'max_element_wise',
        'binary_join_element_wise',
        'atan2',
        'bit_wise_or',
        'bit_wise_and',
        'bit_wise_xor',
        'shift_left',
        'shift_right',
        'years_between',
        'quarters_between',
        'weeks_between',
        'days_between',
        'hours_between',
        'minutes_between',
        'seconds_between',
        'milliseconds_between',
        'microseconds_between',
        'nanoseconds_between',
    }

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
        arrays = [cls.from_arrays(offsets, *column.chunks) for column in self.combine_chunks()]
        return pa.table(arrays, self.column_names)

    def partition(self, *names: str, **predicates: tuple) -> tuple:
        """Return table partitioned by discrete differences and corresponding counts.

        Args:
            names: columns to partition by `not_equal` which will return scalars
            predicates: inequality predicates with optional args which will return list arrays;
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
            names: columns to group by
            counts: alias for optional row counts
            first: columns to take first value
            last: columns to take last value
            funcs: aggregate funcs with columns options
        """
        none = not (counts or first or last or any(funcs.values()))
        column = self[names[0]]
        args = [(column, 'hash_count', pc.CountOptions(mode='all'))] if none or counts else []
        if first or last:
            args.append((Column.indices(column), 'hash_min_max', None))
        lists: Sequence[Agg] = []
        list_cols = [self[agg.name] for agg in funcs.get('list', [])]
        if any(pa.types.is_dictionary(col.type) or Column.is_list_type(col) for col in list_cols):
            args.append((Column.indices(column), 'hash_list', None))
            lists = funcs.pop('list')
        for func in funcs:
            for agg in funcs[func]:
                column = Column.mask(self[agg.name]) if func in ('any', 'all') else self[agg.name]
                args.append((column, *agg.astuple(func)))  # type: ignore
        values, hashes, options = zip(*args)
        keys = map(self.column, names)
        arrays = iter(pc._group_by(values, keys, zip(hashes, options)).flatten())
        if none:
            next(arrays)
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
        columns.update(zip(names, arrays))
        return type(self).from_pydict(columns)

    def list_value_length(self) -> pa.ChunkedArray:
        lists = {name for name in self.column_names if Column.is_list_type(self[name])}
        if not lists:
            raise ValueError(f"no list columns available: {self.column_names}")
        counts, *others = (pc.list_value_length(self[name]) for name in lists)
        if any(counts != other for other in others):
            raise ValueError(f"list columns have different value lengths: {lists}")
        return counts

    def sort_list(self, *names: str, reverse=False, length: int = None) -> pa.Table:
        """Return table with list columns sorted within scalars."""
        keys = {'': 'ascending'}
        keys.update(dict.fromkeys(names, 'descending' if reverse else 'ascending'))
        columns = {name: Column.sort_values(pc.list_flatten(self[name])) for name in names}
        columns[''] = pc.list_parent_indices(self[names[0]])
        indices = pc.sort_indices(pa.table(columns), sort_keys=keys.items())
        counts = Table.list_value_length(self)
        if length is not None:
            indices = pa.concat_arrays(
                scalar.values[:length] for scalar in ListChunk.from_counts(counts, indices)
            )
            counts = pc.min_element_wise(counts, length)
        table = Table.select_list(self, pc.list_flatten).take(indices).combine_chunks()
        arrays = [ListChunk.from_counts(counts, *column.chunks) for column in table]
        return Table.union(self, pa.table(arrays, table.column_names))

    def select_list(self, apply: Callable = lambda c: c) -> pa.Table:
        """Return table with only the list columns."""
        names = [name for name in self.column_names if Column.is_list_type(self[name])]
        return pa.table(list(map(apply, self.select(names))), names)

    def sort(self, *names: str, reverse=False, length: int = None) -> pa.Table:
        """Return table sorted by columns, optimized for fixed length."""
        func = pc.sort_indices
        if length is not None:
            func = functools.partial(pc.select_k_unstable, k=length)
        keys = dict.fromkeys(names, 'descending' if reverse else 'ascending')
        table = pa.table({name: Column.sort_values(self[name]) for name in names})
        return self.take(func(table, sort_keys=keys.items()))

    def mask(self, name: str, apply: dict = {}, **query: dict) -> pa.ChunkedArray:
        """Return mask array which matches query."""
        masks = []
        for op, column in apply.items():
            column = self[apply[op]]
            func = getattr(pc, op)
            if not Column.is_list_type(self[name]):
                mask = func(self[name], column)
            elif Column.is_list_type(column):
                mask = func(pc.list_flatten(self[name]), pc.list_flatten(column))
            else:
                mask = pa.chunked_array(map(func, ListChunk.scalars(self[name]), column))
            masks.append(mask)
        if query:
            masks.append(Column.mask(self[name], **query))
        return functools.reduce(getattr(pc, 'and'), masks)

    def apply(
        self,
        name: str,
        alias: str = '',
        cast: str = '',
        checked=False,
        ignore_case=False,
        regex=False,
        **partials,
    ) -> pa.Table:
        """Return view of table with functions applied across columns."""
        column = self[name]
        options = {'ignore_case': True} if ignore_case else {}
        for func, arg in partials.items():
            if func in Table.projected:
                others = (self[name] for name in (arg if isinstance(arg, list) else [arg]))
                column = getattr(pc, func + '_checked' * checked)(column, *others)
            elif func in Table.applied:
                column = getattr(Column, func)(column, arg)
            elif not isinstance(arg, bool):
                column = getattr(pc, func + '_regex' * regex)(column, arg, **options)
            elif arg and Column.is_list_type(column):
                column = Column.map(column, getattr(ListChunk, func))
            elif arg:
                column = getattr(pc, func + '_checked' * checked)(column)
        if cast:
            column = column.cast(cast)
        if alias:
            return self.append_column(alias, column)
        return self.set_column(self.column_names.index(name), name, column)

    def filter_list(self, mask: pa.BooleanArray):
        """Return table with list columns filtered within scalars."""
        table = Table.select_list(self).combine_chunks()
        arrays = [ListChunk.filter_list(column.chunk(0), mask) for column in table]
        return Table.union(self, pa.table(arrays, table.column_names))

    def matched(self, func: Callable, *names: str) -> pa.Table:
        for name in names:
            if Column.is_list_type(self[name]):
                scalars = list(ListChunk.scalars(self[name]))
                column = pa.array(map(func, scalars), self[name].type.value_type)
                self = Table.filter_list(self, pa.concat_arrays(map(Column.equal, scalars, column)))
            else:
                self = self.filter(Column.equal(self[name], func(self[name])))
        return self
