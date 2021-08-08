"""
Core utilities that add pandas-esque features to arrow arrays and tables.

Arrow forbids subclassing, so the classes are for logical grouping.
Their methods are called as functions.
"""
import bisect
import contextlib
import functools
import operator
from concurrent import futures
from datetime import time
from typing import Callable, Iterable, Iterator, Sequence
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from .arrayed import group_indices, split, unique_indices  # type: ignore


class Compare:
    """Comparable wrapper for bisection search."""

    __slots__ = ('value',)

    def __init__(self, value):
        self.value = value

    def __lt__(self, other):
        return self.value < other.as_py()

    def __gt__(self, other):
        return self.value > other.as_py()


def rpartial(func, *values):
    """Return function with right arguments partially bound."""
    return lambda arg: func(arg, *values)


class Chunk(pa.Array):
    def encode(self):
        if not pa.types.is_dictionary(self.type):
            if not self.null_count and np.can_cast(self.type.to_pandas_dtype(), np.intp):
                return self
            self = self.dictionary_encode()
        if not self.indices.null_count:
            return self.indices
        return self.indices.fill_null(len(self.dictionary))

    def unique_indices(self, reverse=False, count=False) -> tuple:
        array = Chunk.encode(self)
        if not reverse:
            return unique_indices(array, count)
        indices, counts = unique_indices(array[::-1], count)
        return pc.subtract(len(array) - 1, indices), counts

    def fill_null(self: pa.DictionaryArray, dictionary: pa.Array) -> pa.DictionaryArray:
        indices = self.indices.fill_null(len(dictionary) - 1)
        return pa.DictionaryArray.from_arrays(indices, dictionary)

    def take_list(self, indices: pa.lib.BaseListArray) -> pa.lib.BaseListArray:
        assert len(self) == len(indices.values)
        return type(indices).from_arrays(indices.offsets, self.take(indices.values))

    def partition_nth_indices(self, pivot: int) -> pa.IntegerArray:
        if pivot >= 0:
            return pc.partition_nth_indices(self, pivot=pivot)[:pivot]
        return pc.partition_nth_indices(self, pivot=len(self) + pivot)[pivot:]

    def partition_nth(self, pivot: int) -> pa.Array:
        if len(self) <= abs(pivot):
            return self
        return self.take(Chunk.partition_nth_indices(self, pivot))

    def sort_keys(self) -> pa.Array:
        if not pa.types.is_dictionary(self.type):
            return self
        if len(self) <= len(self.dictionary):
            return self.dictionary_decode()
        keys = pc.sort_indices(pc.sort_indices(self.dictionary))
        return keys.take(self.indices)

    def index(self: pa.DictionaryArray, value) -> int:
        index = self.dictionary.index(value).as_py()
        return index if index < 0 else self.indices.index(index).as_py()


class ListChunk(pa.lib.BaseListArray):
    count = operator.methodcaller('value_lengths')

    def getitem(self, index: int) -> pa.Array:
        mask = np.asarray(self.value_lengths().fill_null(0)) == 0
        offsets = np.asarray(self.offsets[1:] if index < 0 else self.offsets[:-1])
        return self.values.take(pa.array(offsets + index, mask=mask))

    def first(self) -> pa.Array:
        """first value of each list scalar"""
        return ListChunk.getitem(self, 0)

    def last(self) -> pa.Array:
        """last value of each list scalar"""
        return ListChunk.getitem(self, -1)

    def unique(self) -> pa.lib.BaseListArray:
        """unique values within each scalar"""
        return ListChunk.map_list(self, lambda arr: Column.decode(arr.unique()))

    def map_list(self, func: Callable) -> pa.lib.BaseListArray:
        """Return list array by mapping function across scalars, with null handling."""
        empty = pa.array([], self.type.value_type)
        values = [func(scalar.values or empty) for scalar in self]
        return split(pa.array(map(len, values)), pa.concat_arrays(values))

    def filter_list(self, mask: pa.BooleanArray) -> pa.lib.BaseListArray:
        """Return list array by selecting true values."""
        masks = type(self).from_arrays(self.offsets, mask)
        counts = pa.array(scalar.values.true_count for scalar in masks)
        return split(counts, self.values.filter(mask))

    def reduce(self, func: Callable, tp=None) -> pa.Array:
        values = (func(scalar.values).as_py() if scalar.values else None for scalar in self)
        return pa.array(values, tp or self.type.value_type)

    def min(self) -> pa.Array:
        """min value of each list scalar"""
        with contextlib.suppress(NotImplementedError):
            return ListChunk.reduce(self, lambda arr: pc.min_max(arr)['min'])
        return ListChunk.reduce(self, lambda arr: Chunk.partition_nth(arr, 1)[0])

    def max(self) -> pa.Array:
        """max value of each list scalar"""
        with contextlib.suppress(NotImplementedError):
            return ListChunk.reduce(self, lambda arr: pc.min_max(arr)['max'])
        return ListChunk.reduce(self, lambda arr: Chunk.partition_nth(arr, -1)[-1])

    def sum(self) -> pa.Array:
        """sum of each list scalar"""
        return ListChunk.reduce(self, pc.sum)

    def mean(self) -> pa.FloatingPointArray:
        """mean of each list scalar"""
        return ListChunk.reduce(self, pc.mean, pa.float64())

    def mode(self, length: int = 0) -> pa.Array:
        """modes of each list scalar"""
        array = ListChunk.map_list(self, lambda arr: pc.mode(arr, length or 1).field(0))
        return array if length else ListChunk.first(array)

    def quantile(self, q: Sequence[float] = ()) -> pa.Array:
        """quantiles of each list scalar"""
        array = ListChunk.map_list(self, functools.partial(pc.quantile, q=q or 0.5))
        return array if q else ListChunk.first(array)

    def stddev(self) -> pa.FloatingPointArray:
        """stddev of each list scalar"""
        return ListChunk.reduce(self, pc.stddev, pa.float64())

    def variance(self) -> pa.FloatingPointArray:
        """variance of each list scalar"""
        return ListChunk.reduce(self, pc.variance, pa.float64())

    def any(self) -> pa.BooleanArray:
        """any true of each list scalar"""
        values = (None if scalar.values is None else Column.any(scalar.values) for scalar in self)
        return pa.array(values, pa.bool_())

    def all(self) -> pa.BooleanArray:
        """all true of each list scalar"""
        values = (None if scalar.values is None else Column.all(scalar.values) for scalar in self)
        return pa.array(values, pa.bool_())


class Column(pa.ChunkedArray):
    """Chunked array interface as a namespace of functions."""

    threader = futures.ThreadPoolExecutor(pa.cpu_count())

    def map(func: Callable, *arrays: pa.ChunkedArray) -> Iterator:
        map_ = Column.threader.map if arrays[0].num_chunks > 1 else map
        return map_(func, *(arr.iterchunks() for arr in arrays))  # type: ignore

    def scalar_type(self):
        return self.type.value_type if pa.types.is_dictionary(self.type) else self.type

    def is_list_type(self):
        return pa.types.is_list(self.type) or pa.types.is_large_list(self.type)

    def decode(self, check=False) -> pa.ChunkedArray:
        with contextlib.suppress(ValueError, AttributeError):
            not check or self.type.value_type.bit_width
            return self.cast(self.type.value_type)
        return self

    def combine_chunks(self) -> pa.Array:
        """Native `combine_chunks` doesn't support empty chunks."""
        if self.num_chunks > 1:
            return self.combine_chunks()
        return self.chunk(0) if self.num_chunks else pa.array([], self.type)

    def mask(self, func='and', ignore_case=False, regex=False, **query) -> pa.ChunkedArray:
        """Return boolean mask array which matches query predicates."""
        masks = []
        if Column.is_list_type(self):
            self = pa.chunked_array(chunk.values for chunk in self.iterchunks())
        options = {'ignore_case': True} if ignore_case else {}
        for op, value in query.items():
            if hasattr(Column, op):
                masks.append(getattr(Column, op)(self, value))
            elif '_is_' not in op:
                op += '_regex' * regex
                masks.append(Column.call(self, getattr(pc, op), value, **options))
            elif value:
                masks.append(Column.call(self, getattr(pc, op)))
        if masks:
            return functools.reduce(getattr(pc, func), masks)
        with contextlib.suppress(NotImplementedError):
            self = Column.call(self, pc.binary_length)
        return self.cast(pa.bool_())

    def call(self, func: Callable, *args, **kwargs) -> pa.ChunkedArray:
        """Call compute function on array with support for dictionaries."""
        self = Column.decode(self, check=True)
        scalar = Column.scalar_type(self)
        args = tuple(pa.scalar(arg, scalar) if isinstance(arg, time) else arg for arg in args)
        try:
            return func(self, *args, **kwargs)
        except NotImplementedError:
            if not pa.types.is_dictionary(self.type):
                raise
        if not self:
            return self
        self = self.unify_dictionaries()
        dictionary = func(self.chunk(0).dictionary, *args, **kwargs)
        array = pa.chunked_array(
            pa.DictionaryArray.from_arrays(chunk.indices, dictionary) for chunk in self.iterchunks()
        )
        return Column.decode(array, check=True)

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

    def is_in(self, values) -> pa.ChunkedArray:
        """Return boolean mask array which matches any value."""
        with contextlib.suppress(NotImplementedError):
            return Column.call(self, pc.is_in_meta_binary, values)
        if not values:
            return pa.array(np.full(len(self), False))
        masks = (pc.equal(self, value) for value in values)
        return functools.reduce(pc.or_, masks).fill_null(False)

    def fill_null(self, value) -> pa.ChunkedArray:
        """Replace each null element in values with fill_value with dictionary support."""
        if not self.null_count:
            return self
        with contextlib.suppress(NotImplementedError):
            return self.fill_null(value)
        self = self.unify_dictionaries()
        end = pa.array([value], self.type.value_type)
        dictionary = pa.concat_arrays([self.chunk(0).dictionary, end])
        return pa.chunked_array(Column.map(rpartial(Chunk.fill_null, dictionary), self))

    def sort(self, reverse=False, length: int = None) -> pa.Array:
        """Return sorted values, optimized for fixed length."""
        if len(self[:length]) < len(self):
            func = rpartial(Chunk.partition_nth, (-length if reverse else length))  # type: ignore
            self = pa.chunked_array(Column.map(func, Column.decode(self)))
        order = 'descending' if reverse else 'ascending'
        array = Column.combine_chunks(self)
        return array.take(pc.array_sort_indices(Chunk.sort_keys(array), order=order)[:length])

    def diff(self, func: Callable = pc.subtract) -> pa.ChunkedArray:
        """Return discrete differences between adjacent values."""
        self = Column.decode(self)
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
        return pa.array(*np.nonzero(pa.concat_arrays(ends + mask.chunks + ends)), pa.int32())

    def min_max(self, reverse=False):
        if not self:
            return None
        if pa.types.is_dictionary(self.type):
            self = pa.chunked_array([self.unique().cast(self.type.value_type)])
        with contextlib.suppress(NotImplementedError):
            return pc.min_max(self)['max' if reverse else 'min'].as_py()
        return Column.sort(self, reverse, length=1)[0].as_py()

    def min(self):
        """Return min of the values."""
        return Column.min_max(self, reverse=False)

    def max(self):
        """Return max of the values."""
        return Column.min_max(self, reverse=True)

    def digitize(self, bins: Iterable, right=False) -> pa.ChunkedArray:
        """Return the indices of the bins to which each value in input array belongs."""
        if not isinstance(bins, (pa.Array, np.ndarray)):
            bins = pa.array(bins, self.type)
        func = functools.partial(np.digitize, bins=bins, right=bool(right))
        return pa.chunked_array(Column.map(func, self))

    def count(self, value) -> int:
        """Return number of occurrences of value."""
        if value is None:
            return self.null_count
        if not isinstance(value, bool):
            self, value = Column.equal(self, value), True
        getter = operator.attrgetter('true_count' if value else 'false_count')
        return sum(map(getter, Column.mask(self).iterchunks()))

    def index(self, value, start=0, end=None) -> int:
        """Return number of occurrences of value."""
        if not pa.types.is_dictionary(self.type):
            return self.index(value, start, end).as_py()  # type: ignore
        offset = start
        for chunk in self[start:end].iterchunks():
            index = Chunk.index(chunk, value)
            if index >= 0:
                return offset + index
            offset += len(chunk)
        return -1

    def any(self) -> bool:
        """Return whether any values evaluate to true."""
        return pc.any(Column.mask(self)).as_py()

    def all(self) -> bool:
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
    }

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

    def group_indices(self, *names: str) -> pa.LargeListArray:
        arrays = (Chunk.encode(Column.combine_chunks(self[name])) for name in names)
        _, indices = group_indices(next(arrays))
        for array in arrays:
            groups = [group_indices(scalar.values)[1] for scalar in Chunk.take_list(array, indices)]
            indices = pa.concat_arrays(
                Chunk.take_list(scalar.values, group) for scalar, group in zip(indices, groups)
            )
        return indices

    def group(self, *names: str, reverse=False, length: int = None) -> tuple:
        """Return table grouped by columns and corresponding counts."""
        indices = Table.group_indices(self, *names)
        indices = (indices[::-1] if reverse else indices)[:length]
        scalars = ListChunk.first(indices)
        columns = {name: self[name].take(scalars) for name in names}
        for name in set(self.column_names) - set(names):
            columns[name] = Chunk.take_list(Column.combine_chunks(self[name]), indices)
        return pa.Table.from_pydict(columns), indices.value_lengths()

    def partition(self, *names: str, **predicates: tuple) -> tuple:
        """Return table partitioned by discrete differences and corresponding counts.

        Args:
            names: columns to partition by `not_equal` which will return scalars
            predicates: inequality predicates with optional args which will return list arrays;
                if the predicate has args, it will be called on the differences
        """
        names += tuple(predicates)
        default = (pc.not_equal,)
        offsets = Column.partition_offsets(self[names[0]], *predicates.get(names[0], default))
        for name in names[1:]:
            groups = [pa.array([0], pa.int32())]
            predicate = predicates.get(name, default)
            for scalar in pa.LargeListArray.from_arrays(offsets, Column.combine_chunks(self[name])):
                group = Column.partition_offsets(pa.chunked_array([scalar.values]), *predicate)
                groups.append(pc.add(group[1:], groups[-1][-1]))
            offsets = pa.concat_arrays(groups)
        columns = {name: self[name].take(offsets[:-1]) for name in set(names) - set(predicates)}
        for name in set(self.column_names) - set(columns):
            column = Column.combine_chunks(self[name])
            columns[name] = pa.LargeListArray.from_arrays(offsets, column)
        return pa.Table.from_pydict(columns), Column.diff(offsets)

    def unique_indices(self, *names: str, reverse=False, count=False) -> tuple:
        array = Chunk.encode(Column.combine_chunks(self[names[-1]]))
        if len(names) == 1:
            return Chunk.unique_indices(array, reverse, count)
        indices = Table.group_indices(self, *names[:-1])
        if reverse:
            indices = indices[::-1]
        items = [
            Chunk.unique_indices(scalar.values, reverse, count)
            for scalar in Chunk.take_list(array, indices)
        ]
        indices = pa.concat_arrays(
            scalar.values.take(group) for scalar, (group, _) in zip(indices, items)
        )
        return indices, (pa.concat_arrays(c for _, c in items) if count else None)

    def unique(self, *names: str, reverse=False, length: int = None, count=False) -> tuple:
        """Return table with first or last occurrences from grouping by columns.

        Optionally compute corresponding counts.
        Faster than [group][graphique.core.Table.group] when only scalars are needed.
        """
        indices, counts = Table.unique_indices(self, *names, reverse=reverse, count=count)
        return self.take(indices[:length]), (counts and counts[:length])

    def sort(self, *names: str, reverse=False, length: int = None) -> pa.Table:
        """Return table sorted by columns, optimized for single column with fixed length."""
        if len(names) == 1 and len(self[:length]) < len(self):
            array = Column.decode(Column.combine_chunks(self[names[-1]]))
            self = self.take(Chunk.partition_nth_indices(array, (-length if reverse else length)))  # type: ignore
        order = 'descending' if reverse else 'ascending'
        columns = {name: Chunk.sort_keys(Column.combine_chunks(self[name])) for name in names}
        table = pa.Table.from_pydict(columns)
        indices = pc.sort_indices(table, sort_keys=[(name, order) for name in names])
        return self and self.take(indices[:length])

    def mask(self, name: str, apply: dict = {}, **query: dict) -> pa.Array:
        """Return mask array which matches query."""
        masks = [getattr(pc, op)(self[name], self[apply[op]]) for op in apply]
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
        **partials
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
            elif arg:
                column = getattr(pc, func + '_checked' * checked)(column)
        if cast:
            column = column.cast(cast)
        if alias:
            return self.append_column(alias, column)
        return self.set_column(self.column_names.index(name), name, column)

    def matched(self, func: Callable, *names: str) -> pa.Table:
        for name in names:
            self = self.filter(Column.equal(self[name], func(self[name])))
        return self
