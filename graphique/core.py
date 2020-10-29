"""
Core utilities that add pandas-esque features to arrow arrays and table.

Arrow forbids subclassing, so the classes are for logical grouping.
Their methods are called as functions.
"""
import abc
import bisect
import contextlib
import functools
import json
import operator
from concurrent import futures
from typing import Callable, Iterable, Iterator, Optional
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from .arrayed import group_indices, unique_indices  # type: ignore


class Compare:
    """Comparable wrapper for bisection search."""

    __slots__ = ('value',)

    def __init__(self, value):
        self.value = value

    def __lt__(self, other):
        return self.value < other.as_py()

    def __gt__(self, other):
        return self.value > other.as_py()


class Scalar(metaclass=abc.ABCMeta):
    """Abstract base class to distinguish scalar values from other iterables."""

    @classmethod
    def __subclasshook__(cls, other):
        return not issubclass(other, Iterable) or NotImplemented


Scalar.register(str)
Scalar.register(bytes)


def rpartial(func, *values):
    """Return function with right arguments partially bound."""
    return lambda arg: func(arg, *values)


class Chunk:
    def encode(self):
        if not isinstance(self, pa.DictionaryArray):
            if not self.null_count and np.can_cast(self.type.to_pandas_dtype(), np.intp):
                return self
            self = self.dictionary_encode()
        if not self.indices.null_count:
            return self.indices
        return self.indices.fill_null(len(self.dictionary))

    def group_indices(self) -> Iterable[pa.Array]:
        _, indices = group_indices(Chunk.encode(self))
        return map(operator.attrgetter('values'), indices)

    def unique_indices(self, reverse=False) -> pa.IntegerArray:
        array = Chunk.encode(self)
        if not reverse:
            return unique_indices(array)
        return pc.subtract(pa.scalar(len(array) - 1), unique_indices(array[::-1]))

    def call(self: pa.DictionaryArray, func: Callable, *args, **kwargs) -> pa.Array:
        dictionary = func(self.dictionary, *args, **kwargs)
        return pa.DictionaryArray.from_arrays(self.indices, dictionary)

    def to_null(array: np.ndarray) -> pa.Array:
        func = np.isnat if array.dtype.type in (np.datetime64, np.timedelta64) else np.isnan
        return pa.array(array, mask=func(array))


class ListChunk(pa.ListArray):
    count = pa.ListArray.value_lengths

    def first(self) -> pa.Array:
        """first value of each list scalar"""
        mask = np.asarray(self.value_lengths().fill_null(0)) == 0
        indices = np.asarray(self.offsets[:-1])
        return self.values.take(pa.array(indices, mask=mask))

    def last(self) -> pa.Array:
        """last value of each list scalar"""
        mask = np.asarray(self.value_lengths().fill_null(0)) == 0
        indices = np.asarray(self.offsets[1:]) - 1
        return self.values.take(pa.array(indices, mask=mask))

    def unique(self) -> pa.ListArray:
        """unique values within each scalar"""
        empty = pa.array([], self.type.value_type)
        values = [empty if scalar.values is None else scalar.values.unique() for scalar in self]
        offsets = np.concatenate([[0], np.cumsum(list(map(len, values)))])
        return pa.ListArray.from_arrays(offsets, pa.concat_arrays(values))

    def reduce(self, func: Callable, tp=None) -> pa.Array:
        values = (func(scalar.values) if scalar.values else None for scalar in self)
        return pa.array(values, tp or self.type.value_type)

    def min(self) -> pa.Array:
        """min value of each list scalar"""

        def func(array):
            if array.null_count:
                array = array.filter(array.is_valid())
            return np.min(array) if len(array) else None

        try:
            return ListChunk.reduce(self, lambda arr: pc.min_max(arr).as_py()['min'])
        except NotImplementedError:
            return ListChunk.reduce(self, func)

    def max(self) -> pa.Array:
        """max value of each list scalar"""

        def func(array):
            if array.null_count:
                array = array.filter(array.is_valid())
            return np.max(array) if len(array) else None

        try:
            return ListChunk.reduce(self, lambda arr: pc.min_max(arr).as_py()['max'])
        except NotImplementedError:
            return ListChunk.reduce(self, func)

    def sum(self) -> pa.Array:
        """sum each list scalar"""
        return ListChunk.reduce(self, Column.sum)

    def mean(self) -> pa.Array:
        """mean of each list scalar"""
        return ListChunk.reduce(self, Column.mean, pa.float64())

    def mode(self):
        """mode of each list scalar"""
        return ListChunk.reduce(self, Column.mode)

    def stddev(self) -> Optional[float]:
        """stddev of each list scalar"""
        return ListChunk.reduce(self, Column.stddev, pa.float64())

    def variance(self) -> Optional[float]:
        """variance of each list scalar"""
        return ListChunk.reduce(self, Column.variance, pa.float64())


class Column(pa.ChunkedArray):
    """Chunked array interface as a namespace of functions."""

    threader = futures.ThreadPoolExecutor(pa.cpu_count())

    def map(func: Callable, *arrays: pa.ChunkedArray) -> Iterator:
        map_ = Column.threader.map if arrays[0].num_chunks > 1 else map
        return map_(func, *(arr.iterchunks() for arr in arrays))  # type: ignore

    def scalar_type(self):
        return self.type.value_type if isinstance(self.type, pa.DictionaryType) else self.type

    def mask(self, func='and', **query) -> pa.ChunkedArray:
        """Return boolean mask array which matches query predicates."""
        masks = []
        for op, value in query.items():
            if op in ('equal', 'not_equal', 'is_in'):
                masks.append(getattr(Column, op)(self, value))
            elif op == 'absolute':
                masks.append(Column.mask(getattr(Column, op)(self), func, **value))
            elif op in ('utf8_lower', 'utf8_upper', 'binary_length'):
                masks.append(Column.mask(Column.call(self, getattr(pc, op)), func, **value))
            elif '_is_' not in op:
                masks.append(Column.call(self, getattr(pc, op), value))
            elif query[op]:
                masks.append(Column.call(self, getattr(pc, op)))
        if masks:
            return functools.reduce(getattr(pc, func), masks)
        with contextlib.suppress(NotImplementedError):
            self = Column.call(self, pc.binary_length)
        return self.cast(pa.bool_())

    def call(self, func: Callable, *args) -> pa.ChunkedArray:
        """Call compute function on array with support for dictionaries."""
        if args and isinstance(args[0], Scalar) and func is not pc.match_substring:
            args = (pa.scalar(args[0], Column.scalar_type(self)),)
        if not isinstance(self.type, pa.DictionaryType):
            return func(self, *args)
        array = pa.chunked_array(Column.map(rpartial(Chunk.call, func, *args), self))
        with contextlib.suppress(ValueError):
            if array.type.value_type.bit_width <= array.type.index_type.bit_width:
                return array.cast(array.type.value_type)
        return array

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
        return Column.call(self, pc.is_in_meta_binary, values)

    def sort(self, reverse=False, length: int = None) -> pa.Array:
        """Return sorted values, optimized for fixed length."""
        if isinstance(self.type, pa.DictionaryType):
            self = self.cast(self.type.value_type)
        if length is not None:
            with contextlib.suppress(IndexError):  # fallback to sorting if length > len(chunk)
                if reverse:
                    indices = pc.partition_nth_indices(self, pivot=len(self) - length)
                    chunks = [chunk[-length:] for chunk in indices.iterchunks()]
                else:
                    indices = pc.partition_nth_indices(self, pivot=length)
                    chunks = [chunk[:length] for chunk in indices.iterchunks()]
                self = pa.chunked_array(map(pa.Array.take, self.iterchunks(), chunks))
        # arrow may seg fault when `sort_indices` is called on a non-chunked array
        if self.num_chunks > 1:
            self = pa.chunked_array([pa.concat_arrays(self.iterchunks())])
        indices = pc.sort_indices(self)
        return self and self.take((indices[::-1] if reverse else indices)[:length])

    def sum(self):
        """Return sum of the values."""
        return pc.sum(self).as_py()

    def mean(self) -> Optional[float]:
        """Return mean of the values."""
        return pc.mean(self).as_py()

    def mode(self):
        """Return mode of the values."""
        return pc.mode(self).as_py()['mode']

    def stddev(self) -> Optional[float]:
        """Return standard deviation of the values."""
        return pc.stddev(self).as_py()

    def variance(self) -> Optional[float]:
        """Return variance of the values."""
        return pc.variance(self).as_py()

    def quantile(self, *q: float) -> list:
        """Return q-th quantiles for values."""
        if self.null_count:
            self = self.filter(self.is_valid())
        if not self:
            return [None] * len(q)
        return np.quantile(self, q).tolist()

    def min_max(self, reverse=False):
        if not self:
            return None
        if isinstance(self.type, pa.DictionaryType):
            self = pa.chunked_array([self.unique().cast(self.type.value_type)])
        try:
            return pc.min_max(self)['max' if reverse else 'min'].as_py()
        except NotImplementedError:
            return Column.sort(self, reverse, length=1)[0].as_py()

    def min(self):
        """Return min of the values."""
        return Column.min_max(self, reverse=False)

    def max(self):
        """Return max of the values."""
        return Column.min_max(self, reverse=True)

    def compare(self, func, value):
        if isinstance(value, pa.ChunkedArray):
            chunks = Column.map(func, self, value)
        else:
            chunks = Column.map(rpartial(func, value), self)
        return pa.chunked_array(map(Chunk.to_null, chunks) if self.null_count else chunks)

    def minimum(self, value) -> pa.ChunkedArray:
        """Return element-wise minimum of values."""
        return Column.compare(self, np.minimum, value)

    def maximum(self, value) -> pa.ChunkedArray:
        """Return element-wise maximum of values."""
        return Column.compare(self, np.maximum, value)

    def absolute(self) -> pa.ChunkedArray:
        """Return absolute values."""
        chunks = Column.map(np.absolute, self)
        return pa.chunked_array(map(Chunk.to_null, chunks) if self.null_count else chunks)

    def count(self, value) -> int:
        """Return number of occurrences of value."""
        if value is None:
            return self.null_count
        if not isinstance(value, bool):
            self, value = Column.equal(self, value), True
        getter = operator.attrgetter('true_count' if value else 'false_count')
        return sum(map(getter, Column.mask(self).iterchunks()))

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

    projected = {
        'add': pc.add,
        'subtract': pc.subtract,
        'multiply': pc.multiply,
        'divide': pc.divide,
        'minimum': Column.minimum,
        'maximum': Column.maximum,
    }
    applied = {
        'fill_null': pc.fill_null,
        'binary_length': pc.binary_length,
        'utf8_lower': pc.utf8_lower,
        'utf8_upper': pc.utf8_upper,
        'absolute': Column.absolute,
    }

    def index(self) -> list:
        """Return index column names from pandas metadata."""
        metadata = self.schema.metadata or {}
        return json.loads(metadata.get(b'pandas', b'{}')).get('index_columns', [])

    def types(self) -> dict:
        """Return mapping of column types."""
        return {name: Column.scalar_type(self[name]) for name in self.column_names}

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

    def group(self, name: str, reverse=False, predicate=int, sort=False) -> Iterator[pa.Table]:
        """Generate tables grouped by column, with filtering and slicing on table length."""
        self = self.combine_chunks()
        groups = Chunk.group_indices(self[name].chunk(0))
        groups = [indices for indices in groups if predicate(len(indices))]
        if sort:
            groups.sort(key=len)
        return map(self.take, reversed(groups) if reverse else groups)

    def unique(self, name: str, reverse=False, count='') -> pa.Table:
        """Return table with first or last occurrences from grouping by column.

        Optionally include counts in an additional column.
        Faster than :meth:`group` when only scalars are needed.
        """
        self = self.combine_chunks()
        table = self.take(Chunk.unique_indices(self[name].chunk(0), reverse))
        if not count:
            return table
        _, counts = (self[name][::-1] if reverse else self[name]).value_counts().flatten()
        return table.add_column(len(table.column_names), count, counts)

    def sort(self, *names: str, reverse=False, length: int = None) -> pa.Table:
        """Return table sorted by columns."""
        self = self.combine_chunks()
        indices = pa.array(np.arange(len(self)))
        for name in reversed(names):
            column = self[name]
            if isinstance(column.type, pa.DictionaryType):
                column = column.cast(column.type.value_type)
            indices = indices.take(pc.sort_indices(column.take(indices)))
        return self.take((indices[::-1] if reverse else indices)[:length])

    def mask(self, name: str, **query: dict) -> pa.Array:
        """Return mask array which matches query."""
        masks, column = [], self[name]
        partials = dict(query.pop('apply', {}))
        for func in set(Table.projected) & set(partials):
            column = Table.projected[func](column, self[partials.pop(func)])
        masks += [getattr(pc, op)(column, self[partials[op]]) for op in partials]
        if query:
            masks.append(Column.mask(column, **query))
        return functools.reduce(getattr(pc, 'and'), masks)

    def apply(self, name: str, alias: str = '', **partials) -> pa.Table:
        """Return view of table with functions applied across columns."""
        column = self[name]
        for func, arg in partials.items():
            if func in Table.projected:
                column = Table.projected[func](column, self[arg])
            elif not isinstance(arg, bool):
                column = Table.applied[func](column, arg)
            elif arg and func in Table.applied:
                column = Table.applied[func](column)
            elif arg:
                column = pa.chunked_array(Column.map(getattr(ListChunk, func), column))
        if alias:
            return self.add_column(len(self.column_names), alias, column)
        return self.set_column(self.column_names.index(name), name, column)

    def matched(self, func: Callable, *names: str) -> pa.Table:
        for name in names:
            self = self.filter(Column.equal(self[name], func(self[name])))
        return self
