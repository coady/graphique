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

    def unique_indices(self, reverse=False, count=False) -> tuple:
        array = Chunk.encode(self)
        if not reverse:
            return unique_indices(array, count)
        indices, counts = unique_indices(array[::-1], count)
        return pc.subtract(pa.scalar(len(array) - 1), indices), counts

    def call(self: pa.DictionaryArray, func: Callable, *args, **kwargs) -> pa.Array:
        dictionary = func(self.dictionary, *args, **kwargs)
        return pa.DictionaryArray.from_arrays(self.indices, dictionary)

    def to_null(array: np.ndarray) -> pa.Array:
        func = np.isnat if array.dtype.type in (np.datetime64, np.timedelta64) else np.isnan
        return pa.array(array, mask=func(array))

    def take_list(self, indices: pa.ListArray) -> pa.ListArray:
        assert len(self) == len(indices.values)  # type: ignore
        return pa.ListArray.from_arrays(indices.offsets, self.take(indices.values))  # type: ignore


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
        empty = Column.decode(pa.array([], self.type.value_type))  # flatten dicts for concatenation
        values = [
            empty if scalar.values is None else Column.decode(scalar.values.unique())
            for scalar in self
        ]
        offsets = np.concatenate([[0], np.cumsum(list(map(len, values)))])
        return pa.ListArray.from_arrays(offsets, pa.concat_arrays(values))

    def reduce(self, func: Callable, tp=None) -> pa.Array:
        values = (func(scalar.values) if scalar.values else None for scalar in self)
        return pa.array(values, tp or self.type.value_type)

    def min(self) -> pa.Array:
        """min value of each list scalar"""

        def func(array):
            return array[pc.partition_nth_indices(array, pivot=1)[0].as_py()].as_py()

        try:
            return ListChunk.reduce(self, lambda arr: pc.min_max(arr).as_py()['min'])
        except NotImplementedError:
            return ListChunk.reduce(self, func)

    def max(self) -> pa.Array:
        """max value of each list scalar"""

        def func(array):
            return array[pc.partition_nth_indices(array, pivot=len(array) - 1)[-1].as_py()].as_py()

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

    def decode(self) -> pa.ChunkedArray:
        return self.cast(self.type.value_type) if isinstance(self.type, pa.DictionaryType) else self

    def combine_chunks(self) -> pa.Array:
        return self.chunk(0) if self.num_chunks == 1 else pa.concat_arrays(self.iterchunks())

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
        self = Column.decode(self)
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
        self = pa.chunked_array([Column.combine_chunks(self)])
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
        with contextlib.suppress(NotImplementedError):
            return pc.min_max(self)['max' if reverse else 'min'].as_py()
        with contextlib.suppress(NotImplementedError):
            return Column.sort(self, reverse, length=1)[0].as_py()
        if self.null_count:
            self = self.filter(self.is_valid())
        if not self:
            return None
        value = (np.max if reverse else np.min)(self)
        return value.item() if hasattr(value, 'item') else value

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

    def group_indices(self, *names: str) -> pa.ListArray:
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
        self = self.combine_chunks()
        indices = Table.group_indices(self, *names)
        indices = (indices[::-1] if reverse else indices)[:length]
        scalars = ListChunk.first(indices)
        columns = {name: self[name].take(scalars) for name in names}
        for name in set(self.column_names) - set(names):
            columns[name] = Chunk.take_list(self[name].chunk(0), indices)
        return pa.Table.from_pydict(columns), indices.value_lengths()

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

    def unique(self, *names: str, reverse=False, count=False) -> tuple:
        """Return table with first or last occurrences from grouping by columns.

        Optionally compute corresponding counts.
        Faster than [group][graphique.core.Table.group] when only scalars are needed.
        """
        self = self.combine_chunks()
        indices, counts = Table.unique_indices(self, *names, reverse=reverse, count=count)
        return self.take(indices), counts

    def sort(self, *names: str, reverse=False, length: int = None) -> pa.Table:
        """Return table sorted by columns, optimized for single column with fixed length."""
        self = self.combine_chunks()
        if len(names) == 1 and len(self[:length]) < len(self):
            column = Column.decode(self[names[-1]])
            if reverse:
                indices = pc.partition_nth_indices(column, pivot=len(self) - length)[-length:]  # type: ignore
            else:
                indices = pc.partition_nth_indices(column, pivot=length)[:length]
            self = self.take(indices)
        columns = (Column.decode(self[name]) for name in reversed(names))
        indices = pc.sort_indices(next(columns))
        for column in columns:
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
