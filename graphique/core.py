"""
Core utilities that add pandas-esque features to arrow arrays and table.

Arrow forbids subclassing, so the classes are for logical grouping.
Their methods are called as functions.
"""
import bisect
import collections
import contextlib
import functools
import json
import operator
from concurrent import futures
from typing import Callable, Iterator, Optional
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from .arrayed import arggroupby, argunique  # type: ignore


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


class Chunk:
    def encode(self):
        if not isinstance(self, pa.DictionaryArray):
            if not self.null_count and np.can_cast(self.type.to_pandas_dtype(), np.intp):
                return None, self
            self = self.dictionary_encode()
        if not self.indices.null_count:
            return self.dictionary, self.indices
        dictionary = pa.concat_arrays([self.dictionary, pa.array([None], self.dictionary.type)])
        return dictionary, self.indices.fill_null(len(self.dictionary))

    def arggroupby(self) -> tuple:
        dictionary, array = Chunk.encode(self)
        keys, indices = arggroupby(array)
        return (dictionary.take(keys) if dictionary else keys), indices

    def argunique(self, reverse=False) -> pa.IntegerArray:
        _, array = Chunk.encode(self)
        indices = argunique(array[::-1] if reverse else array)
        return pc.subtract(pa.scalar(len(array) - 1), indices) if reverse else indices

    def call(self: pa.DictionaryArray, func: Callable, *args, **kwargs) -> pa.Array:
        dictionary = func(self.dictionary, *args, **kwargs)
        return pa.DictionaryArray.from_arrays(self.indices, dictionary)

    def is_in(self, values, invert=False) -> pa.Array:
        if not isinstance(self, pa.DictionaryArray):
            return pa.array(np.isin(self, values, invert=invert))
        return Chunk.call(self, np.isin, values, invert=invert).cast(pa.bool_())

    def to_null(array: np.ndarray) -> pa.Array:
        func = np.isnat if array.dtype.type is np.datetime64 else np.isnan
        return pa.array(array, mask=func(array))


class Column(pa.ChunkedArray):
    """Chunked array interface as a namespace of functions."""

    threader = futures.ThreadPoolExecutor(pa.cpu_count())

    def map(func: Callable, *arrays: pa.ChunkedArray) -> Iterator:
        return Column.threader.map(func, *(arr.iterchunks() for arr in arrays))

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
            return functools.reduce(lambda *args: pc.call_function(func, args), masks)
        with contextlib.suppress(NotImplementedError):
            self = Column.call(self, pc.binary_length)
        return self.cast(pa.bool_())

    def call(self, func: Callable, *args) -> pa.ChunkedArray:
        """Call compute function on array with support for dictionaries."""
        if args and not isinstance(args[0], pa.ChunkedArray) and func is not pc.match_substring:
            args = (pa.scalar(args[0], getattr(self.type, 'value_type', self.type)),)
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

    def is_in(self, values, invert=False) -> pa.ChunkedArray:
        """Return boolean mask array which matches any value."""
        return pa.chunked_array(Column.map(rpartial(Chunk.is_in, values, invert), self))

    def arggroupby(self) -> dict:
        """Return groups of index arrays."""
        empty = pa.array([], pa.int64())
        result = collections.defaultdict(lambda: [empty] * self.num_chunks)  # type: dict
        for index, (keys, groups) in enumerate(Column.map(Chunk.arggroupby, self)):
            for key, group in zip(keys.to_pylist(), groups):
                result[key][index] = group.values
        return {key: pa.chunked_array(result[key]) for key in result}

    def sort(self, reverse=False, length: int = None) -> pa.Array:
        """Return sorted values, optimized for fixed length."""
        # arrow may seg fault when `sort_indices` is called on a non-chunked array
        if length is not None:
            func = lambda v, i: v.take(i[-length:] if reverse else i[:length])
            chunks = Column.map(func, self, pc.call_function('sort_indices', [self]))
            self = pa.chunked_array([pa.concat_arrays(chunks)])
        elif self.num_chunks > 1:
            self = pa.chunked_array([pa.concat_arrays(self.iterchunks())])
        indices = pc.call_function('sort_indices', [self])
        return self.take((indices[::-1] if reverse else indices)[:length])

    def mapreduce(self, mapper, reducer, default=None):
        if self.null_count:
            self = self.filter(self.is_valid())
        try:
            value = reducer(Column.map(mapper, self))
        except ValueError:
            return default
        return value.item() if hasattr(value, 'item') else value

    def sum(self, exp: int = 1):
        """Return sum of the values, with optional exponentiation."""
        if exp == 1:
            return pc.sum(self).as_py()
        return Column.mapreduce(self, lambda ch: np.sum(np.power(ch, exp)), sum)

    def min(self):
        """Return min of the values."""
        return Column.mapreduce(self, np.min, min)

    def max(self):
        """Return max of the values."""
        return Column.mapreduce(self, np.max, max)

    def compare(self, func, value):
        if isinstance(value, pa.ChunkedArray):
            chunks = Column.map(func, self, value)
        else:
            chunks = Column.map(rpartial(func, value), self)
        if self.null_count:
            chunks = Column.threader.map(Chunk.to_null, chunks)
        return pa.chunked_array(chunks)

    def minimum(self, value) -> pa.ChunkedArray:
        """Return element-wise minimum of values."""
        return Column.compare(self, np.minimum, value)

    def maximum(self, value) -> pa.ChunkedArray:
        """Return element-wise maximum of values."""
        return Column.compare(self, np.maximum, value)

    def absolute(self) -> pa.ChunkedArray:
        """Return absolute values."""
        chunks = Column.map(np.absolute, self)
        if self.null_count:
            chunks = Column.threader.map(Chunk.to_null, chunks)
        return pa.chunked_array(chunks)

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
        types = [getattr(col.type, 'value_type', col.type) for col in self.columns]
        return dict(zip(self.column_names, types))

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

    def num_chunks(self) -> Optional[int]:
        """Return number of chunks if consistent across columns, else None."""
        shapes = {tuple(map(len, column.iterchunks())) for column in self.columns}
        return None if len(shapes) > 1 else sum(map(len, shapes))

    def take_chunks(self, indices: pa.ChunkedArray) -> pa.Table:
        """Return table with selected rows from a non-offset chunked array.

        `ChunkedArray.take` concatenates the chunks and as such is not performant for grouping.
        Assumes the shape of the columns is the same.
        """
        assert Table.num_chunks(self) is not None
        columns = [Column.map(pa.Array.take, column, indices) for column in self.columns]
        return pa.Table.from_arrays(list(map(pa.concat_arrays, columns)), self.column_names)

    def group(self, name: str, reverse=False, predicate=int, sort=False) -> Iterator[pa.Table]:
        """Generate tables grouped by column, with filtering and slicing on table length."""
        num_chunks = Table.num_chunks(self)
        if num_chunks is None:
            self, num_chunks = self.combine_chunks(), 1
        if num_chunks == 1:
            _, groups = Chunk.arggroupby(self[name].chunk(0))
            groups = map(operator.attrgetter('values'), groups)
            take = self.take
        else:
            groups = Column.arggroupby(self[name]).values()
            take = Table.take_chunks.__get__(self)  # type: ignore
        groups = [indices for indices in groups if predicate(len(indices))]
        if sort:
            groups.sort(key=len)
        return map(take, reversed(groups) if reverse else groups)

    def unique(self, name: str, reverse=False) -> pa.Table:
        """Return table with first or last occurrences from grouping by column."""
        num_chunks = Table.num_chunks(self)
        if num_chunks is None:
            self, num_chunks = self.combine_chunks(), 1
        if num_chunks > 1:
            chunks = Column.map(rpartial(Chunk.argunique, reverse), self[name])
            chunks = (chunk[::-1] if reverse else chunk for chunk in chunks)
            self = Table.take_chunks(self, pa.chunked_array(chunks))
        return self.take(Chunk.argunique(self[name].chunk(0), reverse) if num_chunks else [])

    def sort(self, *names: str, reverse=False, length: int = None) -> pa.Table:
        """Return table sorted by columns."""
        self = self.combine_chunks()
        indices = pa.array(np.arange(len(self)))
        for name in reversed(names):
            indices = indices.take(pc.call_function('sort_indices', [self[name].take(indices)]))
        return self.take((indices[::-1] if reverse else indices)[:length])

    def mask(self, name: str, **query: dict) -> Iterator[pa.Array]:
        """Return mask array which matches query."""
        masks, column = [], self[name]
        partials = dict(query.pop('apply', {}))
        for func in set(Table.projected) & set(partials):
            column = Table.projected[func](column, self[partials.pop(func)])
        masks += [getattr(pc, op)(column, self[partials[op]]) for op in partials]
        if query:
            masks.append(Column.mask(column, **query))
        return functools.reduce(lambda *args: pc.call_function('and', args), masks)

    def apply(self, name: str, alias: str = '', **partials) -> pa.Table:
        """Return view of table with functions applied across columns."""
        column = self[name]
        for func, arg in partials.items():
            if func in Table.projected:
                column = Table.projected[func](column, self[arg])
            elif not isinstance(arg, bool):
                column = Table.applied[func](column, arg)
            elif arg:
                column = Table.applied[func](column)
        if alias:
            return self.add_column(len(self.column_names), alias, column)
        return self.set_column(self.column_names.index(name), name, column)

    def matched(self, func: Callable, *names: str) -> pa.Table:
        for name in names:
            self = self.filter(Column.equal(self[name], func(self[name])))
        return self
