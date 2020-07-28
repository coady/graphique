import bisect
import collections
import functools
import itertools
import json
from concurrent import futures
from typing import Callable, Iterable, Iterator, List, Optional
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
            if np.can_cast(self.type.to_pandas_dtype(), np.intp):
                return None, self
            self = self.dictionary_encode()
        return self.dictionary, self.indices

    def arggroupby(self) -> tuple:
        dictionary, array = Chunk.encode(self)
        keys, indices = arggroupby(array)
        return (dictionary.take(keys) if dictionary else keys), indices

    def argunique(self, reverse=False) -> pa.IntegerArray:
        _, array = Chunk.encode(self)
        indices = argunique(array[::-1] if reverse else array)
        return pc.subtract(pa.scalar(len(array) - 1), indices) if reverse else indices

    def equal(self, value) -> np.ndarray:
        if not isinstance(self, pa.DictionaryArray):
            return np.equal(self, value)
        (indices,) = np.nonzero(np.equal(self.dictionary, value))
        return np.equal(self.indices, *indices) if len(indices) else np.full(len(self), False)

    def not_equal(self, value) -> np.ndarray:
        if not isinstance(self, pa.DictionaryArray):
            return np.not_equal(self, value)
        return ~Chunk.equal(self, value)

    def isin(self, values, invert=False) -> np.ndarray:
        if not isinstance(self, pa.DictionaryArray):
            return np.isin(self, values, invert=invert)
        (indices,) = np.nonzero(np.isin(self.dictionary, values))
        return np.isin(self.indices, indices, invert=invert)


class Column(pa.ChunkedArray):
    """Chunked array interface as a namespace of functions."""

    threader = futures.ThreadPoolExecutor(pa.cpu_count())

    def map(func: Callable, *arrays: pa.ChunkedArray) -> Iterator:
        return Column.threader.map(func, *(arr.iterchunks() for arr in arrays))

    def reduce(func: Callable, arrays: Iterable[pa.ChunkedArray]) -> pa.ChunkedArray:
        return pa.chunked_array(Column.map(lambda *chs: functools.reduce(func, chs), *arrays))

    def predicate(func=np.logical_and, **query):
        """Return predicate ufunc by combining operators, by default intersecting."""
        ufuncs = [rpartial(getattr(Chunk, op, getattr(np, op)), query[op]) for op in query]
        if not ufuncs:
            return np.asarray
        return lambda ch: functools.reduce(func, (ufunc(ch) for ufunc in ufuncs))

    def mask(self, predicate: Callable = np.asarray) -> pa.ChunkedArray:
        """Return boolean mask array by applying predicate."""
        return pa.chunked_array(Column.map(lambda ch: np.asarray(predicate(ch), bool), self))

    def equal(self, value) -> pa.ChunkedArray:
        """Return boolean mask array which matches scalar value."""
        return Column.mask(self, rpartial(Chunk.equal, value))

    def not_equal(self, value) -> pa.ChunkedArray:
        """Return boolean mask array which doesn't match scalar value."""
        return Column.mask(self, rpartial(Chunk.not_equal, value))

    def isin(self, values, invert=False) -> pa.ChunkedArray:
        """Return boolean mask array which matches any value."""
        return Column.mask(self, rpartial(Chunk.isin, values, invert))

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

    def sum(self, exp: int = 1):
        """Return sum of the values, with optional exponentiation."""
        if exp == 1:
            return pc.sum(self).as_py()
        return sum(Column.map(lambda ch: np.nansum(np.power(ch, exp)).item(), self))

    def min(self):
        """Return min of the values."""
        value = min(Column.map(np.nanmin, self))
        return value.item() if isinstance(value, np.generic) else value

    def max(self):
        """Return max of the values."""
        value = max(Column.map(np.nanmax, self))
        return value.item() if isinstance(value, np.generic) else value

    def any(self, predicate: Callable = np.asarray) -> bool:
        """Return whether any value evaluates to True."""
        return any(np.any(predicate(chunk)) for chunk in self.iterchunks())

    def all(self, predicate: Callable = np.asarray) -> bool:
        """Return whether all values evaluate to True."""
        return all(np.all(predicate(chunk)) for chunk in self.iterchunks())

    def count(self, value) -> int:
        """Return number of occurrences of value.

        Booleans are optimized and can be used regardless of type.
        """
        if value is None:
            return self.null_count
        if not isinstance(value, bool):
            self, value = Column.equal(self, value), True
        count = sum(Column.map(np.count_nonzero, self))
        return count if value else (len(self) - count - self.null_count)

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

    threader = futures.ThreadPoolExecutor(pa.cpu_count())

    def apply(self, func: Callable = None, **funcs: Callable) -> dict:
        """Apply a function to all, or selected, columns."""
        if func is not None:
            funcs = dict(dict.fromkeys(self.column_names, func), **funcs)
        return dict(Table.threader.map(lambda name: (name, funcs[name](self[name])), funcs))

    def mask(self, func=np.logical_and, **predicates: Callable) -> pa.ChunkedArray:
        """Return boolean mask array by applying predicates to columns and reducing."""
        columns = [self[name] for name in predicates]
        return Column.reduce(func, Table.threader.map(Column.mask, columns, predicates.values()))

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

    def isin(self, name: str, *values) -> pa.Table:
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
        func = lambda col: pa.concat_arrays(Column.map(pa.Array.take, col, indices))
        return pa.Table.from_pydict(Table.apply(self, func))

    def group(self, name: str, reverse=False) -> Iterator[pa.Table]:
        """Generate tables grouped by column."""
        num_chunks = Table.num_chunks(self)
        if num_chunks is None:
            self, num_chunks = self.combine_chunks(), 1
        if num_chunks == 1:
            _, groups = Chunk.arggroupby(self[name].chunk(0))
            for group in groups[::-1] if reverse else groups:
                yield self.take(group.values)
        else:
            groups = Column.arggroupby(self[name]).values()
            for indices in list(groups)[::-1] if reverse else groups:
                yield Table.take_chunks(self, indices)

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

    def grouped(self, *names: str, reverse=False, length: int = None) -> List[pa.Table]:
        tables = [self]
        for name in names:
            groups = (Table.group(table, name, reverse) for table in tables)
            tables = list(itertools.islice(itertools.chain.from_iterable(groups), length))
        return tables

    def matched(self, func: Callable, *names: str):
        for name in names:
            self = self.filter(Column.equal(self[name], func(self[name])))
        return self

    def filtered(self, predicates: dict, invert=False) -> pa.Table:
        if not predicates:
            return self
        mask = Table.mask(self, **predicates)
        return self.filter(Column.mask(mask, np.invert) if invert else mask)
