import bisect
import collections
import json
import os
from concurrent import futures
from functools import partial
from typing import Callable, Iterable, Iterator
import numpy as np
import pyarrow as pa
from .arrayed import arggroupby  # type: ignore

max_workers = os.cpu_count() or 1  # same default as ProcessPoolExecutor
type_map = {
    pa.bool_(): bool,
    pa.float16(): float,
    pa.float32(): float,
    pa.float64(): float,
    pa.int8(): int,
    pa.int16(): int,
    pa.int32(): int,
    pa.int64(): int,
    pa.string(): str,
}


class Compare:
    """Comparable wrapper for bisection search."""

    __slots__ = ('value',)

    def __init__(self, value):
        self.value = value

    def __lt__(self, other):
        return self.value < other.as_py()

    def __gt__(self, other):
        return self.value > other.as_py()


class Chunk:
    def arggroupby(self) -> Iterator[tuple]:
        dictionary = None
        if isinstance(self, pa.DictionaryArray):
            self, dictionary = self.indices, self.dictionary  # type: ignore
        try:
            keys, sections = arggroupby(self)
        except TypeError:  # fallback to sorting
            values, counts = self.value_counts().flatten()
            indices = pa.array(np.argsort(values))
            keys = values.take(indices)
            sections = np.split(np.argsort(self), np.cumsum(counts.take(indices)))
        return zip((dictionary.take(keys) if dictionary else keys).to_pylist(), sections)

    def value_counts(self):
        values, counts = self.indices.value_counts().flatten()
        return self.dictionary.take(values), counts

    def equal(self, value) -> np.ndarray:
        if not isinstance(self, pa.DictionaryArray):
            return np.equal(self, value)
        (indices,) = np.nonzero(np.equal(self.dictionary, value))
        return np.equal(self.indices, *indices) if len(indices) else np.full(len(self), False)


class Column(pa.ChunkedArray):
    """Chunked array interface as a namespace of functions."""

    threader = futures.ThreadPoolExecutor(max_workers)

    def map(self, func: Callable, *iterables: Iterable) -> Iterator:
        return Column.threader.map(func, self.iterchunks(), *iterables)

    def mask(self, predicate: Callable = np.asarray) -> pa.ChunkedArray:
        """Return boolean mask array by applying predicate."""
        return pa.chunked_array(Column.map(self, lambda ch: np.asarray(predicate(ch), bool)))

    def equal(self, value) -> pa.ChunkedArray:
        """Return boolean mask array which matches scalar value."""
        return pa.chunked_array(Column.map(self, partial(Chunk.equal, value=value)))

    def take(self, indices: pa.ChunkedArray) -> pa.ChunkedArray:
        """Return array with indexed elements."""
        return pa.chunked_array(Column.map(self, pa.Array.take, indices.chunks))

    def arggroupby(self) -> dict:
        """Return mapping of unique keys to corresponding index arrays.

        Indices are chunked, and not offset, pending release of `ChunkedArray.take`.
        """
        empty = np.full(0, 0)
        result = collections.defaultdict(lambda: [empty] * self.num_chunks)  # type: dict
        if self.type == pa.string():
            self = self.dictionary_encode()
        for index, items in enumerate(Column.map(self, Chunk.arggroupby)):
            for key, values in items:
                result[key][index] = values
        return {key: pa.chunked_array(result[key]) for key in result}

    def sum(self):
        """Return sum of the values."""
        return sum(Column.map(self, lambda ch: ch.sum().as_py()))

    def min(self):
        """Return min of the values."""
        return min(Column.map(self, np.min))

    def max(self):
        """Return max of the values."""
        return max(Column.map(self, np.max))

    def any(self, predicate: Callable = np.asarray) -> bool:
        """Return whether any value evaluates to True."""
        return any(np.any(predicate(chunk)) for chunk in self.iterchunks())

    def all(self, predicate: Callable = np.asarray) -> bool:
        """Return whether all values evaluate to True."""
        return all(np.all(predicate(chunk)) for chunk in self.iterchunks())

    def contains(self, value) -> bool:
        """Return whether value is in array."""
        return Column.any(self, partial(Chunk.equal, value=value))

    def count(self, value) -> int:
        """Return number of occurrences of value.

        Booleans are optimized and can be used regardless of type.
        """
        if value is None:
            return self.null_count
        if not isinstance(value, bool):
            self = Column.equal(self, value)
        return sum(Column.map(self, np.count_nonzero))

    def where(self, index, value):
        (indices,) = np.nonzero(Chunk.equal(self.chunk(index), value))
        return int(indices[0]) + sum(map(len, self.chunks[:index]))

    def argmin(self) -> int:
        """Return first index of the minimum value."""
        values = list(Column.map(self, np.min))
        index = np.argmin(values)
        return Column.where(self, index, values[index])

    def argmax(self) -> int:
        """Return first index of the maximum value."""
        values = list(Column.map(self, np.max))
        index = np.argmax(values)
        return Column.where(self, index, values[index])

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

    def unique(self) -> pa.Array:
        """Return array of unique values with dictionary support."""
        if not isinstance(self.type, pa.DictionaryType):
            return self.unique()
        chunks = Column.map(self, lambda ch: ch.dictionary.take(ch.indices.unique()))
        return pa.chunked_array(chunks).unique()

    def value_counts(self) -> tuple:
        """Return arrays of unique values and counts with dictionary support."""
        if not isinstance(self.type, pa.DictionaryType):
            return self.value_counts().flatten()  # type: ignore
        values, counts = zip(*Column.map(self, Chunk.value_counts))
        values, indices = np.unique(np.concatenate(values), return_inverse=True)
        counts = np.bincount(indices, weights=np.concatenate(counts)).astype(int)
        return pa.array(values), pa.array(counts)


class Table(pa.Table):
    """Table interface as a namespace of functions."""

    threader = futures.ThreadPoolExecutor(max_workers)

    def map(self, func: Callable) -> dict:
        return dict(zip(self.column_names, Table.threader.map(func, self.columns)))

    def index(self) -> list:
        """Return index column names from pandas metadata."""
        return json.loads(self.schema.metadata.get(b'pandas', b'{}')).get('index_columns', [])

    def types(self) -> dict:
        """Return mapping of column types."""
        return {name: type_map[self[name].type] for name in self.column_names}

    def select(self, *names: str) -> pa.Table:
        """Return table with selected columns."""
        return self.from_pydict({name: self[name] for name in names})

    def null_count(self) -> dict:
        """Return count of null values."""
        return Table.map(self, pa.ChunkedArray.null_count.__get__)

    def unique(self, counts=False) -> dict:
        """Return mapping to unique arrays."""
        return Table.map(self, Column.value_counts if counts else Column.unique)

    def sum(self) -> dict:
        """Return mapping of sums."""
        return Table.map(self, Column.sum)

    def min(self) -> dict:
        """Return mapping of min values."""
        return Table.map(self, Column.min)

    def max(self) -> dict:
        """Return mapping of max values."""
        return Table.map(self, Column.max)

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

    def filter(self, **predicates: Callable) -> pa.Table:
        """Return table filtered by applying predicates to columns."""
        for name in predicates:
            mask = Column.mask(self[name], predicates[name])
            data = Table.map(self, partial(pa.ChunkedArray.filter, mask=mask))
            self = self.from_pydict(data)
        return self
