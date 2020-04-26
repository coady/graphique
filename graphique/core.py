import bisect
import collections
import functools
import json
import os
from concurrent import futures
from typing import Callable, Iterable, Iterator
import numpy as np
import pyarrow as pa

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
    """Comparable mixin for bisection search."""

    def __lt__(self, other):
        return super().__lt__(other.as_py())

    def __gt__(self, other):
        return super().__gt__(other.as_py())


class Chunk(pa.Array):
    # TODO: reimplement or reuse spector.vector.arggroupby, optimized for ints
    def arggroupby(self) -> Iterator[tuple]:
        dictionary = None
        if isinstance(self, pa.DictionaryArray):
            self, dictionary = self.indices, self.dictionary
        vc = self.value_counts()
        indices = pa.array(np.argsort(vc.field('values')))
        keys = vc.field('values').take(indices)
        sections = np.split(np.argsort(self), np.cumsum(vc.field('counts').take(indices)))
        return zip((dictionary.take(keys) if dictionary else keys).to_pylist(), sections)

    def value_counts(self):
        vc = self.indices.value_counts()
        return self.dictionary.take(vc.field('values')), vc.field('counts')

    def mask(self, predicate: Callable) -> pa.Array:
        if not isinstance(self, pa.DictionaryArray):
            return pa.array(predicate(np.asarray(self)))
        (indices,) = np.nonzero(predicate(np.asarray(self.dictionary)))
        return pa.array(np.isin(self.indices, indices))


class Array(pa.ChunkedArray):
    """Chunked array interface as a namespace of functions."""

    threader = futures.ThreadPoolExecutor(max_workers)

    def map(self, func: Callable, *iterables: Iterable) -> Iterator:
        return Array.threader.map(func, self.iterchunks(), *iterables)

    def subtype(self) -> type:
        base = type_map[self.type]
        return type(base.__name__, (Compare, base), {})

    def mask(self, predicate: Callable) -> pa.ChunkedArray:
        """Return boolean mask array by applying predicate."""
        return pa.chunked_array(Array.map(self, lambda ch: Chunk.mask(ch, predicate)))

    def take(self, indices: pa.ChunkedArray) -> pa.ChunkedArray:
        """Return array with indexed elements."""
        return pa.chunked_array(Array.map(self, pa.Array.take, indices.chunks))

    def arggroupby(self) -> dict:
        """Return mapping of unique keys to corresponding index arrays.

        Indices are chunked, and not offset, pending release of `ChunkedArray.take`.
        """
        empty = np.full(0, 0)
        result = collections.defaultdict(lambda: [empty] * self.num_chunks)  # type: dict
        if self.type == pa.string():
            self = self.dictionary_encode()
        for index, items in enumerate(Array.map(self, Chunk.arggroupby)):
            for key, values in items:
                result[key][index] = values
        return {key: pa.chunked_array(result[key]) for key in result}

    def sum(self):
        """Return sum of the values."""
        return sum(scalar.as_py() for scalar in Array.map(self, pa.Array.sum))

    def min(self):
        """Return min of the values."""
        return min(Array.map(self, np.min))

    def max(self):
        """Return max of the values."""
        return max(Array.map(self, np.max))

    def argmin(self):
        """Return first index of the minimum value."""
        values = list(Array.map(self, np.min))
        index = np.argmin(values)
        return int(np.argmin(self.chunk(index))) + sum(map(len, self.chunks[:index]))

    def argmax(self):
        """Return first index of the maximum value."""
        values = list(Array.map(self, np.max))
        index = np.argmax(values)
        return int(np.argmax(self.chunk(index))) + sum(map(len, self.chunks[:index]))

    def range(self, lower=None, upper=None, include_lower=True, include_upper=False) -> slice:
        """Return slice within range from a sorted array, by default a half-open interval."""
        cls = Array.subtype(self)
        method = bisect.bisect_left if include_lower else bisect.bisect_right
        start = 0 if lower is None else method(self, cls(lower))
        method = bisect.bisect_right if include_upper else bisect.bisect_left
        stop = None if upper is None else method(self, cls(upper), start)
        return slice(start, stop)

    def find(self, *values) -> Iterator[slice]:
        """Generate slices of matching rows from a sorted array."""
        stop = 0
        for value in map(Array.subtype(self), sorted(values)):
            start = bisect.bisect_left(self, value, stop)
            stop = bisect.bisect_right(self, value, start)
            yield slice(start, stop)

    def unique(self) -> pa.Array:
        """Return array of unique values with dictionary support."""
        if not isinstance(self.type, pa.DictionaryType):
            return self.unique()
        chunks = Array.map(self, lambda ch: ch.dictionary.take(ch.indices.unique()))
        return pa.chunked_array(chunks).unique()

    def value_counts(self) -> tuple:
        """Return arrays of unique values and counts with dictionary support."""
        if not isinstance(self.type, pa.DictionaryType):
            vc = self.value_counts()
            return vc.field('values'), vc.field('counts')  # type: ignore
        values, counts = zip(*Array.map(self, Chunk.value_counts))
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
        return Table.map(self, Array.value_counts if counts else Array.unique)

    def sum(self) -> dict:
        """Return mapping of sums."""
        return Table.map(self, Array.sum)

    def min(self) -> dict:
        """Return mapping of min values."""
        return Table.map(self, Array.min)

    def max(self) -> dict:
        """Return mapping of max values."""
        return Table.map(self, Array.max)

    def range(self, name: str, lower=None, upper=None, **includes) -> pa.Table:
        """Return rows within range, by default a half-open interval.

        Assumes the table is sorted by the column name, i.e., indexed.
        """
        return self[Array.range(self[name], lower, upper, **includes)]

    def isin(self, name: str, *values) -> pa.Table:
        """Return rows which matches one of the values.

        Assumes the table is sorted by the column name, i.e., indexed.
        """
        slices = list(Array.find(self[name], *values)) or [slice(0)]
        return pa.concat_tables(self[slc] for slc in slices)

    def filter(self, **predicates: Callable) -> pa.Table:
        """Return table filtered by applying predicates to columns."""
        for name in predicates:
            mask = Array.mask(self[name], predicates[name])
            data = Table.map(self, functools.partial(pa.ChunkedArray.filter, mask=mask))
            self = self.from_pydict(data)
        return self
