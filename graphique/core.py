import bisect
import collections
import functools
import json
import os
from concurrent import futures
from typing import Callable, Iterator, Union
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


# TODO: reimplement or reuse spector.vector.arggroupby, optimized for ints
def arggroupby(chunk: pa.Array) -> Iterator[tuple]:
    """Generate unique keys with corresponding index arrays."""
    values = np.asarray(getattr(chunk, 'indices', chunk))
    keys, counts = np.unique(values, return_counts=True)
    dictionary = getattr(chunk, 'dictionary', None)
    keys = dictionary.take(pa.array(keys)).to_pylist() if dictionary else keys.tolist()
    return zip(keys, np.split(np.argsort(values), np.cumsum(counts)))


class Compare:
    """Comparable mixin for bisection search."""

    def __lt__(self, other):
        return super().__lt__(other.as_py())

    def __gt__(self, other):
        return super().__gt__(other.as_py())


class Array(pa.Array):
    """Chunked array interface as a namespace of functions."""

    threader = futures.ThreadPoolExecutor(max_workers)

    def map(self, func: Callable, *iterables) -> Iterator:
        return Array.threader.map(func, self.chunks, *iterables)

    def subtype(self) -> type:
        base = type_map[self.type]
        return type(base.__name__, (Compare, base), {})

    def dictionary(self) -> pa.Array:
        if isinstance(self.type, pa.DictionaryType):
            for chunk in self.chunks:
                return chunk.dictionary
        return self[:0]

    def mask(self, predicate: Callable) -> pa.ChunkedArray:
        """Return boolean mask array by applying predicate."""
        dictionary = Array.dictionary(self)
        if not dictionary:
            return pa.chunked_array(Array.map(self, lambda ch: predicate(np.asarray(ch))))
        (indices,) = np.nonzero(predicate(np.asarray(dictionary)))
        return pa.chunked_array(Array.map(self, lambda ch: np.isin(ch.indices, indices)))

    def filter(self, mask: pa.ChunkedArray) -> pa.ChunkedArray:
        """Return array filtered by a boolean mask."""
        return pa.chunked_array(Array.map(self, pa.Array.filter, mask.chunks))

    def take(self, indices: pa.ChunkedArray) -> pa.ChunkedArray:
        """Return array with indexed elements."""
        return pa.chunked_array(Array.map(self, pa.Array.take, indices.chunks))

    def arggroupby(self) -> dict:
        """Return mapping of unique keys to corresponding index arrays."""
        empty = np.full(0, 0)
        result = collections.defaultdict(lambda: [empty] * len(self.chunks))  # type: dict
        if self.type == pa.string():
            self = self.dictionary_encode()
        for index, items in enumerate(Array.map(self, arggroupby)):
            for key, values in items:
                result[key][index] = values
        return {key: pa.chunked_array(result[key]) for key in result}

    def sum(self):
        """Return sum of the values."""
        return sum(scalar.as_py() for scalar in Array.map(self, pa.Array.sum))

    def min(self):
        """Return min of the values."""
        dictionary = Array.dictionary(self)
        return np.min(dictionary) if dictionary else min(Array.map(self, np.min))

    def max(self):
        """Return max of the values."""
        dictionary = Array.dictionary(self)
        return np.max(dictionary) if dictionary else max(Array.map(self, np.max))

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

    def unique(self, counts=False) -> Union[pa.Array, tuple]:
        """Return array of unique values, optionally with counts."""
        if not Array.dictionary(self):
            if not counts:
                return self.unique()
            self = self.dictionary_encode()
        dictionary = Array.dictionary(self)
        if not counts:
            return dictionary
        size = len(dictionary)
        bins = Array.map(self, lambda arr: np.bincount(arr.indices, minlength=size))
        counts = functools.reduce(np.ndarray.__iadd__, bins, np.full(size, 0))
        return dictionary, pa.array(counts)


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
        return self.from_arrays(list(map(self.column, names)), names)

    def null_count(self) -> dict:
        """Return count of null values."""
        return Table.map(self, pa.ChunkedArray.null_count.__get__)

    def unique(self, counts=False) -> dict:
        """Return mapping to unique arrays."""
        return Table.map(self, functools.partial(Array.unique, counts=bool(counts)))

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
            data = Table.map(self, lambda col: Array.filter(col, mask))
            self = self.from_arrays(list(data.values()), list(data))
        return self
