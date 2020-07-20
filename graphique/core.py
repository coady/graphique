import bisect
import collections
import functools
import itertools
import json
from concurrent import futures
from typing import Callable, Iterable, Iterator
import numpy as np
import pyarrow as pa
from .arrayed import arggroupby, argunique, asiarray  # type: ignore


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


def argsplit(dictionary: pa.Array, indices: np.ndarray, *arrays: np.ndarray) -> dict:
    """Return nested groups of indices given an argsort index.

    :param dictionary: an original set of indices to enable working with existing chunks
    :param indices: indices which would sort the arrays
    :param arrays: values that will be grouped in order
    """
    if not arrays:
        return dictionary.take(pa.array(indices))
    values = np.take(arrays[0], np.take(dictionary, indices))
    steps = np.nonzero(np.not_equal(values[1:], values[:-1]))[0] + 1
    keys = np.take(values, np.concatenate([[0], steps]))
    groups = [argsplit(dictionary, idx, *arrays[1:]) for idx in np.split(indices, steps)]
    return dict(zip(keys.tolist(), groups))


def flatten(tree: dict, reverse=False) -> Iterator:
    """Generate leaf nodes sorted by keys."""
    if isinstance(tree, dict):
        for key in sorted(tree, reverse=reverse):
            yield from flatten(tree[key], reverse=reverse)
    else:
        yield tree


class Chunk:
    def arggroupby(self) -> Iterator[tuple]:
        dictionary = None
        if isinstance(self, pa.DictionaryArray):
            self, dictionary = self.indices, self.dictionary  # type: ignore
        try:
            keys, sections = arggroupby(self)
        except TypeError:  # fallback to sorting
            values, counts = self.value_counts().flatten()  # type: ignore
            indices = pa.array(np.argsort(values))
            keys = values.take(indices)
            sections = np.split(np.argsort(self, kind='stable'), np.cumsum(counts.take(indices)))
        return zip((dictionary.take(keys) if dictionary else keys).to_pylist(), sections)

    def argunique(self, reverse=False) -> np.ndarray:
        if isinstance(self, pa.DictionaryArray):
            self = self.indices
        try:
            indices = argunique(pa.array(asiarray(self)[::-1]) if reverse else self)
        except TypeError:  # fallback to sorting
            _, indices = np.unique(np.asarray(self)[::-1] if reverse else self, return_index=True)
        return (len(self) - 1 - indices) if reverse else indices  # type: ignore

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
        """Return mapping of unique keys to corresponding index arrays."""
        empty = np.full(0, 0)
        result = collections.defaultdict(lambda: [empty] * self.num_chunks)  # type: dict
        if self.type == pa.string():
            self = self.dictionary_encode()
        offset = 0
        for index, items in enumerate(Column.map(Chunk.arggroupby, self)):
            for key, values in items:
                result[key][index] = values + offset
            offset += len(self.chunk(index))
        return {key: pa.array(np.concatenate(result[key])) for key in result}

    def offset(self, chunks):
        offsets = itertools.accumulate(map(len, self.iterchunks()))
        return np.concatenate(chunks[:1] + list(map(np.add, chunks[1:], offsets)))

    def argunique(self, reverse=False) -> pa.Array:
        """Return index array of first or last occurrences."""
        if self.type == pa.string():
            self = self.dictionary_encode()
        chunks = list(Column.map(rpartial(Chunk.argunique, reverse), self))
        keys = map(pa.Array.take, self.iterchunks(), map(pa.array, chunks))
        indices = Chunk.argunique(pa.concat_arrays(keys), reverse)
        return pa.array(np.take(Column.offset(self, chunks), indices))

    def sort(self, reverse=False, length: int = None) -> pa.Array:
        """Return sorted values, optimized for fixed length."""
        if length is None:
            select = functools.partial(np.sort, kind='stable')  # type: Callable
        elif reverse:
            select = lambda ch: np.partition(ch, -length)[-length:]  # type: ignore
        else:
            select = lambda ch: np.partition(ch, length)[:length]
        values = np.sort(np.concatenate(list(Column.map(select, self))), kind='stable')
        return pa.array((values[::-1] if reverse else values)[:length])

    def argsort(self, reverse=False, length: int = None) -> pa.Array:
        """Return indices which would sort the values, optimized for fixed length."""
        if length is None:
            indices = np.argsort(self, kind='stable')
        else:
            if reverse:
                select = lambda ch: np.argpartition(ch, -length)[-length:]
            else:
                select = lambda ch: np.argpartition(ch, length)[:length]
            chunks = list(Column.map(select, self))
            values = np.concatenate(list(map(np.take, self.iterchunks(), chunks)))
            indices = np.take(Column.offset(self, chunks), np.argsort(values))
        return pa.array((indices[::-1] if reverse else indices)[:length])

    def sum(self, exp: int = 1):
        """Return sum of the values, with optional exponentiation."""
        if exp == 1:
            func = lambda ch: ch.sum().as_py()
        else:
            func = lambda ch: np.nansum(np.power(ch, exp)).item()
        return sum(Column.map(func, self))

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

    def contains(self, value) -> bool:
        """Return whether value is in array."""
        return Column.any(self, rpartial(Chunk.equal, value))

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

    def where(self, index, value):
        (indices,) = np.nonzero(Chunk.equal(self.chunk(index), value))
        return int(indices[0]) + sum(map(len, self.chunks[:index]))

    def argmin(self) -> int:
        """Return first index of the minimum value."""
        values = list(Column.map(np.nanmin, self))
        index = np.argmin(values)
        return Column.where(self, index, values[index])

    def argmax(self) -> int:
        """Return first index of the maximum value."""
        values = list(Column.map(np.nanmax, self))
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

    def arggroupby(self, *names: str) -> dict:
        """Generate keys and indices from grouping by columns."""
        groups = Column.arggroupby(self[names[0]])
        arrays = [np.asarray(self[name]) for name in names[1:]]
        if arrays:
            for key, group in groups.items():
                indices = np.lexsort([np.take(array, group) for array in arrays[::-1]])
                groups[key] = argsplit(group, indices, *arrays)
        return groups

    def argunique(self, *names: str, reverse=False) -> pa.Array:
        """Return index array of first or last occurrences from grouping by columns.

        Optimized for a single column with fixed length.
        """
        if len(names) <= 1:
            return Column.argunique(self[names[0]], reverse)
        values = pa.concat_arrays(self[names[-1]].iterchunks())
        return pa.concat_arrays(
            indices.take(pa.array(Chunk.argunique(values.take(indices), reverse)))
            for indices in flatten(Table.arggroupby(self, *names[:-1]), reverse)
        )

    def argsort(self, *names: str, reverse=False, length: int = None) -> pa.Array:
        """Return indices which would sort the table by given columns.

        Optimized for a single column with fixed length.
        """
        if length is None or len(names) > 1:
            indices = np.lexsort([self[name] for name in reversed(names)])
            return pa.array((indices[::-1] if reverse else indices)[:length])
        column = self.column(*names)
        if length > 1:
            return Column.argsort(column, reverse, length)
        select = Column.argmax if reverse else Column.argmin
        return pa.array([select(column)])

    def matched(self, func: Callable, *names: str):
        for name in names:
            self = self.filter(Column.equal(self[name], func(self[name])))
        return self

    def filtered(self, predicates: dict, invert=False) -> pa.Table:
        if not predicates:
            return self
        mask = Table.mask(self, **predicates)
        return self.filter(Column.mask(mask, np.invert) if invert else mask)
