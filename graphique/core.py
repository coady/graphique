import bisect
import collections
import contextlib
import functools
import itertools
import json
from concurrent import futures
from typing import Callable, Iterator, List, Optional
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

    def equal(self: pa.DictionaryArray, value, invert=False) -> pa.Array:
        (indices,) = np.nonzero(pc.equal(self.dictionary, pa.scalar(value, self.type.value_type)))
        func = pc.not_equal if invert else pc.equal
        return func(self.indices, *pa.array(indices or [-1], self.indices.type))

    def is_in(self, values, invert=False) -> np.ndarray:
        if not isinstance(self, pa.DictionaryArray):
            return np.isin(self, values, invert=invert)
        (indices,) = np.nonzero(np.isin(self.dictionary, values))
        return np.isin(self.indices, indices, invert=invert)


class Column(pa.ChunkedArray):
    """Chunked array interface as a namespace of functions."""

    threader = futures.ThreadPoolExecutor(pa.cpu_count())

    def map(func: Callable, *arrays: pa.ChunkedArray) -> Iterator:
        return Column.threader.map(func, *(arr.iterchunks() for arr in arrays))

    def mask(self, func='and', **query) -> pa.ChunkedArray:
        """Return boolean mask array which matches query predicates."""
        ops = {'equal', 'not_equal', 'isin'}.intersection(query)
        masks = [getattr(Column, op)(self, query.pop(op)) for op in ops]
        if (not masks or query) and isinstance(self.type, pa.DictionaryType):
            self = self.cast(self.type.value_type)
        for op in query:
            if op == 'match_substring':
                masks.append(getattr(pc, op)(self, query[op]))
            elif op.startswith('utf8_'):
                masks.append(Column.mask(getattr(pc, op)(self), func, **query[op]))
            else:
                masks.append(getattr(pc, op)(self, pa.scalar(query[op], self.type)))
        if masks:
            return functools.reduce(lambda *args: pc.call_function(func, args), masks)
        with contextlib.suppress(NotImplementedError):
            self = pc.call_function('binary_length', [self])
        with contextlib.suppress(NotImplementedError):
            self = self.cast(pa.bool_())
        return self

    def equal(self, value, invert=False) -> pa.ChunkedArray:
        """Return boolean mask array which matches scalar value."""
        if value is None:
            return pc.is_null(self)
        if isinstance(self.type, pa.DictionaryType):
            return pa.chunked_array(Column.map(rpartial(Chunk.equal, value, invert), self))
        return (pc.not_equal if invert else pc.equal)(self, pa.scalar(value, self.type))

    def not_equal(self, value) -> pa.ChunkedArray:
        """Return boolean mask array which doesn't match scalar value."""
        if value is None:
            return pc.is_valid(self)
        return Column.equal(self, value, invert=True)

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

    def any(self) -> bool:
        """Return whether any value evaluates to True."""
        return any(map(np.any, self.iterchunks()))

    def all(self) -> bool:
        """Return whether all values evaluate to True."""
        return all(map(np.all, self.iterchunks()))

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

    def filtered(self, queries: dict, invert=False) -> pa.Table:
        masks = []
        for name in queries:
            proj = queries[name].pop('project', {})
            if proj:
                masks += [getattr(pc, op)(self[name], self[proj[op]]) for op in proj]
            if queries[name]:
                masks.append(Column.mask(self[name], **queries[name]))
        if not masks:
            return self
        mask = functools.reduce(lambda *args: pc.call_function('and', args), masks)
        return self.filter(pc.call_function('invert', [mask]) if invert else mask)
