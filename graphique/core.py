import bisect
import json
from concurrent import futures
from typing import Callable, Iterator
import pyarrow as pa

threader = futures.ThreadPoolExecutor()
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


class Array(pa.Array):
    """Chunked array interface as a namespace of functions."""

    def subtype(self) -> type:
        base = type_map[self.type]
        return type(base.__name__, (Compare, base), {})

    def sum(self):
        """Return sum of the values."""
        return sum(scalar.as_py() for scalar in threader.map(pa.Array.sum, self.chunks))

    def range(self, lower=None, upper=None, include_lower=True, include_upper=False) -> slice:
        """Return start, stop indices within range, by default a half-open interval.

        Assumes the array is sorted.
        """
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


class Table(pa.Table):
    """Table interface as a namespace of functions."""

    def map(self, func: Callable) -> dict:
        return dict(zip(self.column_names, threader.map(func, self.columns)))

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

    def unique(self) -> dict:
        """Return mapping to unique arrays."""
        return Table.map(self, pa.ChunkedArray.unique)

    def sum(self) -> dict:
        """Return mapping of sums."""
        return Table.map(self, Array.sum)

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
