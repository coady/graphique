import bisect
import json
import pyarrow as pa

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

    def sum(self):
        """Return sum of the values."""
        return sum(chunk.sum().as_py() for chunk in self.chunks)

    def range(self, lower=None, upper=None, include_lower=True, include_upper=False) -> pa.Array:
        """Return start, stop indices within range, by default a half-open interval.

        Assumes the array is sorted.
        """
        if lower is upper is None:
            return 0, None
        (base,) = {type(lower), type(upper)} - {type(None)}
        cls = type(base.__name__, (Compare, base), {})
        method = bisect.bisect_left if include_lower else bisect.bisect_right
        start = 0 if lower is None else method(self, cls(lower))
        method = bisect.bisect_right if include_upper else bisect.bisect_left
        stop = None if upper is None else method(self, cls(upper), start)
        return start, stop


class Table(pa.Table):
    """Table utilities as a namespace of functions."""

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
        return {name: self[name].null_count for name in self.column_names}

    def unique(self) -> dict:
        """Return mapping to unique arrays."""
        return {name: self[name].unique() for name in self.column_names}

    def sum(self) -> dict:
        """Return mapping of sums."""
        return {name: Array.sum(self[name]) for name in self.column_names}

    def search(self, name: str, lower=None, upper=None, **includes) -> pa.Table:
        """Return rows within range, by default a half-open interval.

        Assumes the table is sorted by the given field, i.e., indexed.
        """
        start, stop = Array.range(self[name], lower, upper, **includes)
        return self[start:stop]
