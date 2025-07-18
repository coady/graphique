"""
Core utilities that add pandas-esque features to arrow arrays and tables.

Arrow forbids subclassing, so the classes are for logical grouping.
Their methods are called as functions.
"""

import bisect
import contextlib
import functools
import inspect
import itertools
import operator
import json
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import TypeAlias, get_type_hints
import ibis.backends.duckdb
import pyarrow as pa
import pyarrow.acero as ac
import pyarrow.compute as pc
import pyarrow.dataset as ds
from typing_extensions import Self


def to_pyarrow(self, expr, **kwargs):
    table = self._to_duckdb_relation(expr, **kwargs).arrow()
    return expr.__pyarrow_result__(table, data_mapper=ibis.formats.pyarrow.PyArrowData)


# shim to avoid pandas dependency: ibis-project/ibis#11430
ibis.backends.duckdb.Backend.to_pyarrow = to_pyarrow
Array: TypeAlias = pa.Array | pa.ChunkedArray
Batch: TypeAlias = pa.RecordBatch | pa.Table
bit_any = functools.partial(functools.reduce, operator.or_)
bit_all = functools.partial(functools.reduce, operator.and_)


class Agg:
    """Aggregation options."""

    option_map = {
        'all': pc.ScalarAggregateOptions,
        'any': pc.ScalarAggregateOptions,
        'approximate_median': pc.ScalarAggregateOptions,
        'count': pc.CountOptions,
        'count_distinct': pc.CountOptions,
        'distinct': pc.CountOptions,
        'first': pc.ScalarAggregateOptions,
        'first_last': pc.ScalarAggregateOptions,
        'last': pc.ScalarAggregateOptions,
        'list': type(None),
        'max': pc.ScalarAggregateOptions,
        'mean': pc.ScalarAggregateOptions,
        'min': pc.ScalarAggregateOptions,
        'min_max': pc.ScalarAggregateOptions,
        'one': type(None),
        'product': pc.ScalarAggregateOptions,
        'stddev': pc.VarianceOptions,
        'sum': pc.ScalarAggregateOptions,
        'tdigest': pc.TDigestOptions,
        'variance': pc.VarianceOptions,
    }
    ordered = {'first', 'last'}

    def __init__(self, name: str, alias: str = '', **options):
        self.name = name
        self.alias = alias or name
        self.options = options

    def func_options(self, func: str) -> pc.FunctionOptions:
        return self.option_map[func.removeprefix('hash_')](**self.options)


@dataclass(frozen=True, slots=True)
class Compare:
    """Comparable wrapper for bisection search."""

    value: object

    def __lt__(self, other):
        return self.value < other.as_py()

    def __gt__(self, other):
        return self.value > other.as_py()


def sort_key(name: str) -> tuple:
    """Parse sort order."""
    return name.lstrip('-'), ('descending' if name.startswith('-') else 'ascending')


def register(func: Callable, kind: str = 'scalar') -> pc.Function:
    """Register user defined function by kind."""
    doc = inspect.getdoc(func)
    doc = {'summary': doc.splitlines()[0], 'description': doc}  # type: ignore
    annotations = dict(get_type_hints(func))
    result = annotations.pop('return')
    with contextlib.suppress(pa.ArrowKeyError):  # apache/arrow#{31611,31612}
        getattr(pc, f'register_{kind}_function')(func, func.__name__, doc, annotations, result)
    return pc.get_function(func.__name__)


def memcolumn(array: Array) -> ibis.Column:
    """Convert pyarrow array to ibis column."""
    return ibis.memtable(pa.table({'_': array}))['_']


@register
def digitize(
    ctx,
    array: pa.float64(),  # type: ignore
    bins: pa.list_(pa.float64()),  # type: ignore
    right: pa.bool_(),  # type: ignore
) -> pa.int64():  # type: ignore
    """Return the indices of the bins to which each value in input array belongs."""
    column = memcolumn(array).bucket(
        bins.values.to_pylist(),
        include_under=True,
        include_over=True,
        closed='right' if right.as_py() else 'left',
    )
    return column.to_pyarrow().combine_chunks().cast('int64')


class ListChunk(pa.lib.BaseListArray):
    def count(self):
        return memcolumn(self).length().to_pyarrow()

    def distinct(self):
        return memcolumn(self).unique().to_pyarrow().combine_chunks()

    def mean(self, **option):
        return memcolumn(self).means().to_pyarrow()

    def sum(self):
        return memcolumn(self).sums().to_pyarrow()

    def first(self) -> pa.Array:
        """first value of each list scalar"""
        return memcolumn(self)[0].to_pyarrow()

    def last(self) -> pa.Array:
        """last value of each list scalar"""
        return memcolumn(self)[-1].to_pyarrow()

    def min(self, **options) -> pa.Array:
        """min value of each list scalar"""
        return memcolumn(self).mins().to_pyarrow()

    def max(self, **options) -> pa.Array:
        """max value of each list scalar"""
        return memcolumn(self).maxs().to_pyarrow()

    def mode(self, **options) -> pa.Array:
        """modes of each list scalar"""
        return memcolumn(self).modes().to_pyarrow().combine_chunks()

    def index(self, value) -> pa.Array:
        """index for first occurrence of each list scalar"""
        return memcolumn(self).index(value).to_pyarrow().combine_chunks()

    @register
    def list_all(ctx, self: pa.list_(pa.bool_())) -> pa.bool_():  # type: ignore
        """Test whether all elements in a boolean array evaluate to true."""
        return memcolumn(self).alls().to_pyarrow().combine_chunks()

    @register
    def list_any(ctx, self: pa.list_(pa.bool_())) -> pa.bool_():  # type: ignore
        """Test whether any element in a boolean array evaluates to true."""
        return memcolumn(self).anys().to_pyarrow().combine_chunks()


class Column(pa.ChunkedArray):
    """Chunked array interface as a namespace of functions."""

    def is_list_type(self):
        funcs = pa.types.is_list, pa.types.is_large_list, pa.types.is_fixed_size_list
        return any(func(self.type) for func in funcs)

    def call_indices(self, func: Callable) -> Array:
        if not pa.types.is_dictionary(self.type):
            return func(self)
        array = self.combine_chunks()
        return pa.DictionaryArray.from_arrays(func(array.indices), array.dictionary)

    def fill_null_backward(self) -> Array:
        """`fill_null_backward` with dictionary support."""
        return Column.call_indices(self, pc.fill_null_backward)

    def fill_null_forward(self) -> Array:
        """`fill_null_forward` with dictionary support."""
        return Column.call_indices(self, pc.fill_null_forward)

    def fill_null(self, value) -> pa.ChunkedArray:
        """Optimized `fill_null` to check `null_count`."""
        return self.fill_null(value) if self.null_count else self

    def sort_values(self) -> Array:
        if not pa.types.is_dictionary(self.type):
            return self
        array = self if isinstance(self, pa.Array) else self.combine_chunks()
        return pc.rank(array.dictionary).take(array.indices)

    def pairwise_diff(self, period: int = 1) -> Array:
        """`pairwise_diff` with chunked array support."""
        return pc.pairwise_diff(self.combine_chunks(), period)

    def diff(self, func: Callable = pc.subtract, period: int = 1) -> Array:
        """Compute first order difference of an array.

        Unlike `pairwise_diff`, does not return leading nulls.
        """
        return func(self[period:], self[:-period])

    def run_offsets(self, predicate: Callable = pc.not_equal, *args) -> pa.IntegerArray:
        """Run-end encode array with leading zero, suitable for list offsets.

        Args:
            predicate: binary function applied to adjacent values
            *args: apply binary function to scalar, using `subtract` as the difference function
        """
        ends = [pa.array([True])]
        mask = predicate(Column.diff(self), *args) if args else Column.diff(self, predicate)
        return pc.indices_nonzero(pa.chunked_array(ends + mask.chunks + ends))

    def index(self, value, start=0, end=None) -> int:
        """Return the first index of a value."""
        with contextlib.suppress(NotImplementedError):
            return self.index(value, start, end).as_py()  # type: ignore
        offset = start
        for chunk in self[start:end].iterchunks():
            index = chunk.dictionary.index(value).as_py()
            if index >= 0:
                index = chunk.indices.index(index).as_py()
            if index >= 0:
                return offset + index
            offset += len(chunk)
        return -1

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

    def map_batch(self, func: Callable, *args, **kwargs) -> pa.Table:
        return pa.Table.from_batches(func(batch, *args, **kwargs) for batch in self.to_batches())

    def columns(self) -> dict:
        """Return columns as a dictionary."""
        return dict(zip(self.schema.names, self))

    def union(*tables: Batch) -> Batch:
        """Return table with union of columns."""
        columns: dict = {}
        for table in tables:
            columns |= Table.columns(table)
        return type(tables[0]).from_pydict(columns)

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
        return pa.concat_tables([self[: slc.start], self[slc.stop :]])

    def from_offsets(self, offsets: pa.IntegerArray, mask=None) -> pa.RecordBatch:
        """Return record batch with columns converted into list columns."""
        cls = pa.LargeListArray if offsets.type == 'int64' else pa.ListArray
        if isinstance(self, pa.Table):
            (self,) = self.combine_chunks().to_batches() or [pa.record_batch([], self.schema)]
        arrays = [cls.from_arrays(offsets, array, mask=mask) for array in self]
        return pa.RecordBatch.from_arrays(arrays, self.schema.names)

    def from_counts(self, counts: pa.IntegerArray) -> pa.RecordBatch:
        """Return record batch with columns converted into list columns."""
        mask = None
        if counts.null_count:
            mask, counts = counts.is_null(), counts.fill_null(0)
        offsets = pa.concat_arrays([pa.array([0], counts.type), pc.cumulative_sum_checked(counts)])
        return Table.from_offsets(self, offsets, mask=mask)

    def runs(self, *names: str, **predicates: tuple) -> tuple:
        """Return table grouped by pairwise differences, and corresponding counts.

        Args:
            *names: columns to partition by `not_equal` which will return scalars
            **predicates: pairwise predicates with optional args which will return list arrays;
                if the predicate has args, it will be called on the differences
        """
        offsets = pa.chunked_array(
            Column.run_offsets(self[name], *predicates.get(name, ()))
            for name in names + tuple(predicates)
        )
        offsets = offsets.unique().sort()
        scalars = self.select(names).take(offsets[:-1])
        lists = self.select(set(self.schema.names) - set(names))
        table = Table.union(scalars, Table.from_offsets(lists, offsets))
        return table, Column.diff(offsets)

    def list_fields(self) -> set:
        return {field.name for field in self.schema if Column.is_list_type(field)}

    def list_value_length(self) -> pa.Array:
        lists = Table.list_fields(self)
        if not lists:
            raise ValueError(f"no list columns available: {self.schema.names}")
        counts, *others = (pc.list_value_length(self[name]) for name in lists)
        if any(counts != other for other in others):
            raise ValueError(f"list columns have different value lengths: {lists}")
        return counts if isinstance(counts, pa.Array) else counts.chunk(0)

    def map_list(self, func: Callable, *args, **kwargs) -> Batch:
        """Return table with function mapped across list scalars."""
        batches: Iterable = Table.split(self.select(Table.list_fields(self)))
        batches = [None if batch is None else func(batch, *args, **kwargs) for batch in batches]
        counts = pa.array(None if batch is None else len(batch) for batch in batches)
        table = pa.Table.from_batches(batch for batch in batches if batch is not None)
        return Table.union(self, Table.from_counts(table, counts))

    def sort_indices(self, *names: str, length: int | None = None) -> pa.Array:
        """Return indices which would sort the table by columns, optimized for fixed length."""
        func = pc.sort_indices
        if length is not None and length < len(self):
            func = functools.partial(pc.select_k_unstable, k=length)
        keys = dict(map(sort_key, names))
        table = pa.table({name: Column.sort_values(self[name]) for name in keys})
        return func(table, sort_keys=keys.items()) if table else pa.array([], 'int64')

    def sort(self, *names: str, length: int | None = None, indices: str = '') -> Batch:
        """Return table sorted by columns, optimized for fixed length.

        Args:
            *names: columns to sort by
            length: maximum number of rows to return
            indices: include original indices in the table
        """
        if length == 1 and not indices:
            return Table.min_max(self, *names)[:1]
        indices_ = Table.sort_indices(self, *names, length=length)
        table = self.take(indices_)
        if indices:
            table = table.append_column(indices, indices_)
        func = lambda name: not name.startswith('-') and not self[name].null_count  # noqa: E731
        metadata = {'index_columns': list(itertools.takewhile(func, names))}
        return table.replace_schema_metadata({'pandas': json.dumps(metadata)})

    def filter_list(self, expr: ds.Expression) -> Batch:
        """Return table with list columns filtered within scalars."""
        fields = Table.list_fields(self)
        tables = [
            None if batch is None else pa.Table.from_batches([batch]).filter(expr).select(fields)
            for batch in Table.split(self)
        ]
        counts = pa.array(None if table is None else len(table) for table in tables)
        table = pa.concat_tables(table for table in tables if table is not None)
        return Table.union(self, Table.from_counts(table, counts))

    def min_max(self, *names: str) -> Self:
        """Return table filtered by minimum or maximum values."""
        for key, order in map(sort_key, names):
            field, asc = pc.field(key), (order == 'ascending')
            ((value,),) = Nodes.group(self, _=(key, ('min' if asc else 'max'), None)).to_table()
            self = self.filter(field <= value if asc else field >= value)
        return self

    def rank(self, k: int, *names: str) -> Self:
        """Return table filtered by values within dense rank, similar to `select_k_unstable`."""
        if k == 1:
            return Table.min_max(self, *names)
        keys = dict(map(sort_key, names))
        table = Nodes.group(self, *keys).to_table()
        table = table.take(pc.select_k_unstable(table, k, keys.items()))
        exprs = []
        for key, order in keys.items():
            field, asc = pc.field(key), (order == 'ascending')
            exprs.append(field <= pc.max(table[key]) if asc else field >= pc.min(table[key]))
        return self.filter(bit_all(exprs))

    def fragments(self, *names, counts: str = '') -> pa.Table:
        """Return selected fragment keys in a table."""
        try:
            expr = self._scan_options.get('filter')
            if expr is not None:  # raise ValueError if filter references other fields
                ds.dataset([], schema=self.partitioning.schema).scanner(filter=expr)
        except (AttributeError, ValueError):
            return pa.table({})
        fragments = self._get_fragments(expr)
        parts = [ds.get_partition_keys(frag.partition_expression) for frag in fragments]
        table = pa.Table.from_pylist(parts)
        keys = [name for name in names if name in table.schema.names]
        table = table.group_by(keys, use_threads=False).aggregate([])
        if not counts:
            return table
        if not table.schema:
            return table.append_column(counts, pa.array([self.count_rows()]))
        exprs = [bit_all(pc.field(key) == row[key] for key in row) for row in table.to_pylist()]
        column = [self.filter(expr).count_rows() for expr in exprs]
        return table.append_column(counts, pa.array(column))

    def rank_keys(self, k: int, *names: str, dense: bool = True) -> tuple:
        """Return expression and unmatched fields for partitioned dataset which filters by rank.

        Args:
            k: max dense rank or length
            *names: columns to rank by
            dense: use dense rank; false indicates sorting
        """
        keys = dict(map(sort_key, names))
        table = Table.fragments(self, *keys, counts='' if dense else '_')
        keys = {name: keys[name] for name in table.schema.names if name in keys}
        if not keys:
            return None, names
        if dense:
            table = table.take(pc.select_k_unstable(table, k, keys.items()))
        else:
            table = table.sort_by(keys.items())
            totals = itertools.accumulate(table['_'].to_pylist())
            counts = (count for count, total in enumerate(totals, 1) if total >= k)
            table = table[: next(counts, None)].remove_column(len(table.schema) - 1)
        exprs = [bit_all(pc.field(key) == row[key] for key in row) for row in table.to_pylist()]
        remaining = names[len(keys) :]
        if remaining or not dense:  # fields with a single value are no longer needed
            selectors = [len(table[key].unique()) > 1 for key in keys]
            remaining = tuple(itertools.compress(names, selectors)) + remaining
        return bit_any(exprs[: len(table)]), remaining

    def flatten(self, indices: str = '') -> Iterator[pa.RecordBatch]:
        """Generate batches with list arrays flattened, optionally with parent indices."""
        offset = 0
        for batch in self.to_batches():
            _ = Table.list_value_length(batch)
            indices_ = pc.list_parent_indices(batch[Table.list_fields(batch).pop()])
            arrays = [
                pc.list_flatten(array) if Column.is_list_type(array) else array.take(indices_)
                for array in batch
            ]
            columns = dict(zip(batch.schema.names, arrays))
            if indices:
                columns[indices] = pc.add(indices_, offset)
            offset += len(batch)
            yield pa.RecordBatch.from_pydict(columns)

    def split(self) -> Iterator[pa.RecordBatch | None]:
        """Generate tables from splitting list scalars."""
        lists = Table.list_fields(self)
        scalars = set(self.schema.names) - lists
        for index, count in enumerate(Table.list_value_length(self).to_pylist()):
            if count is None:
                yield None
            else:
                row = {name: pa.repeat(self[name][index], count) for name in scalars}
                row |= {name: self[name][index].values for name in lists}
                yield pa.RecordBatch.from_pydict(row)


class Nodes(ac.Declaration):
    """[Acero](https://arrow.apache.org/docs/python/api/acero.html) engine declaration.

    Provides a `Scanner` interface with no "oneshot" limitation.
    """

    option_map = {
        'table_source': ac.TableSourceNodeOptions,
        'scan': ac.ScanNodeOptions,
        'filter': ac.FilterNodeOptions,
        'project': ac.ProjectNodeOptions,
        'aggregate': ac.AggregateNodeOptions,
        'order_by': ac.OrderByNodeOptions,
        'hashjoin': ac.HashJoinNodeOptions,
    }
    to_batches = ac.Declaration.to_reader  # source compatibility

    def __init__(self, name, *args, inputs=None, **options):
        super().__init__(name, self.option_map[name](*args, **options), inputs)

    def scan(self, columns: Iterable[str]) -> Self:
        """Return projected source node, supporting datasets and tables."""
        if isinstance(self, ds.Dataset):
            expr = self._scan_options.get('filter')
            self = Nodes('scan', self, columns=columns)
            if expr is not None:
                self = self.apply('filter', expr)
        elif isinstance(self, pa.Table):
            self = Nodes('table_source', self)
        elif isinstance(self, pa.RecordBatch):
            self = Nodes('table_source', pa.table(self))
        if isinstance(columns, Mapping):
            return self.apply('project', columns.values(), columns)
        return self.apply('project', map(pc.field, columns))

    @property
    def schema(self) -> pa.Schema:
        """projected schema"""
        with self.to_reader() as reader:
            return reader.schema

    def scanner(self, **options) -> ds.Scanner:
        return ds.Scanner.from_batches(self.to_reader(**options))

    def count_rows(self) -> int:
        """Count matching rows."""
        return self.scanner().count_rows()

    def head(self, num_rows: int, **options) -> pa.Table:
        """Load the first N rows."""
        return self.scanner(**options).head(num_rows)

    def take(self, indices: Iterable[int], **options) -> pa.Table:
        """Select rows by index."""
        return self.scanner(**options).take(indices)

    def apply(self, name: str, *args, **options) -> Self:
        """Add a node by name."""
        return type(self)(name, *args, inputs=[self], **options)

    filter = functools.partialmethod(apply, 'filter')

    def group(self, *names, **aggs: tuple) -> Self:
        """Add `aggregate` node with dictionary support.

        Also supports datasets because aggregation determines the projection.
        """
        aggregates, targets = [], set(names)
        for name, (target, _, _) in aggs.items():
            aggregates.append(aggs[name] + (name,))
            targets.update([target] if isinstance(target, str) else target)
        columns = {name: pc.field(name) for name in targets}
        for name in columns:
            field = self.schema.field(name)
            if pa.types.is_dictionary(field.type):
                columns[name] = columns[name].cast(field.type.value_type)
        return Nodes.scan(self, columns).apply('aggregate', aggregates, names)
