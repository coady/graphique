"""
Core utilities that add pandas-esque features to arrow arrays and tables.

Arrow forbids subclassing, so the classes are for logical grouping.
Their methods are called as functions.
"""

import contextlib
import functools
import itertools
import operator
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import TypeAlias
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


def sort_key(name: str) -> tuple:
    """Parse sort order."""
    return name.lstrip('-'), ('descending' if name.startswith('-') else 'ascending')


def order_key(name: str) -> ibis.Deferred:
    """Parse sort order."""
    return (ibis.desc if name.startswith('-') else ibis.asc)(ibis._[name.lstrip('-')])


class Column(pa.ChunkedArray):
    """Chunked array interface as a namespace of functions."""

    def is_list_type(self):
        funcs = pa.types.is_list, pa.types.is_large_list, pa.types.is_fixed_size_list
        return any(func(self.type) for func in funcs)

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


class Table(pa.Table):
    """Table interface as a namespace of functions."""

    def map_batch(self, func: Callable, *args, **kwargs) -> pa.Table:
        batches = self.to_pyarrow_batches() if isinstance(self, ibis.Table) else self.to_batches()
        return pa.Table.from_batches(func(batch, *args, **kwargs) for batch in batches)

    def columns(self) -> dict:
        """Return columns as a dictionary."""
        return dict(zip(self.schema.names, self))

    def union(*tables: Batch) -> Batch:
        """Return table with union of columns."""
        columns: dict = {}
        for table in tables:
            columns |= Table.columns(table)
        return type(tables[0]).from_pydict(columns)

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
        return Nodes.scan(self, list(targets)).apply('aggregate', aggregates, names)


class Parquet(ds.Dataset):
    """Parquet dataset converters."""

    def partition_keys(self) -> pa.Schema:
        """partition schema"""
        return self.partitioning.schema if hasattr(self, 'partitioning') else pa.schema([])

    def filter(self, expr: ds.Expression) -> ds.Dataset | None:
        """Attempt to apply filter to partition keys."""
        try:  # raises ValueError if filter references non-partition keys
            ds.dataset([], schema=self.partitioning.schema).scanner(filter=expr)
        except (AttributeError, ValueError):
            return None
        return self.filter(expr)  # pragma: no cover

    def to_table(self) -> ibis.Table:
        """Return ibis `Table` from filtered dataset."""
        paths = [frag.path for frag in self._get_fragments(self._scan_options.get('filter'))]
        hive = isinstance(self.partitioning, ds.HivePartitioning)
        return ibis.read_parquet(paths, hive_partitioning=hive)
