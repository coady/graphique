"""
Core utilities that add pandas-esque features to arrow arrays and tables.

Arrow forbids subclassing, so the classes are for logical grouping.
Their methods are called as functions.
"""

import functools
import itertools
import operator
from collections.abc import Iterable, Mapping
from typing import TypeAlias
import ibis.backends.duckdb
import pyarrow as pa
import pyarrow.acero as ac
import pyarrow.compute as pc
import pyarrow.dataset as ds
from typing_extensions import Self


Array: TypeAlias = pa.Array | pa.ChunkedArray
Batch: TypeAlias = pa.RecordBatch | pa.Table
bit_any = functools.partial(functools.reduce, operator.or_)
bit_all = functools.partial(functools.reduce, operator.and_)


def sort_key(name: str) -> tuple:
    """Parse sort order."""
    return name.lstrip('-'), ('descending' if name.startswith('-') else 'ascending')


def order_key(name: str) -> ibis.Deferred:
    """Parse sort order."""
    return (ibis.desc if name.startswith('-') else ibis.asc)(ibis._[name.lstrip('-')])


class Table(pa.Table):
    """Table interface as a namespace of functions."""

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
        return table.group_by(keys, use_threads=False).aggregate([])

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
        table = table.take(pc.select_k_unstable(table, k, keys.items()))
        exprs = [bit_all(pc.field(key) == row[key] for key in row) for row in table.to_pylist()]
        remaining = names[len(keys) :]
        if remaining or not dense:  # fields with a single value are no longer needed
            selectors = [len(table[key].unique()) > 1 for key in keys]
            remaining = tuple(itertools.compress(names, selectors)) + remaining
        return bit_any(exprs[: len(table)]), remaining


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
    """Partitioned parquet dataset."""

    def schema(self) -> pa.Schema:
        """partition schema"""
        return self.partitioning.schema if hasattr(self, 'partitioning') else pa.schema([])

    def keys(self, *names) -> list:
        """Return prefix of matching partition keys."""
        keys = set(Parquet.schema(self).names)
        return list(itertools.takewhile(lambda name: name.lstrip('-') in keys, names))

    def fragments(self, counts: str = '') -> ibis.Table:
        """Return partition fragments as a table."""
        parts = []
        for frag in self._get_fragments(self._scan_options.get('filter')):
            parts.append(ds.get_partition_keys(frag.partition_expression))
            parts[-1]['__path__'] = frag.path
            if counts:
                parts[-1][counts] = frag.count_rows()
        return ibis.memtable(pa.Table.from_pylist(parts))

    def group(self, *names, counts: str = '') -> ibis.Table:
        """Return grouped partitions as a table."""
        table = Parquet.fragments(self, counts)
        agg = {counts: table[counts].sum()} if counts else {}
        return table.aggregate(agg, by=names).order_by(*names)

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

    def topk(self, k: int, *names: str) -> ibis.Table:
        """Return topk partitions as a table."""
        table = Parquet.fragments(self, counts='_')
        table = table.order_by(*map(order_key, names))
        totals = itertools.accumulate(table['_'].to_list())
        stops = (stop for stop, total in enumerate(totals, 1) if total >= k)
        table = table[: next(stops, None)]
        hive = isinstance(self.partitioning, ds.HivePartitioning)
        return ibis.read_parquet(table['__path__'].to_list(), hive_partitioning=hive)
