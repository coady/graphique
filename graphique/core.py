"""
Core utilities that add pandas-esque features to arrow arrays and tables.

Arrow forbids subclassing, so the classes are for logical grouping.
Their methods are called as functions.
"""

import functools
import itertools
from collections.abc import Iterable, Mapping
import ibis.backends.duckdb
import pyarrow as pa
import pyarrow.acero as ac
import pyarrow.compute as pc
import pyarrow.dataset as ds
from typing_extensions import Self


def order_key(name: str) -> ibis.Deferred:
    """Parse sort order."""
    return (ibis.desc if name.startswith('-') else ibis.asc)(ibis._[name.lstrip('-')])


class Nodes(ac.Declaration):
    """[Acero](https://arrow.apache.org/docs/python/api/acero.html) engine declaration.

    Provides a `Scanner` interface with no "oneshot" limitation.
    """

    option_map = {
        'table_source': ac.TableSourceNodeOptions,
        'scan': ac.ScanNodeOptions,
        'filter': ac.FilterNodeOptions,
        'project': ac.ProjectNodeOptions,
    }

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
        return self if expr is None else self.filter(expr)

    def to_table(self) -> ibis.Table:
        """Return ibis `Table` from filtered dataset."""
        paths = [frag.path for frag in self._get_fragments(self._scan_options.get('filter'))]
        hive = isinstance(self.partitioning, ds.HivePartitioning)
        return ibis.read_parquet(paths, hive_partitioning=hive)

    def rank(self, limit: int, *names: str, dense: bool = False) -> ibis.Table:
        """Return ordered limited partitions as a table."""
        keys = {name.strip('-'): order_key(name) for name in names}
        table = Parquet.fragments(self, counts='_').order_by(*keys.values()).cache()
        groups = table.aggregate(count=ibis._.count(), total=table['_'].sum(), by=list(keys))
        groups = groups.order_by(*keys.values()).cache()
        if not dense:
            totals = itertools.accumulate(groups['total'].to_list())
            limit = next((index for index, total in enumerate(totals, 1) if total >= limit), None)  # type: ignore
        limit = groups[:limit]['count'].sum().to_pyarrow().as_py()
        hive = isinstance(self.partitioning, ds.HivePartitioning)
        return ibis.read_parquet(table[:limit]['__path__'].to_list(), hive_partitioning=hive)
