"""
Core arrow utilities.
"""

import itertools

import ibis
import pyarrow as pa
import pyarrow.dataset as ds


class links:
    ref = "https://ibis-project.org/reference"
    types = f"[data type]({ref}/datatypes)"
    schema = f"[table schema]({ref}/schemas#ibis.expr.schema.Schema)"


def getitems(obj, *keys):
    """Nested `getitem`."""
    for key in keys:
        obj = obj[key]
    return obj


def order_key(name: str):
    """Parse sort order."""
    return (ibis.desc if name.startswith("-") else ibis.asc)(ibis._[name.lstrip("-")])


class Parquet(ds.Dataset):
    """Partitioned parquet dataset."""

    def schema(self) -> pa.Schema:
        """partition schema"""
        return self.partitioning.schema if hasattr(self, "partitioning") else pa.schema([])

    def keys(self, *names) -> list:
        """Return prefix of matching partition keys."""
        keys = set(Parquet.schema(self).names)
        return list(itertools.takewhile(lambda name: name.lstrip("-") in keys, names))

    def fragments(self, counts: str = "", path: str = "__path__") -> ibis.Table:
        """Return partition fragments as a table."""
        parts = []
        for frag in self.get_fragments():
            parts.append(ds.get_partition_keys(frag.partition_expression))
            parts[-1][path] = frag.path
            if counts:
                parts[-1][counts] = frag.count_rows()
        return ibis.memtable(pa.Table.from_pylist(parts))

    def group(self, *names, counts: str = "") -> ibis.Table:
        """Return grouped partitions as a table."""
        table = Parquet.fragments(self, counts)
        agg = {counts: table[counts].sum()} if counts else {}
        return table.aggregate(agg, by=names).order_by(*names)

    def filter(self, expr: ds.Expression | None) -> ds.Dataset | None:
        """Attempt to apply filter to partition keys."""
        try:  # raises ValueError if filter references non-partition keys
            ds.dataset([], schema=self.partitioning.schema).scanner(filter=expr)
        except (AttributeError, ValueError):
            return None
        if expr is None:
            return self
        paths = [frag.path for frag in self.get_fragments(expr)]
        return ds.dataset(paths, partitioning=self.partitioning)

    def to_table(self) -> ibis.Table:
        """Return ibis `Table` from filtered dataset."""
        paths = [frag.path for frag in self.get_fragments()]
        hive = isinstance(self.partitioning, ds.HivePartitioning)
        return ibis.read_parquet(paths, hive_partitioning=hive)

    def rank(self, limit: int | None, *names: str, dense: bool = False) -> ds.Dataset:
        """Return ordered limited partitions of the dataset."""
        keys = {name.strip("-"): order_key(name) for name in names}
        if dense:
            table = Parquet.fragments(self).order_by(*keys.values()).cache()
            groups = table.select(*keys).value_counts().order_by(*keys.values())
        else:
            table = Parquet.fragments(self, counts="_").order_by(*keys.values()).cache()
            groups = table.aggregate(_=table[-1].sum(), __=ibis._.count(), by=list(keys))
            groups = groups.order_by(*keys.values()).cache()
            totals = itertools.accumulate(groups[-2].to_list())
            limit = next((index for index, total in enumerate(totals, 1) if total >= limit), None)
        limit = groups[:limit][-1].sum().to_pyarrow().as_py()
        return ds.dataset(table[:limit]["__path__"].to_list(), partitioning=self.partitioning)
