"""
Core arrow utilities.
"""

import bisect
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


def rank_over(
    table: ibis.Table,
    by: list[str],
    over: list[str],
    index: ibis.Column,
    rank: int = 1,
) -> ibis.Table:
    """Filter rows by rank within each grouping window."""
    order_by = list(map(order_key, by))
    table = table.mutate(_=index.over(group_by=over, order_by=order_by))
    return table.filter(table["_"] < rank).drop("_").order_by(*order_by)


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

    def to_table(self, name: str | None = None) -> ibis.Table:
        """Return ibis `Table` from filtered dataset."""
        paths = [frag.path for frag in self.get_fragments()]
        return ibis.read_parquet(paths, table_name=name)

    def order(self, *names: str, limit: int | None = None) -> ds.Dataset:
        """Return ordered partitions of the dataset."""
        table = Parquet.fragments(self, counts="" if limit is None else "_")
        table = table.order_by(*map(order_key, names)).cache()
        if limit is not None:
            limit = bisect.bisect_left(table["_"].cumsum().to_list(), limit) + 1
        return ds.dataset(table[:limit]["__path__"].to_list(), partitioning=self.partitioning)

    def first(self, *names: str, rank: int = 1, dense: bool = False) -> ds.Dataset:
        """Return ordered partitions up to max rank (dense or sparse)."""
        keys = {name.strip("-"): order_key(name) for name in names}
        if dense or rank == 1:
            table = Parquet.fragments(self).aggregate(_=ibis._["__path__"].collect(), by=list(keys))
            paths = table.order_by(*keys.values())[:rank]["_"].unnest()
        else:
            table = Parquet.fragments(self, counts="_").order_by(*keys.values()).cache()
            limit = bisect.bisect_left(table["_"].cumsum().to_list(), rank) + 1
            paths = table.semi_join(table[:limit].select(*keys).distinct(), list(keys))["__path__"]
        return ds.dataset(paths.to_list(), partitioning=self.partitioning)
