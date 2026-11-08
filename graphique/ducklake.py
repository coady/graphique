"""DuckLake helpers."""

import functools
from collections.abc import Sequence

import ibis
import pyarrow.dataset as ds


@functools.singledispatch
def register(
    source: str | Sequence[str],
    table: str,
    *partitioning: str,
    catalog: str = ":memory:",
    database: str = "lake.main",
) -> ibis.Table:
    """Attach a DuckLake catalog and register a parquet dataset."""
    alias, _, schema = database.partition(".")
    con = ibis.duckdb.connect(extensions=["ducklake"])
    con.raw_sql(f"ATTACH 'ducklake:{catalog}' AS {alias}")
    con.create_table(table, schema=con.read_parquet(source).schema(), database=database)
    if columns := ", ".join(partitioning):
        con.raw_sql(f"ALTER TABLE {database}.{table} SET PARTITIONED BY ({columns})")
    parameters = alias, table, source, schema
    con.raw_sql("CALL ducklake_add_data_files(?, ?, ?, schema => ?)", parameters=parameters)
    return con.table(table, database=database)


@register.register
def _(source: ds.Dataset, table: str, filter: ds.Expression | None = None, **kwargs) -> ibis.Table:
    paths = [frag.path for frag in source.get_fragments(filter)]
    return register(paths, table, *source.partitioning.schema.names, **kwargs)


def attach(table: str, catalog: str, database: str = "lake.main") -> ibis.Table:
    """Attach a DuckLake catalog read-only and return a table."""
    alias, _, _ = database.partition(".")
    con = ibis.duckdb.connect(extensions=["ducklake"])
    con.raw_sql(f"ATTACH 'ducklake:{catalog}' AS {alias} (READ_ONLY)")
    return con.table(table, database=database)
