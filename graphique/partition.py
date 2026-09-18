"""
Partition datasets out-of-core, into parquet hive format.

First pass uses `ibis.Table.to_parquet(..., partition_by=...)` with optional memory limit.
Second pass compacts partitions with optional sorting.
"""

from pathlib import Path
from typing import Annotated

import ibis
import typer
from tqdm import tqdm

from .core import order_key


def connect(memory_limit: str = "", **config):
    if memory_limit:
        config["memory_limit"] = memory_limit
    return ibis.duckdb.connect(**config)


def compact(root: str, *sort: str, memory_limit: str = "", max_files: int = 1):
    """Compact parquet partitions with optional sorting."""
    order = list(map(order_key, sort))
    files = {path: names for path, dirs, names in Path(root).walk() if not dirs and names}
    con = connect(memory_limit)
    for path in tqdm(files, desc="Compact"):
        temp = path.parent / "temp.parquet"
        table = con.read_parquet(path, hive_partitioning=False)
        if order:
            table = table.order_by(*order)
        elif len(files[path]) <= max_files:
            continue
        table.to_parquet(temp)
        for name in files[path]:
            (path / name).unlink()
        temp.rename(path / "data_0.parquet")


def partition(
    src: Annotated[str, typer.Argument(help="source path")],
    dest: Annotated[str, typer.Argument(help="destination path")],
    partitioning: Annotated[list[str], typer.Argument(help="partition keys")],
    sort: Annotated[list[str], typer.Option(help="sort keys within each partition")] = [],
    memory_limit: Annotated[str, typer.Option(help="duckdb memory limit")] = "",
):
    """Partition dataset by keys."""
    con = connect(memory_limit, preserve_insertion_order=not sort)
    con.settings["enable_progress_bar"] = True
    partition_by = partitioning[0] if len(partitioning) == 1 else tuple(partitioning)
    con.read_parquet(src).to_parquet(dest, partition_by=partition_by, write_partition_columns=False)
    con.disconnect()
    compact(dest, *sort, memory_limit=memory_limit)


if __name__ == "__main__":
    partition.__doc__ = __doc__
    typer.run(partition)
