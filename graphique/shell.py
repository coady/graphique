"""
Partition datasets out-of-core, in parquet hive format.

It follows a 2-pass strategy. First, batches are scanned and partitioned into fragments, with
multiple parts per fragment.

Second, the partitioned dataset is rewritten to merge parts. Often the built-in `write_dataset` is
sufficient once partitioned, but there is a `fragments` option to optimize for memory or show
progress on the second pass.
"""

import operator
import shutil
from pathlib import Path
from typing import Annotated, Callable, Optional
import numpy as np
import pyarrow.compute as pc
import pyarrow.dataset as ds
import typer  # type: ignore
from tqdm import tqdm  # type: ignore


def sort_key(name: str) -> tuple:
    """Parse sort order."""
    return name.lstrip('-'), ('descending' if name.startswith('-') else 'ascending')


def write_batches(
    scanner: ds.Scanner, base_dir: str, *partitioning: str, indices: str = '', **options
):
    """Partition dataset by batches.

    Optionally include original indices.
    """
    options.update(format='parquet', partitioning=partitioning)
    options.update(partitioning_flavor='hive', existing_data_behavior='overwrite_or_ignore')
    with tqdm(total=scanner.count_rows(), desc="Batches") as pbar:
        for index, batch in enumerate(scanner.to_batches()):
            if indices:
                batch = batch.append_column(indices, pc.add(np.arange(len(batch)), pbar.n))
            options['basename_template'] = f'part-{index}-{{i}}.parquet'
            ds.write_dataset(batch, base_dir, **options)
            pbar.update(len(batch))


def write_fragments(dataset: ds.Dataset, base_dir: str, func: Optional[Callable] = None, **options):
    """Rewrite partition files by fragment to consolidate, optionally transforming."""
    options['format'] = 'parquet'
    exprs = {Path(frag.path).parent: frag.partition_expression for frag in dataset.get_fragments()}
    offset = len(dataset.partitioning.schema)
    for path in tqdm(exprs, desc="Fragments"):
        part_dir = Path(base_dir, *path.parts[-offset:])
        part = dataset.filter(exprs[path])
        ds.write_dataset(func(part) if func else part, part_dir, **options)


def partition(
    src: Annotated[str, typer.Argument(help="source path")],
    dest: Annotated[str, typer.Argument(help="destination path")],
    partitioning: Annotated[list[str], typer.Argument(help="partition keys")],
    fragments: Annotated[bool, typer.Option(help="iterate over fragments")] = False,
    sort: Annotated[list[str], typer.Option(help="sort keys; will load fragments")] = [],
):
    """Partition dataset by keys."""
    temp = Path(dest) / 'temp'
    write_batches(ds.dataset(src, partitioning='hive'), str(temp), *partitioning)
    dataset = ds.dataset(temp, partitioning='hive')
    options = dict(partitioning_flavor='hive', existing_data_behavior='overwrite_or_ignore')
    if sorting := list(map(sort_key, sort)):
        write_fragments(dataset, dest, operator.methodcaller('sort_by', sorting))
    elif fragments:
        write_fragments(dataset, dest)
    else:
        with tqdm(desc="Partitions"):
            ds.write_dataset(dataset, dest, partitioning=partitioning, **options)
    shutil.rmtree(temp)


if __name__ == '__main__':
    partition.__doc__ = __doc__
    typer.run(partition)
