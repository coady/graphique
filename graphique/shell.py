"""
Partition datasets out-of-core, in parquet hive format.

It follows a 2-pass strategy. First, batches are scanned and partitioned into fragments, with
multiple parts per fragment.

Second, the partitioned dataset is rewritten to merge parts. Often the built-in `write_dataset` is
sufficient once partitioned, but there is a `fragments` option to optimize for memory or show
progress on the second pass.
"""

import argparse
import shutil
from collections.abc import Iterable
from pathlib import Path
import pyarrow.dataset as ds
from tqdm import tqdm  # type: ignore


def sort_key(name: str) -> tuple:
    """Parse sort order."""
    return name.lstrip('-'), ('descending' if name.startswith('-') else 'ascending')


def write_batches(scanner: ds.Scanner, base_dir: str, *partitioning: str, **options):
    """Partition dataset by batches."""
    options.update(format='parquet', partitioning=partitioning)
    options.update(partitioning_flavor='hive', existing_data_behavior='overwrite_or_ignore')
    with tqdm(total=scanner.count_rows(), desc="Batches") as pbar:
        for index, batch in enumerate(scanner.to_batches()):
            options['basename_template'] = f'part-{index}-{{i}}.parquet'
            ds.write_dataset(batch, base_dir, **options)
            pbar.update(len(batch))


def write_fragments(dataset: ds.Dataset, base_dir: str, sorting=(), **options):
    """Rewrite partition files by fragment to consolidate."""
    options['format'] = 'parquet'
    exprs = {Path(frag.path).parent: frag.partition_expression for frag in dataset.get_fragments()}
    offset = len(dataset.partitioning.schema)
    for path in tqdm(exprs, desc="Fragments"):
        part_dir = Path(base_dir, *path.parts[-offset:])
        part = dataset.filter(exprs[path])
        ds.write_dataset(part.sort_by(sorting) if sorting else part, part_dir, **options)


def partition(
    scanner: ds.Scanner,
    base_dir: str,
    *partitioning: str,
    fragments: bool = False,
    sort: Iterable[str] = (),
    **options,
):
    """Partition dataset by keys."""
    temp = Path(base_dir) / 'temp'
    write_batches(scanner, str(temp), *partitioning)
    dataset = ds.dataset(temp, partitioning='hive')
    if fragments or sort:
        write_fragments(dataset, base_dir, tuple(map(sort_key, sort)), **options)
    else:
        options.update(partitioning_flavor='hive', existing_data_behavior='overwrite_or_ignore')
        with tqdm(desc="Partitions"):
            ds.write_dataset(dataset, base_dir, partitioning=partitioning, **options)
    shutil.rmtree(temp)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('src', help="source path")
parser.add_argument('dest', help="destination path")
parser.add_argument('partitioning', nargs='+', help="partition keys")
parser.add_argument('--fragments', action='store_true', help="iterate over fragments")
parser.add_argument('--sort', nargs='*', default=(), help="sort keys; will load fragments")

if __name__ == '__main__':
    args = parser.parse_args()
    dataset = ds.dataset(args.src, partitioning='hive')
    partition(dataset, args.dest, *args.partitioning, fragments=args.fragments, sort=args.sort)
