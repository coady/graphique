"""
Default GraphQL service.

Copy and customize as needed. Demonstrates:
* federation versus root type
* datasets and tables
* filtering and projection
"""

import json
from pathlib import Path

import ibis
import pyarrow.dataset as ds
from starlette.config import Config

from graphique import GraphQL
from graphique.core import Parquet
from graphique.inputs import Filter

config = Config('.env' if Path('.env').is_file() else None)
PARQUET_PATH = Path(config('PARQUET_PATH')).resolve()
FEDERATED = config('FEDERATED', default='')
METRICS = config('METRICS', cast=bool, default=False)
COLUMNS = config('COLUMNS', cast=json.loads, default=None)
FILTERS = config('FILTERS', cast=json.loads, default=None)

root = ds.dataset(PARQUET_PATH, partitioning='hive' if PARQUET_PATH.is_dir() else None)

if FILTERS is not None:
    if isinstance(COLUMNS, dict):
        COLUMNS = {alias: ds.field(name) for alias, name in COLUMNS.items()}
    root = ibis.memtable(root.to_table(columns=COLUMNS, filter=Filter.to_arrow(**FILTERS)))
elif COLUMNS or not Parquet.schema(root):
    root = Parquet.to_table(root)
    if isinstance(COLUMNS, dict):
        root = root.select(**{alias: ibis._[name] for alias, name in COLUMNS.items()})
    elif COLUMNS:
        root = root.select(COLUMNS)

if FEDERATED:
    app = GraphQL.federated({FEDERATED: root}, metrics=METRICS)
else:
    app = GraphQL(root, metrics=METRICS)
