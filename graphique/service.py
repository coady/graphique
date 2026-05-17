"""
Default GraphQL service.

Copy and customize as needed. Demonstrates:
* named versus root query type
* datasets and tables
* projection at startup
"""

import json
from pathlib import Path

import ibis
import pyarrow.dataset as ds
from starlette.config import Config

from graphique import GraphQL
from graphique.core import Parquet

config = Config(".env" if Path(".env").is_file() else None)
PARQUET_PATH = Path(config("PARQUET_PATH")).resolve()
NAME = config("NAME", default="")
METRICS = config("METRICS", cast=bool, default=False)
COLUMNS = config("COLUMNS", cast=json.loads, default=None)

root = ds.dataset(PARQUET_PATH, partitioning="hive" if PARQUET_PATH.is_dir() else None)

if COLUMNS or not Parquet.schema(root):
    root = Parquet.to_table(root, name=NAME or PARQUET_PATH.name)
    if isinstance(COLUMNS, dict):
        root = root.select(**{alias: ibis._[name] for alias, name in COLUMNS.items()})
    elif COLUMNS:
        root = root.select(COLUMNS)

if NAME:
    app = GraphQL.federated({NAME: root}, metrics=METRICS)
else:
    app = GraphQL(root, metrics=METRICS)
