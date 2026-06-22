"""
Example GraphQL service.

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

from graphique import GraphQL, MetricsExtension

config = Config(".env" if Path(".env").is_file() else None)
PARQUET_PATH = Path(config("PARQUET_PATH")).resolve()
NAME = config("NAME", default="")
COLUMNS = config("COLUMNS", cast=json.loads, default=None)

root = ds.dataset(PARQUET_PATH, partitioning="hive" if PARQUET_PATH.is_dir() else None)

if COLUMNS or not root.partitioning.schema:
    root = ibis.read_parquet(PARQUET_PATH, table_name=NAME or PARQUET_PATH.name)
    if isinstance(COLUMNS, dict):
        root = root.select(**{alias: ibis._[name] for alias, name in COLUMNS.items()})
    elif COLUMNS:
        root = root.select(COLUMNS)

if NAME:
    app = GraphQL.federated({NAME: root}, extensions=[MetricsExtension])
else:
    app = GraphQL(root, extensions=[MetricsExtension])
