"""
Example GraphQL service.

Copy and customize as needed. Demonstrates:
* named versus root query type
* parquet dataset as ibis table, optionally with DuckLake
* projection at startup
"""

import json
from pathlib import Path

import ibis
from starlette.config import Config

from graphique import GraphQL, MetricsExtension, ducklake

config = Config(".env" if Path(".env").is_file() else None)
PARQUET_PATH = Path(config("PARQUET_PATH")).resolve()
NAME = config("NAME", default="")
COLUMNS = config("COLUMNS", cast=json.loads, default=None)

if PARQUET_PATH.is_dir():
    root = ducklake.register(str(PARQUET_PATH), NAME or PARQUET_PATH.name)
    # `register` defaults to caching catalog in memory; use a file and `attach` for production reuse
    # setup: `ducklake.register(PARQUET_PATH, <name>, catalog=<file>)`
    # service: `root = ducklake.attach(<name>, catalog=<file>)`
else:
    root = ibis.read_parquet(PARQUET_PATH, table_name=NAME or PARQUET_PATH.name)

if isinstance(COLUMNS, dict):
    root = root.select(**{alias: ibis._[name] for alias, name in COLUMNS.items()})
elif COLUMNS:
    root = root.select(*COLUMNS)

if NAME:  # typically a class; `type` expression only needed because `NAME` is dynamic
    root = type("Query", (), {NAME: root})
app = GraphQL(root, extensions=[MetricsExtension])
