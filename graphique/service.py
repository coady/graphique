"""
Default GraphQL service.

Copy and customize as needed. Demonstrates:
* federation versus root type
* datasets, scanners, and tables
* filtering and projection
"""

import json
from pathlib import Path
import pyarrow.dataset as ds
from starlette.config import Config
from graphique.inputs import Expression
from graphique import GraphQL

config = Config('.env' if Path('.env').is_file() else None)
PARQUET_PATH = Path(config('PARQUET_PATH')).resolve()
FEDERATED = config('FEDERATED', default='')
DEBUG = config('DEBUG', cast=bool, default=False)
COLUMNS = config('COLUMNS', cast=json.loads, default=None)
FILTERS = config('FILTERS', cast=json.loads, default=None)

root = ds.dataset(PARQUET_PATH, partitioning='hive' if PARQUET_PATH.is_dir() else None)

if isinstance(COLUMNS, dict):
    COLUMNS = {alias: ds.field(name) for alias, name in COLUMNS.items()}
if FILTERS is not None:
    root = root.to_table(columns=COLUMNS, filter=Expression.from_query(**FILTERS).to_arrow())
elif COLUMNS:
    root = root.scanner(columns=COLUMNS)

if FEDERATED:
    app = GraphQL.federated({FEDERATED: root}, debug=DEBUG)
else:
    app = GraphQL(root, debug=DEBUG)
