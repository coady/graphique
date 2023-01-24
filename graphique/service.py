"""
Default GraphQL service.
"""
import pyarrow.compute as pc
import pyarrow.dataset as ds
from .inputs import Expression
from .middleware import GraphQL
from .settings import COLUMNS, DEBUG, FEDERATED, FILTERS, PARQUET_PATH

root = dataset = ds.dataset(PARQUET_PATH, partitioning='hive' if PARQUET_PATH.is_dir() else None)

if isinstance(COLUMNS, dict):
    COLUMNS = {alias: pc.field(name) for alias, name in COLUMNS.items()}
if FILTERS is not None:
    root = dataset.to_table(columns=COLUMNS, filter=Expression.from_query(**FILTERS).to_arrow())
elif COLUMNS:
    root = dataset.scanner(columns=COLUMNS)

if FEDERATED:
    app = GraphQL.federated({FEDERATED: root}, debug=DEBUG)
else:
    app = GraphQL(root, debug=DEBUG)
