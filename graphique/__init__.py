from .interface import Dataset
from .middleware import GraphQL, MetricsExtension, implement, typed


def fetch_record_batch(self, rows_per_batch):
    return self.to_arrow_reader(rows_per_batch)


def connect(database=":memory:", read_only=False, config={}):
    return original(database, read_only, {"lambda_syntax": "ENABLE_SINGLE_ARROW", **config})


try:
    import duckdb
except ImportError:  # pragma: no cover
    pass
else:  # ibis-project/ibis#12196
    if not hasattr(duckdb.DuckDBPyConnection, "fetch_arrow_table"):  # pragma: no branch
        duckdb.DuckDBPyConnection.fetch_arrow_table = duckdb.DuckDBPyConnection.to_arrow_table
        duckdb.DuckDBPyConnection.fetch_record_batch = fetch_record_batch
        original, duckdb.connect = duckdb.connect, connect  # type: ignore
