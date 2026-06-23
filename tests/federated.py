from pathlib import Path

import ibis
import pyarrow.dataset as ds

from graphique import implement

from .conftest import TestClient

fixtures = Path(__file__).parent / "fixtures"
dataset = ds.dataset(fixtures / "zipcodes.parquet")
table = ibis.read_parquet(fixtures / "zipcodes.parquet")


class Query:
    zipcodes = implement(table.schema(), name="zipcodes", keys=["zipcode"])(source=dataset)
    states = table.mutate({"indices": ibis.row_number()}).order_by("state", "county")
    zip_db = ds.dataset(fixtures / "zip_db.parquet")


app = TestClient(Query)
