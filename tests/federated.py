from pathlib import Path

import ibis

from graphique import typed

from .conftest import TestClient

fixtures = Path(__file__).parent / "fixtures"
table = ibis.read_parquet(fixtures / "zipcodes.parquet")


class Query:
    zipcodes = typed(table, name="zipcode", keys=["zipcode"])
    states = table.mutate(indices=ibis.row_number()).order_by("state", "county")
    zip_db = ibis.read_parquet(fixtures / "zip_db.parquet")


app = TestClient(Query)
