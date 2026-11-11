import asyncio
import json

import ibis
import pyarrow.dataset as ds
import pytest

from graphique import GraphQL, middleware, typed
from graphique.interface import cached

from .conftest import fixtures, load, unordered


def test_extensions():
    ext = middleware.MetricsExtension(execution_context=type("", (), {"context": {}}))
    for name in ("operation", "parse", "validate"):
        assert list(getattr(ext, "on_" + name)()) == [None]
    assert set(ext.get_results()["metrics"]) == {"duration", "execution"}


def test_deprecated_dataset_root():
    dataset = ds.dataset(fixtures / "zipcodes.parquet")
    with pytest.warns(DeprecationWarning, match="ibis.read_parquet|ducklake.register"):
        client = GraphQL(dataset)
    assert client.root_value.table.count().to_pyarrow().as_py() == 41700
    with pytest.warns(DeprecationWarning, match="ibis.read_parquet|ducklake.register"):
        table = typed(ds.dataset(fixtures / "partitioned", partitioning="hive"), name="hive")
    assert list(table.table.schema())[-2:] == ["north", "west"]


def test_partition(partclient):
    data = partclient.execute("{ filter(north: {eq: 1}) { count } }")
    assert data == {"filter": {"count": 20850}}
    data = partclient.execute("""{ group(by: ["north", "west"], counts: "c", order: "_") {
        columns { north { values } } column(name: "c") { ... on BigIntColumn { values } } } }""")
    assert data["group"] == {
        "columns": {"north": {"values": [0, 0, 1, 1]}},
        "column": {"values": unordered([9301, 11549, 11549, 9301])},
    }


def test_schema(partclient):
    schema = partclient.execute("{ schema { type backend names types } }")["schema"]
    assert schema["type"] == "Table"
    assert schema["backend"] == "duckdb"
    assert len(schema["names"]) == 8
    assert set(schema["types"]) == {"float64", "int32", "int64", "string"}
    data = partclient.execute(" { slice { count optional { schema { type } } } }")
    assert data == {"slice": {"count": 41700, "optional": {"schema": {"type": "CachedTable"}}}}
    data = partclient.execute(" { slice { __typename optional { schema { type } } } }")
    assert data == {"slice": {"__typename": "Table", "optional": {"schema": {"type": "Table"}}}}


def test_materialize(partclient, monkeypatch):
    data = partclient.execute("{ materialize { schema { backend } count } }")
    assert data == {"materialize": {"schema": {"backend": "duckdb"}, "count": 41700}}
    monkeypatch.delattr(ibis.expr.types.Table, "cache")
    with pytest.raises(TypeError, match="materialize"):
        cached(partclient.root_value.table.filter(ibis._["state"] == "CA"))


def test_rank_over(partclient):
    data = partclient.execute('{ first(over: "state", by: "-zipcode") { count } }')
    assert data == {"first": {"count": 52}}
    data = partclient.execute('{ first(over: "state", by: "-zipcode", rank: 2) { count } }')
    assert data == {"first": {"count": 104}}
    data = partclient.execute('{ order(over: "state", by: "-zipcode", limit: 2) { count } }')
    assert data == {"order": {"count": 104}}


def test_order(partclient):
    data = partclient.execute('{ order(by: "-north") { row { north } } }')
    assert data == {"order": {"row": {"north": 1}}}
    data = partclient.execute('{ order(by: "north", limit: 3) { count } }')
    assert data == {"order": {"count": 3}}
    data = partclient.execute('{ order(by: ["north", "state"]) { row { state } } }')
    assert data == {"order": {"row": {"state": "AL"}}}
    data = partclient.execute('{ first(by: "north", rank: 1) { count } }')
    assert data == {"first": {"count": 20850}}
    data = partclient.execute('{ first(by: "north", rank: 2, dense: true) { count } }')
    assert data == {"first": {"count": 41700}}
    data = partclient.execute('{ first(by: ["north", "west"], rank: 1) { count } }')
    assert data == {"first": {"count": 9301}}
    data = partclient.execute('{ first(by: ["north", "west"], rank: 2, dense: true) { count } }')
    assert data == {"first": {"count": 20850}}
    data = partclient.execute('{ first(by: ["north", "west"], rank: 3, dense: true) { count } }')
    assert data == {"first": {"count": 32399}}
    data = partclient.execute("""{ first(by: ["north", "state"], rank: 2)
        { columns { state { nunique(approx: true) } } } }""")
    assert data == {"first": {"columns": {"state": {"nunique": 1}}}}
    data = partclient.execute("""{ first(by: ["north", "state"], rank: 2, dense: true)
        { columns { state { nunique(approx: true) } } } }""")
    assert data == {"first": {"columns": {"state": {"nunique": 2}}}}


def test_root():
    app = load("partitioned", NAME="test")
    assert asyncio.run(app.get_root_value(None)) is app.root_value
    assert app.root_value.test
    with pytest.warns(UserWarning):
        assert load("nofields.parquet", NAME="test")
    app = load("zipcodes.parquet", COLUMNS=json.dumps(["state"]))
    assert app.root_value.schema().names == ("state",)
    app = load("zipcodes.parquet", COLUMNS=json.dumps({"zipCode": "zipcode"}))
    assert app.root_value.schema().names == ("zipCode",)


def test_federation(fedclient):
    data = fedclient.execute(
        "{ _service { sdl } zipcodes { __typename count } zipDb { __typename count } }"
    )
    assert data["_service"]["sdl"]
    assert data["zipcodes"] == {"__typename": "ZipcodeTable", "count": 41700}
    assert data["zipDb"] == {"__typename": "ZipDbTable", "count": 42724}

    data = fedclient.execute("""{ _entities(representations: {__typename: "ZipcodeTable", zipcode: 90001}) {
        ... on ZipcodeTable { count row { state } } } }""")
    assert data == {"_entities": [{"count": 1, "row": {"state": "CA"}}]}
    data = fedclient.execute("""{ states { filter(state: {eq: "CA"}) { columns { indices {
        takeFrom(field: "zipcodes") { __typename column(name: "state") { count } } } } } } }""")
    table = data["states"]["filter"]["columns"]["indices"]["takeFrom"]
    assert table == {"__typename": "ZipcodeTable", "column": {"count": 2647}}


def test_sorted(fedclient):
    data = fedclient.execute(
        '{ states { filter(state: {eq: "CA"}, county: {eq: "Santa Clara"}) { count } } }'
    )
    assert data == {"states": {"filter": {"count": 108}}}
    data = fedclient.execute(
        '{ states { filter(state: {eq: ["CA", "OR"]}, county: {eq: "Santa Clara"}) { count } } }'
    )
    assert data == {"states": {"filter": {"count": 108}}}
    data = fedclient.execute(
        '{ states { filter(state: {le: "CA"}, county: {eq: "Santa Clara"}) { count } } }'
    )
    assert data == {"states": {"filter": {"count": 108}}}
    data = fedclient.execute('{ states { filter { filter(state: {eq: "CA"}) { count } } } }')
    assert data == {"states": {"filter": {"filter": {"count": 2647}}}}


def test_join(fedclient):
    with pytest.raises(ValueError):
        fedclient.execute("""{ zipcodes { join(right: "zip_db", keys: "zipcode") { any } } }""")
    data = fedclient.execute("""{ zipcodes { join(right: "zip_db", keys: "zipcode", rkeys: "zip")
        { count schema { names } } } }""")
    table = data["zipcodes"]["join"]
    assert table["count"] == 41684
    assert set(table["schema"]["names"]) > {"zipcode", "timezone", "latitude"}
    data = fedclient.execute("""{ zipcodes { join(right: "zip_db", keys: "zipcode", rkeys: "zip", how: "right")
        { count schema { names } } } }""")
    table = data["zipcodes"]["join"]
    assert table["count"] == 42724
    assert set(table["schema"]["names"]) > {"zip", "timezone", "latitude"}
    data = fedclient.execute(
        '{ zipcodes { asofJoin(right: "zipcodes", on: "zipcode", tolerance: 1){ count } } }'
    )
    assert data == {"zipcodes": {"asofJoin": {"count": 41700}}}
    with pytest.raises(ValueError):
        fedclient.execute(
            '{ zipcodes { asofJoin(right: "zipcodes", on: "zipcode", scalar: {duration: "PT1S"}){ count } } }'
        )
    data = fedclient.execute(
        '{ zipcodes { asofJoin(right: "zipcodes", on: "zipcode", keys: "city", rkeys: "city"){ count } } }'
    )
    assert data == {"zipcodes": {"asofJoin": {"count": 41700}}}
    data = fedclient.execute('{ zipcodes { crossJoin(right: "zip_db") { count } } }')
    assert data == {"zipcodes": {"crossJoin": {"count": 1781590800}}}


def test_sets(fedclient):
    data = fedclient.execute('{ zipcodes { difference(table: "zipcodes") { count } } }')
    assert data == {"zipcodes": {"difference": {"count": 0}}}
    data = fedclient.execute('{ zipcodes { intersect(table: "zipcodes") { count } } }')
    assert data == {"zipcodes": {"intersect": {"count": 41700}}}
    data = fedclient.execute('{ zipcodes { union(table: "zipcodes") { count } } }')
    assert data == {"zipcodes": {"union": {"count": 83400}}}
