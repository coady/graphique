import ibis
import pyarrow.compute as pc
import pytest

from graphique.core import Parquet, rank_over
from graphique.scalars import (
    BigInt,
    Duration,
    duration_isoformat,
    parse_duration,
    py_type,
    schema_types,
)


def test_duration():
    assert duration_isoformat(parse_duration("P1Y1M1DT1H1M1.1S")) == "P13M1DT1H1M1.1S"
    assert duration_isoformat(parse_duration("P1M1DT1H1M1.1S")) == "P1M1DT1H1M1.1S"
    assert duration_isoformat(parse_duration("P1DT1H1M1.1S")) == "P1DT1H1M1.1S"
    assert duration_isoformat(parse_duration("PT1H1M1.1S")) == "PT1H1M1.1S"
    assert duration_isoformat(parse_duration("PT1M1.1S")) == "PT1M1.1S"
    assert duration_isoformat(parse_duration("PT1.1S")) == "PT1.1S"
    assert duration_isoformat(parse_duration("PT1S")) == "PT1S"
    assert duration_isoformat(parse_duration("P0D")) == "P0D"
    assert duration_isoformat(parse_duration("PT0S")) == "P0D"
    assert duration_isoformat(parse_duration("P0MT")) == "P0M0D"
    assert duration_isoformat(parse_duration("P0YT")) == "P0M0D"
    with pytest.raises(ValueError):
        duration_isoformat(parse_duration("T1H"))
    with pytest.raises(ValueError):
        duration_isoformat(parse_duration("P1H"))


def test_types():
    assert py_type(ibis.expr.datatypes.Int64()) is BigInt
    assert py_type(ibis.expr.datatypes.Int32()) is int
    assert py_type(ibis.expr.datatypes.Interval("s")) is Duration
    assert py_type(ibis.expr.datatypes.UUID()) is str
    schema = ibis.Schema({"_": ibis.dtype("map<string, int>")})
    with pytest.warns(UserWarning):
        assert not dict(schema_types(schema))
    schema = ibis.Schema({"_": ibis.dtype("array<array<string>>")})
    assert not dict(schema_types(schema, filters=True))


def test_parquet(dataset):
    assert not Parquet.schema(dataset)
    assert not Parquet.keys(dataset, "key")
    table = Parquet.fragments(dataset, "count")
    (path,) = table["__path__"].to_list()
    assert path.endswith(".parquet")
    assert table["count"].to_list() == [41700]
    assert Parquet.filter(dataset, None) is dataset
    assert Parquet.filter(dataset, pc.field("key")) is None
    assert Parquet.to_table(dataset).count().to_pyarrow().as_py() == 41700


def test_ordering(partitioned):
    assert Parquet.keys(partitioned, "north") == ["north"]
    assert Parquet.order(partitioned, "north").count_rows() == 41700
    assert Parquet.order(partitioned, "north", limit=1).count_rows() == 9301

    assert Parquet.first(partitioned, "north").count_rows() == 20850
    assert Parquet.first(partitioned, "north", rank=21000).count_rows() == 41700
    assert Parquet.first(partitioned, "north", dense=True).count_rows() == 20850
    assert Parquet.first(partitioned, "north", rank=2, dense=True).count_rows() == 41700


def test_rank_over(dataset):
    table = Parquet.to_table(dataset)
    data = rank_over(table, ["longitude"], ["state"], ibis.row_number(), 2)
    assert data.count().to_pyarrow().as_py() == 104
    assert data[:4]["state"].to_list() == ["AK", "AK", "HI", "HI"]
    data = rank_over(table, ["longitude"], ["state"], ibis.rank())
    assert data.count().to_pyarrow().as_py() == 52
    assert data[:2]["state"].to_list() == ["AK", "HI"]
