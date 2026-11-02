import shutil

import ibis
import pytest

from graphique import ducklake
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


def test_ducklake(partitioned, tmp_path):
    table = ducklake.register(partitioned.files, "partitioned")
    assert list(table.schema())[-2:] == ["north", "west"]
    catalog = tmp_path / "ducklake"
    table = ducklake.register(partitioned, "partitioned", catalog=catalog)
    assert list(table.schema())[-2:] == ["north", "west"]
    del table
    table = ducklake.attach("partitioned", catalog=catalog)
    assert list(table.schema())[-2:] == ["north", "west"]
    shutil.rmtree(tmp_path)
