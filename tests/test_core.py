import ibis
import pyarrow.compute as pc
import pytest

from graphique.core import Parquet
from graphique.scalars import BigInt, Duration, duration_isoformat, parse_duration, py_type


def test_duration():
    assert duration_isoformat(parse_duration('P1Y1M1DT1H1M1.1S')) == 'P13M1DT1H1M1.1S'
    assert duration_isoformat(parse_duration('P1M1DT1H1M1.1S')) == 'P1M1DT1H1M1.1S'
    assert duration_isoformat(parse_duration('P1DT1H1M1.1S')) == 'P1DT1H1M1.1S'
    assert duration_isoformat(parse_duration('PT1H1M1.1S')) == 'PT1H1M1.1S'
    assert duration_isoformat(parse_duration('PT1M1.1S')) == 'PT1M1.1S'
    assert duration_isoformat(parse_duration('PT1.1S')) == 'PT1.1S'
    assert duration_isoformat(parse_duration('PT1S')) == 'PT1S'
    assert duration_isoformat(parse_duration('P0D')) == 'P0D'
    assert duration_isoformat(parse_duration('PT0S')) == 'P0D'
    assert duration_isoformat(parse_duration('P0MT')) == 'P0M0D'
    assert duration_isoformat(parse_duration('P0YT')) == 'P0M0D'
    with pytest.raises(ValueError):
        duration_isoformat(parse_duration('T1H'))
    with pytest.raises(ValueError):
        duration_isoformat(parse_duration('P1H'))


def test_types():
    assert py_type(ibis.expr.datatypes.Int64()) is BigInt
    assert py_type(ibis.expr.datatypes.Int32()) is int
    assert py_type(ibis.expr.datatypes.Interval('s')) is Duration


def test_parquet(dataset):
    assert not Parquet.schema(dataset)
    assert not Parquet.keys(dataset, 'key')
    table = Parquet.fragments(dataset, 'count')
    (path,) = table['__path__'].to_list()
    assert path.endswith('.parquet')
    assert table['count'].to_list() == [41700]
    table = Parquet.group(dataset, '__path__', counts='count')
    assert table['count'].to_list() == [41700]
    assert Parquet.filter(dataset, None) is dataset
    assert Parquet.filter(dataset, pc.field('key')) is None
    assert Parquet.to_table(dataset).count().to_pyarrow().as_py() == 41700
