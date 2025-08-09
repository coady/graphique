import pyarrow as pa
import pyarrow.compute as pc
import pytest
from graphique.core import Nodes
from graphique.scalars import parse_duration, duration_isoformat


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


def test_nodes(dataset):
    dataset = dataset.filter(pc.field('state') == 'CA')
    (column,) = Nodes.scan(dataset, columns={'_': pc.field('state')}).to_table()
    assert column.unique().to_pylist() == ['CA']
    scanner = Nodes.scan(dataset, columns=['state'])
    assert scanner.schema.names == ['state']
    assert scanner.count_rows() == 2647
    assert scanner.head(3) == pa.table({'state': ['CA'] * 3})
    assert scanner.take([0, 2]) == pa.table({'state': ['CA'] * 2})
