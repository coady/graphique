import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
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


def test_nodes(table):
    dataset = ds.dataset(table).filter(pc.field('state') == 'CA')
    (column,) = Nodes.scan(dataset, columns={'_': pc.field('state')}).to_table()
    assert column.unique().to_pylist() == ['CA']
    scanner = Nodes.scan(dataset, columns=['state'])
    assert scanner.schema.names == ['state']
    assert scanner.count_rows() == 2647
    assert scanner.head(3) == pa.table({'state': ['CA'] * 3})
    assert scanner.take([0, 2]) == pa.table({'state': ['CA'] * 2})


def test_not_implemented():
    dictionary = pa.array(['']).dictionary_encode()
    with pytest.raises((NotImplementedError, TypeError)):
        pc.sort_indices(pa.table({'': dictionary}), [('', 'ascending')])
    with pytest.raises(NotImplementedError):
        pc.first_last(dictionary)
    with pytest.raises(NotImplementedError):
        pc.min_max(dictionary)
    with pytest.raises(NotImplementedError):
        pc.count_distinct(dictionary)
    with pytest.raises(NotImplementedError):
        pc.utf8_lower(dictionary)
    with pytest.raises(NotImplementedError):
        pa.StructArray.from_arrays([], []).dictionary_encode()
    for index in (-1, 1):
        with pytest.raises(ValueError):
            pc.list_element(pa.array([[0]]), index)
    with pytest.raises(NotImplementedError):
        pc.any([0])
    array = pa.array(list('aba'))
    with pytest.raises(NotImplementedError):
        pa.table({'': array.dictionary_encode()}).group_by('').aggregate([('', 'min')])
    with pytest.raises(NotImplementedError):
        pa.table({'': array}).group_by('').aggregate([('', 'any')])
    agg = 'value', 'max', pc.ScalarAggregateOptions(min_count=4)
    table = pa.table({'value': list('abc'), 'key': [0, 1, 0]})
    table = table.group_by('key').aggregate([agg])
    assert table['value_max'].to_pylist() == list('cb')  # min_count has no effect
    for name in ('one', 'list', 'distinct'):
        assert not hasattr(pc, name)
    with pytest.raises(NotImplementedError):
        pa.table({'': list('aba')}).group_by([]).aggregate([('', 'first'), ('', 'last')])
    with pytest.raises(NotImplementedError):
        pc.rank(pa.table({'': array}), [('', 'ascending')])
