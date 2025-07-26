import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pytest
from graphique.core import Nodes, Column as C, Table as T
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


def test_lists():
    assert C.is_list_type(pa.FixedSizeListArray.from_arrays([], 1))
    batch = T.from_offsets(pa.record_batch([list('abcde')], ['col']), pa.array([0, 3, 5]))
    assert batch['col'].to_pylist() == [list('abc'), list('de')]
    assert not T.from_offsets(pa.table({}), pa.array([0]))
    with pytest.raises(ValueError):
        T.list_value_length(pa.table({'x': pa.array([[''], []]), 'y': pa.array([[], ['']])}))


def test_nodes(table):
    dataset = ds.dataset(table).filter(pc.field('state') == 'CA')
    (column,) = Nodes.scan(dataset, columns={'_': pc.field('state')}).to_table()
    assert column.unique().to_pylist() == ['CA']
    table = Nodes.group(dataset, 'county', 'city', counts=([], 'hash_count_all', None)).to_table()
    assert len(table) == 1241
    assert pc.sum(table['counts']).as_py() == 2647
    scanner = Nodes.scan(dataset, columns=['state'])
    assert scanner.schema.names == ['state']
    assert scanner.group('state').to_table() == pa.table({'state': ['CA']})
    assert scanner.count_rows() == 2647
    assert scanner.head(3) == pa.table({'state': ['CA'] * 3})
    assert scanner.take([0, 2]) == pa.table({'state': ['CA'] * 2})


def test_runs(table):
    groups, counts = T.runs(table, 'state')
    assert len(groups) == len(counts) == 66
    assert pc.sum(counts).as_py() == 41700
    assert groups['state'][0].as_py() == 'NY'
    assert groups['county'][0].values.to_pylist() == ['Suffolk', 'Suffolk']
    groups, counts = T.runs(table, 'state', 'county')
    assert len(groups) == len(counts) == 22751
    groups, counts = T.runs(table, zipcode=(pc.greater, 100))
    assert len(groups) == len(counts) == 59
    tbl = table.sort_by([('state', 'ascending'), ('longitude', 'ascending')])
    groups, counts = T.runs(tbl, 'state', longitude=(pc.greater, 1.0))
    assert len(groups) == len(counts) == 62
    assert groups['state'].value_counts()[0].as_py() == {'values': 'AK', 'counts': 7}
    assert groups['longitude'][:2].to_pylist() == [[-174.213333], [-171.701685]]
    groups, counts = T.runs(tbl, 'state', longitude=(pc.less,))
    assert len(groups) == len(counts) == 52


def test_sort(table):
    counts = T.rank(table, 1, 'state')['state'].value_counts().to_pylist()
    assert counts == [{'values': 'AK', 'counts': 273}]
    counts = T.rank(table, 1, 'state', '-county')['county'].value_counts().to_pylist()
    assert counts == [{'values': 'Yukon Koyukuk', 'counts': 30}]
    counts = T.rank(table, 2, 'state')['state'].value_counts().to_pylist()
    assert counts == [{'values': 'AL', 'counts': 838}, {'values': 'AK', 'counts': 273}]
    counts = T.rank(table, 2, 'state', '-county')['county'].value_counts().to_pylist()
    assert counts == [{'counts': 30, 'values': 'Yukon Koyukuk'}, {'counts': 1, 'values': 'Yakutat'}]


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
    with pytest.raises(ValueError):
        pc.pairwise_diff(pa.chunked_array([[0]]))
    with pytest.raises(NotImplementedError):
        pc.rank(pa.table({'': array}), [('', 'ascending')])
