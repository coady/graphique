import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pytest
from graphique.core import ListChunk, Nodes, Column as C, Table as T
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


def test_dictionary(table):
    array = table['state'].dictionary_encode()
    table = pa.table({'state': array})
    assert T.sort(table, 'state')['state'][0].as_py() == 'AK'
    array = pa.chunked_array([['a', 'b'], ['a', 'b', None]]).dictionary_encode()
    assert C.fill_null_backward(array) == array.combine_chunks()
    assert C.fill_null_forward(array)[-1].as_py() == 'b'
    assert C.fill_null(array[3:], "c").to_pylist() == list('bc')
    assert C.fill_null(array[:3], "c").to_pylist() == list('aba')
    assert C.sort_values(array.combine_chunks()).to_pylist() == [1, 2, 1, 2, None]


def test_chunks():
    array = pa.chunked_array([pa.array(list(chunk)).dictionary_encode() for chunk in ('aba', 'ca')])
    assert C.index(array, 'a') == 0
    assert C.index(array, 'c') == 3
    assert C.index(array, 'a', start=3) == 4
    assert C.index(array, 'b', start=2) == -1


def test_lists():
    array = pa.array([[2, 1], [0, 0], [None], [], None])
    assert ListChunk.first(array).to_pylist() == [2, 0, None, None, None]
    assert ListChunk.element(array, -2).to_pylist() == [2, 0, None, None, None]
    assert ListChunk.last(array).to_pylist() == [1, 0, None, None, None]
    assert ListChunk.last(pa.chunked_array([array])).to_pylist() == [1, 0, None, None, None]
    assert ListChunk.element(array, 1).to_pylist() == [1, 0, None, None, None]
    assert ListChunk.min(array).to_pylist() == [1, 0, None, None, None]
    assert ListChunk.max(array).to_pylist() == [2, 0, None, None, None]
    assert ListChunk.mode(array)[0].as_py() == [{'mode': 1, 'count': 1}]
    assert ListChunk.quantile(array).to_pylist() == [[1.5], [0.0], [None], [None], [None]]
    quantile = ListChunk.quantile(array, q=[0.75])
    assert quantile.to_pylist() == [[1.75], [0.0], [None], [None], [None]]
    array = pa.array([[True, True], [False, False], [None], [], None])
    array = pa.ListArray.from_arrays([0, 2, 3], pa.array(["a", "b", None]).dictionary_encode())
    assert ListChunk.min(array).to_pylist() == ["a", None]
    assert ListChunk.max(array).to_pylist() == ["b", None]
    assert C.is_list_type(pa.FixedSizeListArray.from_arrays([], 1))
    array = pa.array([[list('ab'), ['c']], [list('de')]])
    assert ListChunk.inner_flatten(array).to_pylist() == [list('abc'), list('de')]
    batch = T.from_offsets(pa.record_batch([list('abcde')], ['col']), pa.array([0, 3, 5]))
    assert batch['col'].to_pylist() == [list('abc'), list('de')]
    assert not T.from_offsets(pa.table({}), pa.array([0]))
    array = ListChunk.from_counts(pa.array([3, None, 2]), list('abcde'))
    assert array.to_pylist() == [list('abc'), None, list('de')]
    with pytest.raises(ValueError):
        T.list_value_length(pa.table({'x': array, 'y': array[::-1]}))


def test_membership():
    array = pa.chunked_array([[1, 1]])
    assert C.index(array, 1) == C.index(array, 1, end=1) == 0
    assert C.index(array, 1, start=1) == 1
    assert C.index(array, 1, start=2) == -1


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
    tbl = T.sort(table, 'state', 'longitude')
    groups, counts = T.runs(tbl, 'state', longitude=(pc.greater, 1.0))
    assert len(groups) == len(counts) == 62
    assert groups['state'].value_counts()[0].as_py() == {'values': 'AK', 'counts': 7}
    assert groups['longitude'][:2].to_pylist() == [[-174.213333], [-171.701685]]
    groups, counts = T.runs(tbl, 'state', longitude=(pc.less,))
    assert len(groups) == len(counts) == 52


def test_sort(table):
    data = T.sort(table, 'state').to_pydict()
    assert (data['state'][0], data['county'][0]) == ('AK', 'Anchorage')
    data = T.sort(table, 'state', 'county', length=1).to_pydict()
    assert (data['state'], data['county']) == (['AK'], ['Aleutians East'])
    tbl = T.sort(table, 'state', '-county', length=2, null_placement='at_start')
    assert tbl.schema.pandas_metadata == {'index_columns': ['state']}
    assert tbl['state'].to_pylist() == ['AK'] * 2
    assert tbl['county'].to_pylist() == ['Yukon Koyukuk'] * 2
    counts = T.rank(table, 1, 'state')['state'].value_counts().to_pylist()
    assert counts == [{'values': 'AK', 'counts': 273}]
    counts = T.rank(table, 1, 'state', '-county')['county'].value_counts().to_pylist()
    assert counts == [{'values': 'Yukon Koyukuk', 'counts': 30}]
    counts = T.rank(table, 2, 'state')['state'].value_counts().to_pylist()
    assert counts == [{'values': 'AL', 'counts': 838}, {'values': 'AK', 'counts': 273}]
    counts = T.rank(table, 2, 'state', '-county')['county'].value_counts().to_pylist()
    assert counts == [{'counts': 30, 'values': 'Yukon Koyukuk'}, {'counts': 1, 'values': 'Yakutat'}]
    table = pa.table({'x': [list('ab'), [], None, ['c']]})
    (column,) = T.map_list(table, T.sort, '-x', length=2)
    assert column.to_pylist() == [list('ba'), [], None, ['c']]


def test_numeric():
    array = pa.array([0.0, 10.0, 20.0])
    scalar = pa.scalar([10.0])
    assert pc.call_function('digitize', [array, scalar, False]).to_pylist() == [0, 1, 1]
    assert pc.call_function('digitize', [array, scalar, True]).to_pylist() == [0, 0, 1]


def test_list():
    array = pa.array([[False], [True], [False, True]])
    assert pc.call_function('list_all', [array]).to_pylist() == [False, True, False]
    assert pc.call_function('list_any', [array]).to_pylist() == [False, True, True]


def test_not_implemented():
    dictionary = pa.array(['']).dictionary_encode()
    with pytest.raises((NotImplementedError, TypeError)):
        pc.sort_indices(pa.table({'': dictionary}), [('', 'ascending')])
    with pytest.raises(NotImplementedError):
        dictionary.index('')
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
        pc.fill_null_forward(dictionary)
    with pytest.raises(NotImplementedError):
        pa.table({'': list('aba')}).group_by([]).aggregate([('', 'first'), ('', 'last')])
    with pytest.raises(ValueError):
        pc.pairwise_diff(pa.chunked_array([[0]]))
