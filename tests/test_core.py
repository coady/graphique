import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pytest
from graphique.core import Agg, ListChunk, Column as C, Table as T


def test_dictionary(table):
    array = table['state'].dictionary_encode()
    assert C.min(array) == 'AK'
    assert C.max(array) == 'WY'
    assert C.min_max(array[:0]) == {'min': None, 'max': None}
    table = pa.table({'state': array})
    assert T.sort(table, 'state')['state'][0].as_py() == 'AK'
    array = pa.chunked_array([['a', 'b'], ['a', 'b', None]]).dictionary_encode()
    assert C.fill_null_backward(array) == array.combine_chunks()
    assert C.fill_null_forward(array)[-1].as_py() == 'b'
    assert C.fill_null(array[3:], "c").to_pylist() == list('bc')
    assert C.fill_null(array[:3], "c").to_pylist() == list('aba')


def test_chunks():
    array = pa.chunked_array([list('aba'), list('bcb')])
    table = pa.table({'col': array})
    groups = T.group(table, 'col', counts='counts')
    assert groups['col'].to_pylist() == list('abc')
    assert groups['counts'].to_pylist() == [2, 3, 1]
    assert table['col'].unique() == pa.array('abc')
    array = pa.chunked_array([pa.array(list(chunk)).dictionary_encode() for chunk in ('aba', 'ca')])
    assert C.index(array, 'a') == 0
    assert C.index(array, 'c') == 3
    assert C.index(array, 'a', start=3) == 4
    assert C.index(array, 'b', start=2) == -1
    table = pa.table({'col': array})
    tbl = T.group(table, 'col', count_distinct=[Agg('col', 'count')], list=[Agg('col', 'list')])
    assert tbl['col'].type == 'string'
    assert tbl['count'].to_pylist() == [1] * 3
    (counts,) = ListChunk.aggregate(tbl['list'], count_distinct=None)
    assert counts.to_pylist() == [1] * 3
    assert len(T.map_batch(table, T.group, 'col')) == 4
    scanner = ds.dataset(table).scanner(filter=ds.field('col') == '')
    assert not T.map_batch(scanner, T.group, 'col')


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


def test_membership():
    array = pa.chunked_array([[1, 1]])
    assert C.index(array, 1) == C.index(array, 1, end=1) == 0
    assert C.index(array, 1, start=1) == 1
    assert C.index(array, 1, start=2) == -1


def test_group(table):
    groups = T.group(table, 'state', list=[Agg('county'), Agg('city')])
    assert len(groups) == 52
    assert groups['state'][0].as_py() == 'NY'
    table = T.filter_list(groups, pc.field('county') == pc.field('city'))
    assert len(pc.list_flatten(table['city'])) == 2805
    mins = T.matched(groups, C.min, 'state', 'county')
    assert mins['state'].to_pylist() == ['AK']
    assert mins['county'].to_pylist() == [['Aleutians East'] * 5]
    assert mins['city'][0].values[0].as_py() == 'Akutan'
    groups = T.sort_list(groups, 'county')
    assert groups['county'][0].values[0].as_py() == 'Albany'
    groups = T.sort_list(groups, '-county', '-city', length=1)
    assert groups['county'][0].values.to_pylist() == ['Yates']
    assert groups['city'][0].values.to_pylist() == ['Rushville']
    groups = groups.append_column('other', pa.array([[0]] * len(groups)))
    with pytest.raises(ValueError):
        T.sort_list(groups, 'county')


def test_aggregate(table):
    tbl = T.union(table, table.select([0]).rename_columns(['test']))
    assert tbl.column_names == table.column_names + ['test']
    groups = T.group(table, 'state', 'county')
    assert len(groups) == 3216
    assert groups.column_names == ['state', 'county']
    groups = T.group(table, 'state', counts='counts', first=[Agg('county')])
    assert len(groups) == 52
    assert groups['state'][0].as_py() == 'NY'
    assert groups['counts'][0].as_py() == 2205
    assert groups['county'][0].as_py() == 'Suffolk'
    groups = T.group(table, 'state', last=[Agg('city', 'last')], min=[Agg('zipcode')])
    assert groups['last'][0].as_py() == 'Elmira'
    assert groups['zipcode'][0].as_py() == 501
    groups = T.group(table, 'state', max=[Agg('zipcode', 'max', skip_nulls=False)])
    assert groups['max'][0].as_py() == 14925
    groups = T.group(table, 'state', min_max=[Agg('zipcode')])
    assert groups['zipcode'][0].as_py() == {'min': 501, 'max': 14925}
    groups = T.group(
        table, 'state', approximate_median=[Agg('longitude')], tdigest=[Agg('latitude')]
    )
    assert groups['longitude'][0].as_py() == pytest.approx(-74.25370)
    assert groups['latitude'][0].as_py() == [pytest.approx(42.34672)]
    row = T.aggregate(table, min=[Agg('state')])
    assert row['state'].as_py() == 'AK'
    assert row['zipcode'] == table['zipcode'].combine_chunks()
    row = T.aggregate(table, counts='counts', quantile=[Agg('zipcode')])
    assert row['counts'] == 41700
    assert row['zipcode'].to_pylist() == [48817.5]
    row = T.aggregate(table, element=[Agg('zipcode', index=-1)])
    assert row['zipcode'].as_py() == 99950
    row = T.aggregate(table, slice=[Agg('zipcode', start=-3, stop=None, step=1)])
    assert row['zipcode'].to_pylist() == [99928, 99929, 99950]


def test_partition(table):
    groups, counts = T.partition(table, 'state')
    assert len(groups) == len(counts) == 66
    assert pc.sum(counts).as_py() == 41700
    assert groups['state'][0].as_py() == 'NY'
    assert groups['county'][0].values.to_pylist() == ['Suffolk', 'Suffolk']
    groups, counts = T.partition(table, 'state', 'county')
    assert len(groups) == len(counts) == 22751
    groups, counts = T.partition(table, zipcode=(pc.greater, 100))
    assert len(groups) == len(counts) == 59
    tbl = T.sort(table, 'state', 'longitude')
    groups, counts = T.partition(tbl, 'state', longitude=(pc.greater, 1.0))
    assert len(groups) == len(counts) == 62
    assert groups['state'].value_counts()[0].as_py() == {'values': 'AK', 'counts': 7}
    assert groups['longitude'][:2].to_pylist() == [[-174.213333], [-171.701685]]
    groups, counts = T.partition(tbl, 'state', longitude=(pc.less,))
    assert len(groups) == len(counts) == 52


def test_sort(table):
    data = T.sort(table, 'state').to_pydict()
    assert (data['state'][0], data['county'][0]) == ('AK', 'Anchorage')
    data = T.sort(table, 'state', 'county', length=1).to_pydict()
    assert (data['state'], data['county']) == (['AK'], ['Aleutians East'])
    tbl = T.sort(table, 'state', '-county', length=2)
    assert tbl.schema.pandas_metadata == {'index_columns': ['state']}
    assert tbl['state'].to_pylist() == ['AK'] * 2
    assert tbl['county'].to_pylist() == ['Yukon Koyukuk'] * 2


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
    with pytest.raises(NotImplementedError):
        pc.sort_indices(dictionary)
    with pytest.raises(NotImplementedError):
        dictionary.index('')
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
    dictionary = pa.array(['']).dictionary_encode()
    with pytest.raises(ValueError, match="string vs dictionary"):
        pc.index_in(dictionary.unique(), value_set=dictionary)
    array = pa.array(list('aba'))
    with pytest.raises(NotImplementedError):
        pa.table({'': array.dictionary_encode()}).group_by('').aggregate([('', 'min')])
    with pytest.raises(NotImplementedError):
        pa.table({'': array}).group_by('').aggregate([('', 'any')])
    func = Agg('value', min_count=4).astuple('max')
    table = pa.table({'value': list('abc'), 'key': [0, 1, 0]})
    values, _ = table.group_by('key').aggregate([func])
    assert values.to_pylist() == list('cb')  # min_count has no effect
    value = pa.MonthDayNano([1, 2, 3])
    with pytest.raises(NotImplementedError):
        pc.equal(value, value)
