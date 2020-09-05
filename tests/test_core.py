import pyarrow as pa
import pyarrow.compute as pc
from graphique.core import Chunk, Column as C, Table as T


def eq(left, right):
    return left == right and type(left) is type(right)


def test_dictionary(table):
    array = table['state'].dictionary_encode()
    values, counts = C.value_counts(array).flatten()
    assert len(values) == len(counts) == 52
    assert set(C.unique(array)) == set(values)
    assert C.min(array) == 'AK'
    assert C.max(array) == 'WY'
    assert C.min(array[:0]) is None
    assert C.max(array[:0]) is None
    assert sum(C.mask(array, match_substring="CA").to_pylist()) == 2647
    assert sum(C.mask(array, is_in=["CA"]).to_pylist()) == 2647
    assert "ca" in C.call(array, pc.utf8_lower).unique().dictionary.to_pylist()


def test_chunks():
    array = pa.chunked_array([list('aba'), list('bcb')])
    groups = {key: value.to_pylist() for key, value in C.arggroupby(array).items()}
    assert groups == {'a': [0, 2], 'b': [1, 0, 2], 'c': [1]}
    assert C.minimum(array, array).equals(array)
    assert C.maximum(array, array).equals(array)
    table = pa.Table.from_pydict({'col': array})
    tables = list(T.group(table, 'col'))
    assert list(map(len, tables)) == [2, 3, 1]
    assert table['col'].unique() == pa.array('abc')
    assert C.sort(array, length=1).to_pylist() == ['a']
    assert C.sort(array, reverse=True).to_pylist() == list('cbbbaa')
    groups = [''.join(tbl['col'].to_pylist()) for tbl in T.group(table, 'col')]
    assert groups == ['aa', 'bbb', 'c']
    assert ''.join(T.unique(table, 'col')['col'].to_pylist()) == 'abc'
    table = pa.Table.from_pydict({'col': array.dictionary_encode()})
    groups = [''.join(tbl['col'].to_pylist()) for tbl in T.group(table, 'col', reverse=True)]
    assert groups == ['c', 'bbb', 'aa']
    assert ''.join(T.unique(table, 'col', reverse=True)['col'].to_pylist()) == 'bca'
    table = pa.Table.from_pydict({'col': array, 'other': range(6)})
    assert len(list(T.group(table, 'col'))) == 3
    assert len(T.unique(table, 'col')) == 3


def test_reduce():
    array = pa.chunked_array([[0, 1], [2, 3]])
    assert eq(C.min(array), 0)
    assert eq(C.max(array), 3)
    assert eq(C.sum(array), 6)
    assert eq(C.sum(array, exp=2), 14)
    array = pa.chunked_array([[None, 1]])
    assert eq(C.min(array), 1)
    assert eq(C.max(array), 1)


def test_membership():
    array = pa.chunked_array([[0]])
    assert C.count(array, True) == 0
    array = pa.chunked_array([[0, 1]])
    assert C.count(array, True) == 1
    array = pa.chunked_array([[1, 1]])
    assert C.count(array, True) == 2
    assert C.count(array, False) == C.count(array, None) == 0
    assert C.count(array, 0) == 0 and C.count(array, 1) == 2


def test_functional(table):
    array = table['state'].dictionary_encode()
    assert set(C.mask(array).to_pylist()) == {True}
    assert set(C.equal(array, None).to_pylist()) == {False}
    assert set(C.not_equal(array, None).to_pylist()) == {True}
    mask = C.equal(array, 'CA')
    assert mask == C.is_in(array, ['CA']) == C.is_in(table['state'], ['CA'])
    assert C.not_equal(array, 'CA') == C.is_in(array, ['CA'], invert=True)
    assert len(array.filter(mask)) == 2647
    assert sum(C.mask(array, less_equal='CA', greater_equal='CA').to_pylist()) == 2647
    assert sum(C.mask(array, binary_length={'equal': 2}).to_pylist()) == 41700
    assert sum(C.mask(array, utf8_is_upper=True).to_pylist()) == 41700
    assert sum(C.mask(array, utf8_is_upper=False).to_pylist()) == 41700
    assert sum(C.mask(table['longitude'], absolute={'less': 0}).to_pylist()) == 0
    mask = T.mask(table, 'state', equal='CA', apply={'minimum': 'state'})
    assert sum(mask.to_pylist()) == 2647
    mask = T.mask(table, 'city', apply={'equal': 'county'})
    assert sum(mask.to_pylist()) == 2805
    table = T.apply(table, 'latitude', alias='diff', subtract='longitude')
    assert table['latitude'][0].as_py() - table['longitude'][0].as_py() == table['diff'][0].as_py()
    table = T.apply(table, 'zipcode', fill_null=0)
    assert not table['zipcode'].null_count
    table = T.apply(table, 'state', utf8_lower=True, utf8_upper=False)
    assert table['state'][0].as_py() == 'ny'


def test_group(table):
    groups = C.arggroupby(table['zipcode'])
    assert len(groups) == 41700
    assert set(map(len, groups.values())) == {1}
    groups = C.arggroupby(table['state'])
    tables = list(T.group(table, 'state'))
    assert len(groups) == len(tables) == 52
    assert list(map(len, groups.values())) == list(map(len, tables))
    for chunk, indices in zip(table['state'].chunks, groups['CA'].chunks):
        assert set(chunk.take(indices)) <= {pa.scalar('CA')}
    groups = C.arggroupby(table['latitude'])
    assert max(map(len, groups.values())) == 6
    keys, indices = Chunk.arggroupby(pa.array([1, None, 1]))
    assert keys.to_pylist() == [1, None]
    assert indices.to_pylist() == [[0, 2], [1]]
    tables = list(T.group(table, 'state', predicate=(100).__ge__, sort=True))
    assert [table['state'][0].as_py() for table in tables] == ['RI', 'DE']


def test_unique(table):
    assert len(T.unique(table, 'zipcode')) == 41700
    zipcodes = T.unique(table, 'state')['zipcode'].to_pylist()
    assert len(zipcodes) == 52
    assert zipcodes[0] == 501
    assert zipcodes[-1] == 99501
    zipcodes = T.unique(table, 'state', reverse=True)['zipcode'].to_pylist()
    assert len(zipcodes) == 52
    assert zipcodes[0] == 99950
    assert zipcodes[-1] == 988
    indices = Chunk.argunique(pa.array([1, None, 1]))
    assert indices.to_pylist() == [0, 1]


def test_sort(table):
    states = C.sort(table['state']).to_pylist()
    assert (states[0], states[-1]) == ('AK', 'WY')
    states = C.sort(table.combine_chunks()['state'], reverse=True).to_pylist()
    assert (states[0], states[-1]) == ('WY', 'AK')
    assert C.sort(table['state'], length=1).to_pylist() == ['AK']
    assert C.sort(table['state'], reverse=True, length=1).to_pylist() == ['WY']
    data = T.sort(table, 'state').to_pydict()
    assert (data['state'][0], data['county'][0]) == ('AK', 'Anchorage')
    data = T.sort(table, 'state', 'county', length=1).to_pydict()
    assert (data['state'], data['county']) == (['AK'], ['Aleutians East'])
    data = T.sort(table, 'state', 'county', 'city', reverse=True, length=2).to_pydict()
    assert data['state'] == ['WY', 'WY']
    assert data['county'] == ['Weston', 'Weston']
    assert data['city'] == ['Upton', 'Osage']
