from graphique.core import Array as A


def test_dictionary(table):
    array = table['state'].dictionary_encode()
    values, counts = A.unique(array, counts=True)
    assert len(values) == len(counts) == 52
    assert set(A.unique(array)) == set(values)
    assert A.min(array) == 'AK'
    assert A.max(array) == 'WY'
