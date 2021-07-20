import pytest


def test_dataset(dsclient):
    data = dsclient.execute('{ column(name: "state") { length } }')
    assert data == {'column': {'length': 41700}}
    data = dsclient.execute('{ length row { state } }')
    assert data == {'length': 41700, 'row': {'state': 'NY'}}
    data = dsclient.execute('{ read(state: {isIn: ["CA", "NY"]}) { length } }')
    assert data == {'read': {'length': 4852}}
    data = dsclient.execute('{ read(state: {notEqual: "CA"}) { length } }')
    assert data == {'read': {'length': 39053}}
    with pytest.raises(ValueError, match="optional, not nullable"):
        dsclient.execute('{ read(state: null) { length } }')
