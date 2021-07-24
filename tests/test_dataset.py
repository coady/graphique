def test_filter(dsclient):
    data = dsclient.execute('{ column(name: "state") { length } }')
    assert data == {'column': {'length': 41700}}
    data = dsclient.execute('{ length row { state } }')
    assert data == {'length': 41700, 'row': {'state': 'NY'}}
    data = dsclient.execute('{ filter(query: {state: {isIn: ["CA", "NY"]}}) { length } }')
    assert data == {'filter': {'length': 4852}}
    data = dsclient.execute('{ filter(query: {state: {notEqual: "CA"}}) { length } }')
    assert data == {'filter': {'length': 39053}}
    data = dsclient.execute('{ filter { length } }')
    assert data == {'filter': {'length': 41700}}


def test_search(dsclient):
    data = dsclient.execute('{ search(zipcode: {less: 10000}) { length } }')
    assert data == {'search': {'length': 3224}}
