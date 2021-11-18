from graphique import middleware


def test_extension(capsys):
    ext = middleware.TimingExtension(execution_context=None)
    assert ext.on_request_start() is None
    assert ext.on_request_end() is None
    assert capsys.readouterr().out.startswith('[')


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
    data = dsclient.execute('{ filter(query: {state: {notEqual: null}}) { length } }')
    assert data == {'filter': {'length': 41700}}
    data = dsclient.execute('{ filter(query: {state: {equal: "CA"}}, invert: true) { length } }')
    assert data == {'filter': {'length': 39053}}
    data = dsclient.execute(
        '''{ filter(query: {state: {equal: "CA"}, county: {equal: "Santa Clara"}}, reduce: OR)
        { length } }'''
    )
    assert data == {'filter': {'length': 2647}}


def test_search(dsclient):
    data = dsclient.execute('{ search(zipcode: {less: 10000}) { length } }')
    assert data == {'search': {'length': 3224}}
    data = dsclient.execute('{ search(zipcode: {}) { length } }')
    assert data == {'search': {'length': 41700}}


def test_federation(fedclient):
    data = fedclient.execute('{ _service { sdl } aTable { length } }')
    assert data['aTable']['length'] == 2
    assert data['_service']['sdl']
