import pyarrow as pa
import pytest
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
    data = dsclient.execute('{ filter(query: {state: {equal: "CA"}}, reduce: XOR) { length } }')
    assert data == {'filter': {'length': 2647}}


def test_search(dsclient):
    data = dsclient.execute('{ search(zipcode: {less: 10000}) { length } }')
    assert data == {'search': {'length': 3224}}
    data = dsclient.execute('{ search(zipcode: {}) { length } }')
    assert data == {'search': {'length': 41700}}
    data = dsclient.execute('{ search(zipcode: {}) { row { zipcode } } }')
    assert data == {'search': {'row': {'zipcode': 501}}}
    data = dsclient.execute(
        '''{ search(zipcode: {greater: 90000}) { filter(query: {state: {equal: "CA"}}) {
        length } } }'''
    )
    assert data == {'search': {'filter': {'length': 2647}}}
    data = dsclient.execute(
        '''{ search(zipcode: {greater: 90000}) { filter(query: {state: {equal: "CA"}}) {
        length row { zipcode } } } }'''
    )
    assert data == {'search': {'filter': {'length': 2647, 'row': {'zipcode': 90001}}}}


def test_slice(dsclient):
    data = dsclient.execute('{ slice(length: 3) { length } }')
    assert data == {'slice': {'length': 3}}
    data = dsclient.execute('{ slice(offset: -3) { length } }')
    assert data == {'slice': {'length': 3}}
    data = dsclient.execute('{ slice { length } }')
    assert data == {'slice': {'length': 41700}}


@pytest.mark.skipif(pa.__version__ < '7', reason="requires pyarrow >=7")
def test_group(dsclient):
    data = dsclient.execute(
        '''{ group(by: ["state"], aggregate: {min: {name: "county"}}) { row { state county } } }'''
    )
    assert data == {'group': {'row': {'state': 'NY', 'county': 'Albany'}}}
    data = dsclient.execute(
        '''{ group(by: ["state"], counts: "c") { slice(length: 1) {
        column(name: "c") { ... on LongColumn { values } } } } }'''
    )
    assert data == {'group': {'slice': {'column': {'values': [2205]}}}}
    data = dsclient.execute(
        '''{ group(by: ["state"], aggregate: {mean: {name: "zipcode"}}) { slice(length: 1) {
        column(name: "zipcode") { ... on FloatColumn { values } } } } }'''
    )
    assert data == {'group': {'slice': {'column': {'values': [pytest.approx(12614.62721)]}}}}
    data = dsclient.execute(
        '''{ group(by: ["state"]) { aggregate(mean: {name: "zipcode"}) { slice(length: 1) {
        column(name: "zipcode") { ... on FloatColumn { values } } } } } }'''
    )
    assert data == {
        'group': {'aggregate': {'slice': {'column': {'values': [pytest.approx(12614.62721)]}}}}
    }


def test_federation(fedclient):
    data = fedclient.execute('{ _service { sdl } aTable { length } }')
    assert data['aTable']['length'] == 2
    assert data['_service']['sdl']
