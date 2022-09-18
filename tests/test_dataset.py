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
    data = dsclient.execute('{ filter(state: {eq: ["CA", "NY"]}) { length } }')
    assert data == {'filter': {'length': 4852}}
    data = dsclient.execute('{ filter(state: {ne: "CA"}) { length } }')
    assert data == {'filter': {'length': 39053}}
    data = dsclient.execute('{ filter { length } }')
    assert data == {'filter': {'length': 41700}}
    data = dsclient.execute('{ filter(state: {ne: null}) { length } }')
    assert data == {'filter': {'length': 41700}}


def test_search(dsclient):
    data = dsclient.execute('{ filter(zipcode: {lt: 10000}) { length } }')
    assert data == {'filter': {'length': 3224}}
    data = dsclient.execute('{ filter(zipcode: {}) { length } }')
    assert data == {'filter': {'length': 41700}}
    data = dsclient.execute('{ filter(zipcode: {}) { row { zipcode } } }')
    assert data == {'filter': {'row': {'zipcode': 501}}}
    data = dsclient.execute(
        '''{ filter(zipcode: {gt: 90000}) { filter(state: {eq: "CA"}) {
        length } } }'''
    )
    assert data == {'filter': {'filter': {'length': 2647}}}
    data = dsclient.execute(
        '''{ filter(zipcode: {gt: 90000}) { filter(state: {eq: "CA"}) {
        length row { zipcode } } } }'''
    )
    assert data == {'filter': {'filter': {'length': 2647, 'row': {'zipcode': 90001}}}}
    data = dsclient.execute(
        '''{ filter(zipcode: {lt: 90000}) { filter(state: {eq: "CA"}) {
        group(by: "county") { length } } } }'''
    )
    assert data == {'filter': {'filter': {'group': {'length': 0}}}}


def test_slice(dsclient):
    data = dsclient.execute('{ slice(length: 3) { length } }')
    assert data == {'slice': {'length': 3}}
    data = dsclient.execute('{ slice(offset: -3) { length } }')
    assert data == {'slice': {'length': 3}}
    data = dsclient.execute('{ slice { length } }')
    assert data == {'slice': {'length': 41700}}
    data = dsclient.execute('{ take(indices: [0]) { row { zipcode } } }')
    assert data == {'take': {'row': {'zipcode': 501}}}


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
        '{ group(by: ["state"], aggregate: {first: {name: "county"}}) { row { county } } }'
    )
    assert data == {'group': {'row': {'county': 'Suffolk'}}}
    data = dsclient.execute(
        '{ group(by: ["state"], aggregate: {one: {name: "county"}}) { row { county } } }'
    )
    assert data['group']['row']['county']
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


def test_schema(dsclient):
    schema = dsclient.execute('{ schema { names types partitioning } }')['schema']
    assert set(schema['names']) >= {'zipcode', 'state', 'county'}
    assert set(schema['types']) >= {'int32', 'string'}
    assert len(schema['partitioning']) in (0, 6)
    assert dsclient.execute('{ type }')['type'] in {'FileSystemDataset', 'Scanner', 'Table'}


def test_scan(dsclient):
    data = dsclient.execute(
        '{ scan(columns: {name: "zipcode", alias: "zip"}) { column(name: "zip") { type } } }'
    )
    assert data == {'scan': {'column': {'type': 'int32'}}}
    data = dsclient.execute(
        '{ scan(filter: {eq: [{name: "county"}, {name: "state"}]}) { length } }'
    )
    assert data == {'scan': {'length': 0}}
    data = dsclient.execute('{ scan(filter: {eq: [{name: "zipcode"}, {value: null}]}) { length } }')
    assert data == {'scan': {'length': 0}}
    data = dsclient.execute(
        '{ scan(filter: {inv: {ne: [{name: "zipcode"}, {value: null}]}}) { length } }'
    )
    assert data == {'scan': {'length': 0}}
    data = dsclient.execute(
        '{ scan(filter: {eq: [{name: "state"} {string: "CA", cast: "string"}]}) { length } }'
    )
    assert data == {'scan': {'length': 2647}}
    data = dsclient.execute(
        '{ scan(filter: {eq: [{name: "state"} {string: ["CA", "OR"]}]}) { length } }'
    )
    assert data == {'scan': {'length': 3131}}
    with pytest.raises(ValueError, match="conflicting inputs"):
        dsclient.execute('{ scan(filter: {name: "state", string: "CA"}) { length } }')
    with pytest.raises(ValueError, match="name or alias"):
        dsclient.execute('{ scan(columns: {}) { length } }')
    data = dsclient.execute(
        '''{ scan(filter: {eq: [{name: "state"}, {string: "CA"}]})
        { scan(filter: {eq: [{name: "county"}, {string: "Santa Clara"}]})
        { length row { county } } } }'''
    )
    assert data == {'scan': {'scan': {'length': 108, 'row': {'county': 'Santa Clara'}}}}
    data = dsclient.execute(
        '''{ scan(filter: {or: [{eq: [{name: "state"}, {string: "CA"}]},
        {eq: [{name: "county"}, {string: "Santa Clara"}]}]}) { length } }'''
    )
    assert data == {'scan': {'length': 2647}}


def test_federation(fedclient):
    data = fedclient.execute(
        '{ _service { sdl } zipcodes { __typename length } zipDb { __typename length } }'
    )
    assert data['_service']['sdl']
    assert data['zipcodes'] == {'__typename': 'ZipcodesTable', 'length': 41700}
    assert data['zipDb'] == {'__typename': 'ZipDbTable', 'length': 42724}

    data = fedclient.execute(
        '''{ zipcodes { scan(columns: {name: "zipcode", cast: "int64"}) {
        join(right: "zip_db", keys: "zipcode", rightKeys: "zip") { length schema { names } } } } }'''
    )
    table = data['zipcodes']['scan']['join']
    assert table['length'] == 41700
    assert set(table['schema']['names']) > {'zipcode', 'timezone', 'latitude'}
    data = fedclient.execute(
        '''{ zipcodes { scan(columns: {alias: "zip", name: "zipcode", cast: "int64"}) {
        join(right: "zip_db", keys: "zip", joinType: "right outer") { length schema { names } } } } }'''
    )
    table = data['zipcodes']['scan']['join']
    assert table['length'] == 42724
    assert set(table['schema']['names']) > {'zip', 'timezone', 'latitude'}

    data = fedclient.execute(
        '''{ _entities(representations: {__typename: "ZipcodesTable", zipcode: 90001}) {
        ... on ZipcodesTable { length row { state } schema { names } } } }'''
    )
    assert data == {
        '_entities': [{'length': 1, 'row': {'state': 'CA'}, 'schema': {'names': ['state']}}]
    }
    data = fedclient.execute(
        '''{ states { filter(state: {eq: "CA"}) { columns { indices {
        takeFrom(field: "zipcodes") { __typename column(name: "state") { length } } } } } } }'''
    )
    table = data['states']['filter']['columns']['indices']['takeFrom']
    assert table == {'__typename': 'ZipcodesTable', 'column': {'length': 2647}}
