import asyncio
import pytest
from graphique import middleware
from .conftest import load


def test_extensions():
    ext = middleware.MetricsExtension(type('', (), {'context': {}}))
    for name in ('operation', 'parse', 'validate'):
        assert list(getattr(ext, 'on_' + name)()) == [None]
    assert set(ext.get_results()['metrics']) == {'duration', 'execution'}


def test_filter(dsclient):
    data = dsclient.execute('{ column(name: "state") { count } }')
    assert data == {'column': {'count': 41700}}
    data = dsclient.execute('{ count row { state } }')
    assert data == {'count': 41700, 'row': {'state': 'NY'}}
    data = dsclient.execute('{ filter(state: {eq: ["CA", "NY"]}) { count } }')
    assert data == {'filter': {'count': 4852}}
    data = dsclient.execute('{ filter(state: {ne: "CA"}) { count } }')
    assert data == {'filter': {'count': 39053}}
    data = dsclient.execute('{ filter { count } }')
    assert data == {'filter': {'count': 41700}}
    data = dsclient.execute('{ filter(state: {ne: null}) { count } }')
    assert data == {'filter': {'count': 41700}}
    data = dsclient.execute('{ dropNull { count } }')
    assert data == {'dropNull': {'count': 41700}}


def test_search(dsclient):
    data = dsclient.execute('{ filter(zipcode: {lt: 10000}) { count } }')
    assert data == {'filter': {'count': 3224}}
    data = dsclient.execute('{ filter(zipcode: {}) { count } }')
    assert data == {'filter': {'count': 41700}}
    data = dsclient.execute('{ filter(zipcode: {}) { row { zipcode } } }')
    assert data == {'filter': {'row': {'zipcode': 501}}}
    data = dsclient.execute("""{ filter(zipcode: {gt: 90000}) { filter(state: {eq: "CA"}) {
        count } } }""")
    assert data == {'filter': {'filter': {'count': 2647}}}
    data = dsclient.execute("""{ filter(zipcode: {gt: 90000}) { filter(state: {eq: "CA"}) {
        count row { zipcode } } } }""")
    assert data == {'filter': {'filter': {'count': 2647, 'row': {'zipcode': 90001}}}}
    data = dsclient.execute("""{ filter(zipcode: {lt: 90000}) { filter(state: {eq: "CA"}) {
        group(by: "county") { count } } } }""")
    assert data == {'filter': {'filter': {'group': {'count': 0}}}}


def test_slice(dsclient):
    data = dsclient.execute('{ slice(limit: 3) { count } }')
    assert data == {'slice': {'count': 3}}
    data = dsclient.execute('{ slice(offset: -3) { count } }')
    assert data == {'slice': {'count': 3}}
    data = dsclient.execute('{ slice { count } }')
    assert data == {'slice': {'count': 41700}}
    data = dsclient.execute('{ take(indices: [0]) { row { zipcode } } }')
    assert data == {'take': {'row': {'zipcode': 501}}}
    data = dsclient.execute('{ any many: any(length: 50000)}')
    assert data == {'any': True, 'many': False}


def test_group(dsclient):
    data = dsclient.execute("""{ group(by: ["state"], rowNumber: "idx", aggregate: {min: {name: "county"}}) { 
        order(by: "idx") { row { state county } } } }""")
    assert data == {'group': {'order': {'row': {'state': 'NY', 'county': 'Albany'}}}}
    data = dsclient.execute("""{ group(by: ["state"], counts: "c") {
        column(name: "c") { ... on LongColumn { values } } } }""")
    assert 2205 in data['group']['column']['values']
    data = dsclient.execute("""{ group(by: ["state"], aggregate: {first: {name: "county"}}) {
        columns { county { values } } } }""")
    assert 'Suffolk' in data['group']['columns']['county']['values']


def test_fragments(partclient):
    data = partclient.execute('{ group(by: ["north", "west"]) { columns { north { values } } } }')
    data = partclient.execute(
        '{ group(by: ["north", "west"], counts: "c") { column(name: "c") { ... on LongColumn { values } } } }'
    )
    assert data == {'group': {'column': {'values': [9301, 11549, 11549, 9301]}}}
    data = partclient.execute('{ rank(by: "north") { row { north } } }')
    assert data == {'rank': {'row': {'north': 0}}}
    data = partclient.execute('{ rank(by: ["-north", "-zipcode"]) { row { zipcode } } }')
    assert data == {'rank': {'row': {'zipcode': 99950}}}
    data = partclient.execute('{ order(by: "north", limit: 1) { row { north } } }')
    assert data == {'order': {'row': {'north': 0}}}
    data = partclient.execute(
        '{ group(by: ["north"], aggregate: {max: {name: "zipcode"}}) { row { north zipcode } } }'
    )
    assert data['group']['row']['zipcode'] >= 96898
    data = partclient.execute(
        '{ group(by: [], aggregate: {min: {name: "state"}}) { count row { state } } }'
    )
    assert data == {'group': {'count': 1, 'row': {'state': 'AK'}}}
    data = partclient.execute(
        """{ group(by: ["north", "west"], aggregate: {collect: {name: "city", distinct: true}, mean: {name: "zipcode"}}) {
        count column(name: "city") { type } } }"""
    )
    assert data == {'group': {'count': 4, 'column': {'type': 'list<l: string>'}}}
    data = partclient.execute(
        '{ group(by: "north", counts: "c") { column(name: "c") { ... on LongColumn { values } } } }'
    )
    assert data == {'group': {'column': {'values': [20850, 20850]}}}


def test_schema(dsclient):
    schema = dsclient.execute('{ schema { names types partitioning } }')['schema']
    assert set(schema['names']) >= {'zipcode', 'state', 'county'}
    assert set(schema['types']) >= {'int32', 'string'}
    assert len(schema['partitioning']) in (0, 6)
    data = dsclient.execute('{ scan(filter: {}) { type } }')
    assert data == {'scan': {'type': 'Table'}}
    data = dsclient.execute('{ scan(columns: {name: "zipcode"}) { type } }')
    assert data == {'scan': {'type': 'Nodes'}}


def test_scan(dsclient):
    data = dsclient.execute(
        '{ scan(columns: {name: "zipcode", alias: "zip"}) { column(name: "zip") { type } } }'
    )
    assert data == {'scan': {'column': {'type': 'int32'}}}
    data = dsclient.execute('{ scan(filter: {eq: [{name: "county"}, {name: "state"}]}) { count } }')
    assert data == {'scan': {'count': 0}}
    data = dsclient.execute('{ scan(filter: {eq: [{name: "zipcode"}, {value: null}]}) { count } }')
    assert data == {'scan': {'count': 0}}
    data = dsclient.execute(
        '{ scan(filter: {inv: {ne: [{name: "zipcode"}, {value: null}]}}) { count } }'
    )
    assert data == {'scan': {'count': 0}}
    data = dsclient.execute(
        '{ scan(filter: {eq: [{name: "state"} {value: "CA", cast: "string"}]}) { count } }'
    )
    assert data == {'scan': {'count': 2647}}
    data = dsclient.execute(
        '{ scan(filter: {eq: [{name: "state"} {value: ["CA", "OR"]}]}) { count } }'
    )
    assert data == {'scan': {'count': 3131}}
    with pytest.raises(ValueError, match="conflicting inputs"):
        dsclient.execute('{ scan(filter: {name: "state", value: "CA"}) { count } }')
    with pytest.raises(ValueError, match="name or alias"):
        dsclient.execute('{ scan(columns: {}) { count } }')
    data = dsclient.execute("""{ scan(filter: {eq: [{name: "state"}, {value: "CA"}]})
        { scan(filter: {eq: [{name: "county"}, {value: "Santa Clara"}]})
        { count row { county } } } }""")
    assert data == {'scan': {'scan': {'count': 108, 'row': {'county': 'Santa Clara'}}}}
    data = dsclient.execute("""{ scan(filter: {or: [{eq: [{name: "state"}, {value: "CA"}]},
        {eq: [{name: "county"}, {value: "Santa Clara"}]}]}) { count } }""")
    assert data == {'scan': {'count': 2647}}


def test_rank(partclient):
    data = partclient.execute('{ rank(by: ["state"]) { count row { state } } }')
    assert data == {'rank': {'count': 273, 'row': {'state': 'AK'}}}
    data = partclient.execute('{ rank(by: ["-state", "-county"]) { count row { state county } } }')
    assert data == {'rank': {'count': 4, 'row': {'state': 'WY', 'county': 'Weston'}}}
    data = partclient.execute('{ order(by: "state", limit: 3) { columns { state { values } } } }')
    assert data == {'order': {'columns': {'state': {'values': ['AK'] * 3}}}}
    data = partclient.execute('{ rank(by: "north") { count } }')
    assert data == {'rank': {'count': 20850}}
    data = partclient.execute('{ rank(by: "north", max: 2) { count } }')
    assert data == {'rank': {'count': 41700}}
    data = partclient.execute('{ rank(by: ["north", "west"]) { count } }')
    assert data == {'rank': {'count': 9301}}
    data = partclient.execute('{ rank(by: ["north", "west"], max: 2) { count } }')
    assert data == {'rank': {'count': 20850}}
    data = partclient.execute('{ rank(by: ["north", "west"], max: 3) { count } }')
    assert data == {'rank': {'count': 32399}}
    data = partclient.execute(
        '{ rank(by: ["north", "state"], max: 2) { columns { state { unique { values } } } } }'
    )
    assert data == {'rank': {'columns': {'state': {'unique': {'values': ['AL', 'AR']}}}}}
    data = partclient.execute('{ order(by: "north", limit: 3) { count } }')
    assert data == {'order': {'count': 3}}
    data = partclient.execute('{ order(by: "north", limit: 50000) { count } }')
    assert data == {'order': {'count': 41700}}


def test_root():
    app = load('zipcodes.parquet', FEDERATED='test')
    assert asyncio.run(app.get_root_value(None)) is app.root_value
    assert app.root_value.test
    with pytest.warns(UserWarning):
        assert load('nofields.parquet', FEDERATED='test')


def test_federation(fedclient):
    data = fedclient.execute(
        '{ _service { sdl } zipcodes { __typename count } zipDb { __typename count } }'
    )
    assert data['_service']['sdl']
    assert data['zipcodes'] == {'__typename': 'ZipcodesTable', 'count': 41700}
    assert data['zipDb'] == {'__typename': 'ZipDbTable', 'count': 42724}

    data = fedclient.execute("""{ zipcodes { scan(columns: {name: "zipcode", cast: "int64"}) {
        join(right: "zip_db", keys: "zipcode", rkeys: "zip") { count schema { names } } } } }""")
    table = data['zipcodes']['scan']['join']
    assert table['count'] == 41684
    assert set(table['schema']['names']) > {'zipcode', 'timezone', 'latitude'}
    data = fedclient.execute(
        """{ zipcodes { scan(columns: {alias: "zip", name: "zipcode", cast: "int64"}) {
        join(right: "zip_db", keys: "zip", how: "right") { count schema { names } } } } }"""
    )
    table = data['zipcodes']['scan']['join']
    assert table['count'] == 42724
    assert set(table['schema']['names']) > {'zip', 'timezone', 'latitude'}

    data = fedclient.execute(
        """{ _entities(representations: {__typename: "ZipcodesTable", zipcode: 90001}) {
        ... on ZipcodesTable { count type row { state } } } }"""
    )
    assert data == {'_entities': [{'count': 1, 'type': 'Nodes', 'row': {'state': 'CA'}}]}
    data = fedclient.execute("""{ states { filter(state: {eq: "CA"}) { columns { indices {
        takeFrom(field: "zipcodes") { __typename column(name: "state") { count } } } } } } }""")
    table = data['states']['filter']['columns']['indices']['takeFrom']
    assert table == {'__typename': 'ZipcodesTable', 'column': {'count': 2647}}


def test_sorted(fedclient):
    data = fedclient.execute(
        '{ states { filter(state: {eq: "CA"}, county: {eq: "Santa Clara"}) { count } } }'
    )
    assert data == {'states': {'filter': {'count': 108}}}
    data = fedclient.execute(
        '{ states { filter(state: {eq: ["CA", "OR"]}, county: {eq: "Santa Clara"}) { count } } }'
    )
    assert data == {'states': {'filter': {'count': 108}}}
    data = fedclient.execute(
        '{ states { filter(state: {le: "CA"}, county: {eq: "Santa Clara"}) { count } } }'
    )
    assert data == {'states': {'filter': {'count': 108}}}
    data = fedclient.execute('{ states { filter { filter(state: {eq: "CA"}) { count } } } }')
    assert data == {'states': {'filter': {'filter': {'count': 2647}}}}
