import asyncio
import json

import pytest

from graphique import middleware

from .conftest import load


def test_extensions():
    ext = middleware.MetricsExtension(type('', (), {'context': {}}))
    for name in ('operation', 'parse', 'validate'):
        assert list(getattr(ext, 'on_' + name)()) == [None]
    assert set(ext.get_results()['metrics']) == {'duration', 'execution'}


def test_search(dsclient):
    data = dsclient.execute('{ filter(zipcode: {lt: 10000}) { count } }')
    assert data == {'filter': {'count': 3224}}
    data = dsclient.execute('{ filter(zipcode: {}) { count } }')
    assert data == {'filter': {'count': 41700}}
    data = dsclient.execute("""{ filter(zipcode: {gt: 90000}) { filter(state: {eq: "CA"}) {
        count } } }""")
    assert data == {'filter': {'filter': {'count': 2647}}}
    data = dsclient.execute("""{ filter(zipcode: {gt: 90000}) { filter(state: {eq: "CA"}) {
        count row { zipcode } } } }""")
    assert data == {'filter': {'filter': {'count': 2647, 'row': {'zipcode': 90001}}}}
    data = dsclient.execute("""{ filter(where: {lt: [{name: "zipcode"}, {value: 90000}],
        eq: [{name: "state"}, {value: "CA"}]}) { count } }""")
    assert data == {'filter': {'count': 0}}


def test_fragments(dsclient):
    data = dsclient.execute('{ filter(north: {eq: 1}) { count } }')
    assert data == {'filter': {'count': 20850}}
    data = dsclient.execute('{ group(by: ["north", "west"]) { columns { north { values } } } }')
    data = dsclient.execute(
        '{ group(by: ["north", "west"], counts: "c") { column(name: "c") { ... on BigIntColumn { values } } } }'
    )
    assert data == {'group': {'column': {'values': [9301, 11549, 11549, 9301]}}}
    data = dsclient.execute('{ order(by: "north", limit: 1, dense: true) { row { north } } }')
    assert data == {'order': {'row': {'north': 0}}}
    data = dsclient.execute("""{ order(by: ["-north", "-zipcode"], limit: 1, dense: true) {
        row { zipcode } } }""")
    assert data == {'order': {'row': {'zipcode': 99950}}}
    data = dsclient.execute('{ order(by: "north", limit: 1) { row { north } } }')
    assert data == {'order': {'row': {'north': 0}}}
    data = dsclient.execute(
        '{ group(by: ["north"], aggregate: {max: {name: "zipcode"}}) { row { north zipcode } } }'
    )
    assert data['group']['row']['zipcode'] >= 96898
    data = dsclient.execute(
        '{ group(by: [], aggregate: {min: {name: "state"}}) { count row { state } } }'
    )
    assert data == {'group': {'count': 1, 'row': {'state': 'AK'}}}
    data = dsclient.execute(
        """{ group(by: ["north", "west"], aggregate: {collect: {name: "city", distinct: true}, mean: {name: "zipcode"}}) {
        count column(name: "city") { type } } }"""
    )
    assert data == {'group': {'count': 4, 'column': {'type': 'array<string>'}}}
    data = dsclient.execute(
        '{ group(by: "north", counts: "c") { column(name: "c") { ... on BigIntColumn { values } } } }'
    )
    assert data == {'group': {'column': {'values': [20850, 20850]}}}


def test_schema(dsclient):
    schema = dsclient.execute('{ schema { names types partitioning } }')['schema']
    assert len(schema['names']) == 8
    assert set(schema['types']) == {'float64', 'int32', 'string'}
    assert schema['partitioning'] == ['north', 'west']
    data = dsclient.execute('{ type }')
    assert data['type'].endswith('Dataset')


def test_order(dsclient):
    data = dsclient.execute("""{ order(by: "state", limit: 1, dense: true) {
        count row { state } } }""")
    assert data == {'order': {'count': 273, 'row': {'state': 'AK'}}}
    data = dsclient.execute("""{ order(by: ["-state", "-county"], limit: 1, dense: true) {
        count row { state county } } }""")
    assert data == {'order': {'count': 4, 'row': {'state': 'WY', 'county': 'Weston'}}}
    data = dsclient.execute('{ order(by: "state", limit: 3) { columns { state { values } } } }')
    assert data == {'order': {'columns': {'state': {'values': ['AK'] * 3}}}}
    data = dsclient.execute('{ order(by: "north", limit: 1, dense: true) { count } }')
    assert data == {'order': {'count': 20850}}
    data = dsclient.execute('{ order(by: "north", limit: 2, dense: true) { count } }')
    assert data == {'order': {'count': 41700}}
    data = dsclient.execute('{ order(by: ["north", "west"], limit: 1, dense: true) { count } }')
    assert data == {'order': {'count': 9301}}
    data = dsclient.execute('{ order(by: ["north", "west"], limit: 2, dense: true) { count } }')
    assert data == {'order': {'count': 20850}}
    data = dsclient.execute('{ order(by: ["north", "west"], limit: 3, dense: true) { count } }')
    assert data == {'order': {'count': 32399}}
    data = dsclient.execute("""{ order(by: ["north", "state"], limit: 2, dense: true)
        { columns { state { nunique(approx: true) } } } }""")
    assert data == {'order': {'columns': {'state': {'nunique': 2}}}}
    data = dsclient.execute('{ order(by: "north", limit: 3) { count } }')
    assert data == {'order': {'count': 3}}
    data = dsclient.execute('{ order(by: "north", limit: 50000) { count } }')
    assert data == {'order': {'count': 41700}}


def test_root():
    app = load('zipcodes.parquet', FEDERATED='test')
    assert asyncio.run(app.get_root_value(None)) is app.root_value
    assert app.root_value.test
    with pytest.warns(UserWarning):
        assert load('nofields.parquet', FEDERATED='test')
    app = load('zipcodes.parquet', COLUMNS=json.dumps(['state']))
    assert app.root_value.schema().names == ('state',)
    app = load('zipcodes.parquet', COLUMNS=json.dumps({'zipCode': 'zipcode'}))
    assert app.root_value.schema().names == ('zipCode',)
    app = load('zipcodes.parquet', FILTERS=json.dumps({'state': {'eq': 'CA'}}))
    assert app.root_value.count() == 2647
    app = load('zipcodes.parquet', FILTERS=json.dumps({}), COLUMNS=json.dumps({'state': 'state'}))
    assert app.root_value.schema().names == ('state',)


def test_federation(fedclient):
    data = fedclient.execute(
        '{ _service { sdl } zipcodes { __typename count } zipDb { __typename count } }'
    )
    assert data['_service']['sdl']
    assert data['zipcodes'] == {'__typename': 'ZipcodesTable', 'count': 41700}
    assert data['zipDb'] == {'__typename': 'ZipDbTable', 'count': 42724}

    with pytest.raises(ValueError):
        fedclient.execute("""{ zipcodes { join(right: "zip_db", keys: "zipcode") { type } } }""")
    data = fedclient.execute("""{ zipcodes { join(right: "zip_db", keys: "zipcode", rkeys: "zip")
        { count schema { names } } } }""")
    table = data['zipcodes']['join']
    assert table['count'] == 41684
    assert set(table['schema']['names']) > {'zipcode', 'timezone', 'latitude'}
    data = fedclient.execute("""{ zipcodes { join(right: "zip_db", keys: "zipcode", rkeys: "zip", how: "right")
        { count schema { names } } } }""")
    table = data['zipcodes']['join']
    assert table['count'] == 42724
    assert set(table['schema']['names']) > {'zip', 'timezone', 'latitude'}

    data = fedclient.execute(
        """{ _entities(representations: {__typename: "ZipcodesTable", zipcode: 90001}) {
        ... on ZipcodesTable { count type row { state } } } }"""
    )
    assert data == {'_entities': [{'count': 1, 'type': 'CachedTable', 'row': {'state': 'CA'}}]}
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
