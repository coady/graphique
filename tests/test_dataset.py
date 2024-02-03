import asyncio
import pytest
import strawberry
from graphique import middleware
from .conftest import load


def test_extensions():
    context = strawberry.types.ExecutionContext(None, None, context={})
    ext = middleware.MetricsExtension(execution_context=context)
    for name in ('operation', 'parse', 'validate'):
        assert list(getattr(ext, 'on_' + name)()) == [None]
    assert set(ext.get_results()['metrics']) == {'duration', 'execution'}
    ext = middleware.ContextExtension(execution_context=context)
    assert ext.get_results() == {}


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
    data = dsclient.execute('{ dropNull { length } }')
    assert data == {'dropNull': {'length': 41700}}


def test_search(dsclient):
    data = dsclient.execute('{ filter(zipcode: {lt: 10000}) { length } }')
    assert data == {'filter': {'length': 3224}}
    data = dsclient.execute('{ filter(zipcode: {}) { length } }')
    assert data == {'filter': {'length': 41700}}
    data = dsclient.execute('{ filter(zipcode: {}) { row { zipcode } } }')
    assert data == {'filter': {'row': {'zipcode': 501}}}
    data = dsclient.execute("""{ filter(zipcode: {gt: 90000}) { filter(state: {eq: "CA"}) {
        length } } }""")
    assert data == {'filter': {'filter': {'length': 2647}}}
    data = dsclient.execute("""{ filter(zipcode: {gt: 90000}) { filter(state: {eq: "CA"}) {
        length row { zipcode } } } }""")
    assert data == {'filter': {'filter': {'length': 2647, 'row': {'zipcode': 90001}}}}
    data = dsclient.execute("""{ filter(zipcode: {lt: 90000}) { filter(state: {eq: "CA"}) {
        group(by: "county") { length } } } }""")
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
    data = dsclient.execute('{ any many: any(length: 50000)}')
    assert data == {'any': True, 'many': False}
    data = dsclient.execute('{ size }')
    assert data == {'size': None}
    data = dsclient.execute('{ slice { size } }')
    assert data == {'slice': {'size': 0}}


def test_group(dsclient):
    data = dsclient.execute(
        """{ group(by: ["state"], aggregate: {min: {name: "county"}}) { row { state county } } }"""
    )
    assert data == {'group': {'row': {'state': 'NY', 'county': 'Albany'}}}
    data = dsclient.execute("""{ group(by: ["state"], counts: "c") { slice(length: 1) {
        column(name: "c") { ... on LongColumn { values } } } } }""")
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
        """{ group(by: ["state"], aggregate: {mean: {name: "zipcode"}}) { slice(length: 1) {
        column(name: "zipcode") { ... on FloatColumn { values } } } } }"""
    )
    assert data == {'group': {'slice': {'column': {'values': [pytest.approx(12614.62721)]}}}}
    data = dsclient.execute(
        """{ group(by: ["state"], aggregate: {list: {name: "zipcode"}}) { aggregate(mean: {name: "zipcode"}) {
        slice(length: 1) { column(name: "zipcode") { ... on FloatColumn { values } } } } } }"""
    )
    assert data == {
        'group': {'aggregate': {'slice': {'column': {'values': [pytest.approx(12614.62721)]}}}}
    }
    data = dsclient.execute("""{ group(aggregate: {min: {alias: "st", name: "state"}}) {
        column(name: "st") { ... on StringColumn { values } } } }""")
    assert data == {'group': {'column': {'values': ['AK']}}}


def test_list(partclient):
    data = partclient.execute(
        """{ group(by: "state", aggregate: {distinct: {alias: "counties", name: "county"}}) {
        tables { row { state } column(name: "counties") { length } } } } """
    )
    (table,) = [table for table in data['group']['tables'] if table['row']['state'] == 'PR']
    assert table == {'row': {'state': 'PR'}, 'column': {'length': 78}}
    data = partclient.execute("""{ group(by: "north", aggregate: {distinct: {name: "west"}}) {
        tables { row { north } columns { west { length } } } } }""")
    tables = data['group']['tables']
    assert {table['row']['north'] for table in tables} == {0, 1}
    assert [table['columns'] for table in tables] == [{'west': {'length': 2}}] * 2


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
    data = partclient.execute('{ sort(by: "north", length: 1) { row { north } } }')
    assert data == {'sort': {'row': {'north': 0}}}
    data = partclient.execute(
        '{ group(by: ["north"], aggregate: {max: {name: "zipcode"}}) { row { north zipcode } } }'
    )
    assert data == {'group': {'row': {'north': 0, 'zipcode': 96898}}}
    data = partclient.execute(
        '{ group(by: [], aggregate: {min: {name: "state"}}) { length row { state } } }'
    )
    assert data == {'group': {'length': 1, 'row': {'state': 'AK'}}}
    data = partclient.execute(
        """{ group(by: ["north", "west"], aggregate: {distinct: {name: "city"}, mean: {name: "zipcode"}}) {
        length column(name: "city") { type } } }"""
    )
    assert data == {'group': {'length': 4, 'column': {'type': 'large_list<item: string>'}}}
    data = partclient.execute("""{ group(by: "north", aggregate: {countDistinct: {name: "west"}}) { 
        column(name: "west") { ... on LongColumn { values } } } }""")
    assert data == {'group': {'column': {'values': [2, 2]}}}
    data = partclient.execute(
        '{ group(by: "north", counts: "c") { column(name: "c") { ... on LongColumn { values } } } }'
    )
    assert data == {'group': {'column': {'values': [20850, 20850]}}}


def test_schema(dsclient):
    schema = dsclient.execute('{ schema { names types partitioning } }')['schema']
    assert set(schema['names']) >= {'zipcode', 'state', 'county'}
    assert set(schema['types']) >= {'int32', 'string'}
    assert len(schema['partitioning']) in (0, 6)
    assert dsclient.execute('{ type }')['type'] in {'FileSystemDataset', 'Scanner'}
    assert dsclient.execute('{ scan { type length } }')['scan']['type'] == 'Scanner'
    data = dsclient.execute('{ scan { type length l: length } }')
    assert data['scan']['type'] in {'Scanner', 'Table'}
    result = dsclient._execute('{ length optional { tables { length } } }')
    assert result.data == {'length': 41700, 'optional': None}
    assert len(result.errors) == 1


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
        '{ scan(filter: {eq: [{name: "state"} {value: "CA", cast: "string"}]}) { length } }'
    )
    assert data == {'scan': {'length': 2647}}
    data = dsclient.execute(
        '{ scan(filter: {eq: [{name: "state"} {value: ["CA", "OR"]}]}) { length } }'
    )
    assert data == {'scan': {'length': 3131}}
    with pytest.raises(ValueError, match="conflicting inputs"):
        dsclient.execute('{ scan(filter: {name: "state", value: "CA"}) { length } }')
    with pytest.raises(ValueError, match="name or alias"):
        dsclient.execute('{ scan(columns: {}) { length } }')
    data = dsclient.execute("""{ scan(filter: {eq: [{name: "state"}, {value: "CA"}]})
        { scan(filter: {eq: [{name: "county"}, {value: "Santa Clara"}]})
        { length row { county } } } }""")
    assert data == {'scan': {'scan': {'length': 108, 'row': {'county': 'Santa Clara'}}}}
    data = dsclient.execute("""{ scan(filter: {or: [{eq: [{name: "state"}, {value: "CA"}]},
        {eq: [{name: "county"}, {value: "Santa Clara"}]}]}) { length } }""")
    assert data == {'scan': {'length': 2647}}


def test_rank(partclient):
    data = partclient.execute('{ rank(by: ["state"]) { length row { state } } }')
    assert data == {'rank': {'length': 273, 'row': {'state': 'AK'}}}
    data = partclient.execute('{ rank(by: ["-state", "-county"]) { length row { state county } } }')
    assert data == {'rank': {'length': 4, 'row': {'state': 'WY', 'county': 'Weston'}}}
    data = partclient.execute('{ sort(by: "state", length: 3) { columns { state { values } } } }')
    assert data == {'sort': {'columns': {'state': {'values': ['AK'] * 3}}}}
    data = partclient.execute('{ rank(by: "north") { length } }')
    assert data == {'rank': {'length': 20850}}
    data = partclient.execute('{ rank(by: "north", max: 2) { length } }')
    assert data == {'rank': {'length': 41700}}
    data = partclient.execute('{ rank(by: ["north", "west"]) { length } }')
    assert data == {'rank': {'length': 9301}}
    data = partclient.execute('{ rank(by: ["north", "west"], max: 2) { length } }')
    assert data == {'rank': {'length': 20850}}
    data = partclient.execute('{ rank(by: ["north", "west"], max: 3) { length } }')
    assert data == {'rank': {'length': 32399}}
    data = partclient.execute(
        '{ rank(by: ["north", "state"], max: 2) { columns { state { unique { values } } } } }'
    )
    assert data == {'rank': {'columns': {'state': {'unique': {'values': ['AL', 'AR']}}}}}


def test_root():
    app = load('zipcodes.parquet', FEDERATED='test')
    assert asyncio.run(app.get_root_value(None)) is app.root_value
    assert app.root_value.test
    with pytest.warns(UserWarning):
        assert load('nofields.parquet', FEDERATED='test')


def test_federation(fedclient):
    data = fedclient.execute(
        '{ _service { sdl } zipcodes { __typename length } zipDb { __typename length } }'
    )
    assert data['_service']['sdl']
    assert data['zipcodes'] == {'__typename': 'ZipcodesTable', 'length': 41700}
    assert data['zipDb'] == {'__typename': 'ZipDbTable', 'length': 42724}

    data = fedclient.execute("""{ zipcodes { scan(columns: {name: "zipcode", cast: "int64"}) {
        join(right: "zip_db", keys: "zipcode", rightKeys: "zip") { length schema { names } } } } }""")
    table = data['zipcodes']['scan']['join']
    assert table['length'] == 41700
    assert set(table['schema']['names']) > {'zipcode', 'timezone', 'latitude'}
    data = fedclient.execute(
        """{ zipcodes { scan(columns: {alias: "zip", name: "zipcode", cast: "int64"}) {
        join(right: "zip_db", keys: "zip", joinType: "right outer") { length schema { names } } } } }"""
    )
    table = data['zipcodes']['scan']['join']
    assert table['length'] == 42724
    assert set(table['schema']['names']) > {'zip', 'timezone', 'latitude'}

    data = fedclient.execute(
        """{ _entities(representations: {__typename: "ZipcodesTable", zipcode: 90001}) {
        ... on ZipcodesTable { length row { state } schema { names } } } }"""
    )
    assert data == {
        '_entities': [{'length': 1, 'row': {'state': 'CA'}, 'schema': {'names': ['state']}}]
    }
    data = fedclient.execute("""{ states { filter(state: {eq: "CA"}) { columns { indices {
        takeFrom(field: "zipcodes") { __typename column(name: "state") { length } } } } } } }""")
    table = data['states']['filter']['columns']['indices']['takeFrom']
    assert table == {'__typename': 'ZipcodesTable', 'column': {'length': 2647}}


def test_sorted(fedclient):
    data = fedclient.execute(
        '{ states { filter(state: {eq: "CA"}, county: {eq: "Santa Clara"}) { length } } }'
    )
    assert data == {'states': {'filter': {'length': 108}}}
    data = fedclient.execute(
        '{ states { filter(state: {eq: ["CA", "OR"]}, county: {eq: "Santa Clara"}) { length } } }'
    )
    assert data == {'states': {'filter': {'length': 108}}}
    data = fedclient.execute(
        '{ states { filter(state: {le: "CA"}, county: {eq: "Santa Clara"}) { length } } }'
    )
    assert data == {'states': {'filter': {'length': 108}}}
    data = fedclient.execute('{ states { filter { filter(state: {eq: "CA"}) { length } } } }')
    assert data == {'states': {'filter': {'filter': {'length': 2647}}}}
