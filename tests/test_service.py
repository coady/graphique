import pytest


def test_slice(client):
    data = client.execute('{ length slice(length: 3) { columns { zipcode { values } } } }')
    assert data == {'length': 41700, 'slice': {'columns': {'zipcode': {'values': [501, 544, 601]}}}}
    data = client.execute('{ slice(offset: 1) { columns { zipcode { values } } } }')
    zipcodes = data['slice']['columns']['zipcode']['values']
    assert zipcodes[0] == 544
    assert len(zipcodes) == 41699
    data = client.execute('{ slice(offset: -1, reverse: true) { columns { zipcode { values } } } }')
    assert data['slice']['columns']['zipcode']['values'] == [99950]
    data = client.execute('{ columns { zipcode { count } } }')
    assert data['columns']['zipcode']['count'] == 41700
    data = client.execute('{ columns { zipcode { count(mode: "only_null") } } }')
    assert data['columns']['zipcode']['count'] == 0


def test_ints(client):
    data = client.execute('{ columns { zipcode { values sum mean } } }')
    zipcodes = data['columns']['zipcode']
    assert len(zipcodes['values']) == 41700
    assert zipcodes['sum'] == 2066562337
    assert zipcodes['mean'] == pytest.approx(49557.849808)
    data = client.execute('{ columns { zipcode { values min max unique { values counts } } } }')
    zipcodes = data['columns']['zipcode']
    assert len(zipcodes['values']) == 41700
    assert zipcodes['min'] == 501
    assert zipcodes['max'] == 99950
    assert len(zipcodes['unique']['values']) == 41700
    assert set(zipcodes['unique']['counts']) == {1}
    data = client.execute('{ columns { zipcode { size } } }')
    assert data['columns']['zipcode']['size'] > 0


def test_floats(client):
    data = client.execute('{ columns { latitude { values sum mean } } }')
    latitudes = data['columns']['latitude']
    assert len(latitudes['values']) == 41700
    assert latitudes['sum'] == pytest.approx(1606220.07592)
    assert latitudes['mean'] == pytest.approx(38.518467)
    data = client.execute('{ columns { longitude { min max } } }')
    longitudes = data['columns']['longitude']
    assert longitudes['min'] == pytest.approx(-174.21333)
    assert longitudes['max'] == pytest.approx(-65.301389)
    data = client.execute('{ columns { latitude { quantile(q: [0.5]) } } }')
    (quantile,) = data['columns']['latitude']['quantile']
    assert quantile == pytest.approx(39.12054)
    data = client.execute(
        """{ scan(columns: {alias: "latitude", elementWise: {min: [{name: "latitude"}, {name: "longitude"}]}}) {
        columns { latitude { min } } } }"""
    )
    assert data == {'scan': {'columns': {'latitude': {'min': pytest.approx(-174.213333)}}}}
    data = client.execute(
        """{scan(columns: {alias: "l", setLookup: {digitize: [{name: "latitude"}, {value: [40]}]}}) {
        column(name: "l") { ... on LongColumn { unique { values counts } } } } }"""
    )
    assert data == {"scan": {"column": {"unique": {"values": [1, 0], "counts": [17955, 23745]}}}}
    data = client.execute(
        '{ scan(columns: {alias: "latitude", log: {logb: [{name: "latitude"}, {value: 3}]}}) { row { latitude } } }'
    )
    assert data == {'scan': {'row': {'latitude': pytest.approx(3.376188)}}}
    data = client.execute(
        '{ scan(columns: {alias: "latitude", rounding: {round: {name: "latitude"}}}) {row { latitude } } }'
    )
    assert data == {'scan': {'row': {'latitude': 41.0}}}
    data = client.execute(
        '{ scan(columns: {alias: "latitude", rounding: {round: {name: "latitude"}, multiple: 2.0}}) {row { latitude } } }'
    )
    assert data == {'scan': {'row': {'latitude': 40.0}}}
    data = client.execute(
        '{ scan(columns: {alias: "latitude", trig: {sin: {name: "latitude"}}}) {row { latitude } } }'
    )
    assert data == {'scan': {'row': {'latitude': pytest.approx(0.02273553)}}}
    data = client.execute('{ scan(filter: {isFinite: {name: "longitude"}}) { length } }')
    assert data == {'scan': {'length': 41700}}
    data = client.execute('{ column(name: "latitude", cast: "int32", safe: false) { type } }')
    assert data == {'column': {'type': 'int32'}}


def test_strings(client):
    data = client.execute("""{ columns {
        state { values unique { values counts } countDistinct }
        county { unique { length values } }
        city { min max }
    } }""")
    states = data['columns']['state']
    assert len(states['values']) == 41700
    assert len(states['unique']['values']) == states['countDistinct'] == 52
    assert sum(states['unique']['counts']) == 41700
    counties = data['columns']['county']
    assert len(counties['unique']['values']) == counties['unique']['length'] == 1920
    assert data['columns']['city'] == {'min': 'Aaronsburg', 'max': 'Zwolle'}
    data = client.execute("""{ filter(state: {eq: "CA"}) {
        scan(filter: {gt: [{utf8: {length: {name: "city"}}}, {value: 23}]}) { length } } }""")
    assert data == {'filter': {'scan': {'length': 1}}}
    data = client.execute(
        '{ scan(columns: {utf8: {swapcase: {name: "city"}}, alias: "city"}) { row { city } } }'
    )
    assert data == {'scan': {'row': {'city': 'hOLTSVILLE'}}}
    data = client.execute(
        '{ scan(columns: {utf8: {capitalize: {name: "state"}}, alias: "state"}) { row { state } } }'
    )
    assert data == {'scan': {'row': {'state': 'Ny'}}}
    data = client.execute('{ scan(filter: {utf8: {isLower: {name: "city"}}}) { length } }')
    assert data == {'scan': {'length': 0}}
    data = client.execute('{ scan(filter: {utf8: {isTitle: {name: "city"}}}) { length } }')
    assert data == {'scan': {'length': 41700}}
    data = client.execute(
        """{ scan(columns: {alias: "city", substring: {match: {name: "city"}, pattern: "Mountain"}})
        { scan(filter: {name: "city"}) { length } } }"""
    )
    assert data == {'scan': {'scan': {'length': 88}}}
    data = client.execute(
        """{ scan(filter: {substring: {match: {name: "city"}, pattern: "mountain", ignoreCase: true}})
        { length } }"""
    )
    assert data == {'scan': {'length': 88}}
    data = client.execute(
        """{ scan(filter: {substring: {match: {name: "city"}, pattern: "^Mountain", regex: true}})
        { length } }"""
    )
    assert data == {'scan': {'length': 42}}
    data = client.execute(
        """{ scan(columns: {alias: "idx", setLookup: {indexIn: [{name: "state"}, {value: ["CA", "OR"]}]}})
        { column(name: "idx") { ... on IntColumn { unique { values } } } } }"""
    )
    assert data == {'scan': {'column': {'unique': {'values': [None, 0, 1]}}}}


def test_string_methods(client):
    data = client.execute(
        """{ scan(columns: {alias: "split", substring: {split: {name: "city"}, pattern: "-", maxSplits: 1}}) {
        column(name: "split") { type } } }"""
    )
    assert data == {'scan': {'column': {'type': 'list<item: string>'}}}
    data = client.execute("""{ scan(columns: {alias: "split", substring: {split: {name: "city"}}}) {
        column(name: "split") { type } } }""")
    assert data == {'scan': {'column': {'type': 'list<item: string>'}}}
    data = client.execute(
        """{ scan(columns: {alias: "state", utf8: {trim: {name: "state"}, characters: "C"}}) {
        columns { state { values } } } }"""
    )
    states = data['scan']['columns']['state']['values']
    assert 'CA' not in states and 'A' in states
    data = client.execute(
        '{ scan(columns: {alias: "state", utf8: {ltrim: {name: "state"}}}) { length } }'
    )
    assert data == {'scan': {'length': 41700}}
    data = client.execute(
        """{ scan(columns: {alias: "state", utf8: {center: {name: "state"}, width: 4, padding: "_"}})
        { row { state } } }"""
    )
    assert data == {'scan': {'row': {'state': '_NY_'}}}
    data = client.execute(
        """{ scan(columns: {alias: "state", utf8: {replaceSlice: {name: "state"}, start: 0, stop: 2, replacement: ""}})
        { columns { state { unique { values } } } } }"""
    )
    assert data == {'scan': {'columns': {'state': {'unique': {'values': ['']}}}}}
    data = client.execute(
        '{ scan(columns: {alias: "state", utf8: {sliceCodeunits: {name: "state"}, start: 0, stop: 1}}) { row { state } } }'
    )
    assert data == {'scan': {'row': {'state': 'N'}}}
    data = client.execute(
        """{ scan(columns: {alias: "state", substring: {replace: {name: "state"}, pattern: "C", replacement: "A"}})
        { columns { state { values } } } }"""
    )
    assert 'AA' in data['scan']['columns']['state']['values']


def test_search(client):
    data = client.execute('{ schema { index } filter { length } }')
    assert data == {'schema': {'index': ['zipcode']}, 'filter': {'length': 41700}}
    data = client.execute('{ filter(zipcode: {eq: 501}) { columns { zipcode { values } } } }')
    assert data == {'filter': {'columns': {'zipcode': {'values': [501]}}}}
    data = client.execute('{ filter(zipcode: {ne: 501}) { length } }')
    assert data['filter']['length'] == 41699

    data = client.execute('{ filter(zipcode: {ge: 99929}) { columns { zipcode { values } } } }')
    assert data == {'filter': {'columns': {'zipcode': {'values': [99929, 99950]}}}}
    data = client.execute('{ filter(zipcode: {lt: 601}) { columns { zipcode { values } } } }')
    assert data == {'filter': {'columns': {'zipcode': {'values': [501, 544]}}}}
    data = client.execute(
        '{ filter(zipcode: {gt: 501, le: 601}) { columns { zipcode { values } } } }'
    )
    assert data == {'filter': {'columns': {'zipcode': {'values': [544, 601]}}}}

    data = client.execute('{ filter(zipcode: {eq: []}) { length } }')
    assert data == {'filter': {'length': 0}}
    data = client.execute('{ filter(zipcode: {eq: [0]}) { length } }')
    assert data == {'filter': {'length': 0}}
    data = client.execute(
        '{ filter(zipcode: {eq: [501, 601]}) { columns { zipcode { values } } } }'
    )
    assert data == {'filter': {'columns': {'zipcode': {'values': [501, 601]}}}}
    data = client.execute('{ slice(reverse: true) { filter(zipcode: {ge: 90000}) { length } } }')
    assert data == {'slice': {'filter': {'length': 4275}}}


def test_filter(client):
    data = client.execute('{ filter { length } }')
    assert data['filter']['length'] == 41700
    data = client.execute('{ filter(city: {eq: "Mountain View"}) { length } }')
    assert data['filter']['length'] == 11
    data = client.execute('{ filter(state: {ne: "CA"}) { length } }')
    assert data['filter']['length'] == 39053
    data = client.execute('{ filter(city: {eq: "Mountain View"}, state: {le: "CA"}) { length } }')
    assert data['filter']['length'] == 7
    data = client.execute('{ filter(state: {eq: null}) { columns { state { values } } } }')
    assert data['filter']['columns']['state']['values'] == []
    data = client.execute(
        '{ scan(filter: {le: [{abs: {name: "longitude"}}, {value: 66}]}) { length } }'
    )
    assert data['scan']['length'] == 30
    with pytest.raises(ValueError, match="optional, not nullable"):
        client.execute('{ filter(city: {le: null}) { length } }')


def test_scan(client):
    data = client.execute('{ scan(filter: {eq: [{name: "county"}, {name: "city"}]}) { length } }')
    assert data['scan']['length'] == 2805
    data = client.execute("""{ scan(filter: {or: [{eq: [{name: "state"}, {name: "county"}]},
        {eq: [{name: "county"}, {name: "city"}]}]}) { length } }""")
    assert data['scan']['length'] == 2805
    data = client.execute(
        """{ scan(columns: {alias: "zipcode", add: [{name: "zipcode"}, {name: "zipcode"}]})
        { columns { zipcode { min } } } }"""
    )
    assert data['scan']['columns']['zipcode']['min'] == 1002
    data = client.execute(
        """{ scan(columns: {alias: "zipcode", subtract: [{name: "zipcode"}, {name: "zipcode"}]})
        { columns { zipcode { unique { values } } } } }"""
    )
    assert data['scan']['columns']['zipcode']['unique']['values'] == [0]
    data = client.execute(
        """{ scan(columns: {alias: "product", multiply: [{name: "latitude"}, {name: "longitude"}]})
        { scan(filter: {gt: [{name: "product"}, {value: 0}]}) { length } } }"""
    )
    assert data['scan']['scan']['length'] == 0
    data = client.execute(
        '{ scan(columns: {name: "zipcode", cast: "float"}) { column(name: "zipcode") { type } } }'
    )
    assert data['scan']['column']['type'] == 'float'
    data = client.execute(
        '{ scan(filter: {inv: {eq: [{name: "state"}, {value: "CA"}]}}) { length } }'
    )
    assert data == {'scan': {'length': 39053}}
    data = client.execute(
        '{ scan(columns: {name: "latitude", cast: "int32", safe: false}) { column(name: "latitude") { type } } }'
    )
    assert data == {'scan': {'column': {'type': 'int32'}}}
    data = client.execute(
        """{ scan(columns: {alias: "longitude", elementWise: {max: [{name: "longitude"}, {name: "latitude"}]}})
        { columns { longitude { min } } } }"""
    )
    assert data['scan']['columns']['longitude']['min'] == pytest.approx(17.963333)
    data = client.execute(
        """{ scan(columns: {alias: "latitude", elementWise: {min: [{name: "longitude"}, {name: "latitude"}]}})
        { columns { latitude { max } } } }"""
    )
    assert data['scan']['columns']['latitude']['max'] == pytest.approx(-65.301389)
    data = client.execute(
        """{ scan(columns: {alias: "state", elementWise: {min: [{name: "state"}, {name: "county"}], skipNulls: false}})
        { columns { state { values } } } }"""
    )
    assert data['scan']['columns']['state']['values'][0] == 'NY'


def test_apply(client):
    data = client.execute(
        """{ scan(columns: {alias: "city", substring: {find: {name: "city"}, pattern: "mountain"}})
        { column(name: "city") { ... on IntColumn { unique { values } } } } }"""
    )
    assert data['scan']['column']['unique']['values'] == [-1]
    data = client.execute(
        """{ scan(columns: {alias: "city", substring: {count: {name: "city"}, pattern: "mountain", ignoreCase: true}})
        { column(name: "city") { ... on IntColumn { unique { values } } } } }"""
    )
    assert data['scan']['column']['unique']['values'] == [0, 1]
    data = client.execute("""{ scan(columns: {alias: "state", binary: {joinElementWise: [
        {name: "state"}, {name: "county"}, {value: " "}]}}) { columns { state { values } } } }""")
    assert data['scan']['columns']['state']['values'][0] == 'NY Suffolk'
    data = client.execute("""{ apply(cumulativeSum: {name: "zipcode", skipNulls: false})
        { columns { zipcode { value(index: -1) } } } }""")
    assert data == {'apply': {'columns': {'zipcode': {'value': 2066562337}}}}
    data = client.execute("""{ apply(cumulativeSum: {name: "zipcode", checked: true})
        { columns { zipcode { value(index: -1) } } } }""")
    assert data == {'apply': {'columns': {'zipcode': {'value': 2066562337}}}}
    data = client.execute(
        '{ apply(pairwiseDiff: {name: "zipcode"}) { columns { zipcode { value } } } }'
    )
    assert data == {'apply': {'columns': {'zipcode': {'value': None}}}}
    data = client.execute('{ apply(rank: {name: "zipcode"}) { row { zipcode } } }')
    assert data == {'apply': {'row': {'zipcode': 1}}}
    data = client.execute(
        """{ apply(rank: {name: "zipcode", sortKeys: "descending", nullPlacement: "at_start", tiebreaker: "dense"})
        { row { zipcode } } }"""
    )
    assert data == {'apply': {'row': {'zipcode': 41700}}}


def test_sort(client):
    with pytest.raises(ValueError, match="is required"):
        client.execute('{ sort { columns { state { values } } } }')
    data = client.execute('{ sort(by: ["state"]) { columns { state { values } } } }')
    assert data['sort']['columns']['state']['values'][0] == 'AK'
    data = client.execute('{ sort(by: "-state") { columns { state { values } } } }')
    assert data['sort']['columns']['state']['values'][0] == 'WY'
    data = client.execute('{ sort(by: ["state"], length: 1) { columns { state { values } } } }')
    assert data['sort']['columns']['state']['values'] == ['AK']
    data = client.execute('{ sort(by: "-state", length: 1) { columns { state { values } } } }')
    assert data['sort']['columns']['state']['values'] == ['WY']
    data = client.execute('{ sort(by: ["state", "county"]) { columns { county { values } } } }')
    assert data['sort']['columns']['county']['values'][0] == 'Aleutians East'
    data = client.execute(
        """{ sort(by: ["-state", "-county"], length: 1) { columns { county { values } } } }"""
    )
    assert data['sort']['columns']['county']['values'] == ['Weston']
    data = client.execute('{ sort(by: ["state"], length: 2) { columns { state { values } } } }')
    assert data['sort']['columns']['state']['values'] == ['AK', 'AK']
    data = client.execute(
        """{ group(by: ["state"], aggregate: {list: {name: "county"}}) { apply(list: {sort: {by: ["county"]}})
        { aggregate(first: [{name: "county"}]) { row { state county } } } } }"""
    )
    assert data['group']['apply']['aggregate']['row'] == {'state': 'NY', 'county': 'Albany'}
    data = client.execute(
        """{ group(by: ["state"], aggregate: {list: {name: "county"}}) { apply(list: {sort: {by: ["-county"], length: 1}})
        { aggregate(first: [{name: "county"}]) { row { state county } } } } }"""
    )
    assert data['group']['apply']['aggregate']['row'] == {'state': 'NY', 'county': 'Yates'}
    data = client.execute(
        """{ group(by: ["state"], aggregate: {list: {name: "county"}}) { apply(list: {sort: {by: "county", length: 2}})
        { row { state } column(name: "county") { ... on ListColumn { value { length } } } } } }"""
    )
    assert data['group']['apply'] == {'row': {'state': 'NY'}, 'column': {'value': {'length': 2}}}


def test_group(client):
    with pytest.raises(ValueError, match="list"):
        client.execute('{ group(by: ["state"]) { tables { length } } }')
    with pytest.raises(ValueError, match="cannot represent"):
        client.execute('{ group(by: "state", aggregate: {list: {name: "city"}}) { row { city } } }')
    data = client.execute(
        """{ group(by: ["state"], aggregate: {list: {name: "county"}}) { length tables { length
        columns { state { values } county { min max } } }
        scan(columns: {list: {valueLength: {name: "county"}}, alias: "c"}) {
        column(name: "c") { ... on IntColumn { values } } } } }"""
    )
    assert len(data['group']['tables']) == data['group']['length'] == 52
    table = data['group']['tables'][0]
    assert table['length'] == data['group']['scan']['column']['values'][0] == 2205
    assert set(table['columns']['state']['values']) == {'NY'}
    assert table['columns']['county'] == {'min': 'Albany', 'max': 'Yates'}
    data = client.execute(
        """{ group(by: ["state", "county"], counts: "counts", aggregate: {list: {name: "city"}}) {
        scan(filter: {gt: [{name: "counts"}, {value: 200}]}) {
        aggregate(min: [{name: "city", alias: "min"}], max: [{name: "city", alias: "max"}]) {
        min: column(name: "min") { ... on StringColumn { values } }
        max: column(name: "max") { ... on StringColumn { values } } } } } }"""
    )
    agg = data['group']['scan']['aggregate']
    assert agg['min']['values'] == ['Naval Anacost Annex', 'Alsip', 'Alief', 'Acton']
    assert agg['max']['values'] == ['Washington Navy Yard', 'Worth', 'Webster', 'Woodland Hills']
    data = client.execute(
        """{ group(by: ["state", "county"], counts: "c", aggregate: {list: [{name: "zipcode"}, {name: "latitude"}, {name: "longitude"}]}) {
        sort(by: ["-c"], length: 4) { aggregate(sum: [{name: "latitude"}], mean: [{name: "longitude"}]) {
        columns { latitude { values } longitude { values } }
        column(name: "zipcode") { type } } } } }"""
    )
    agg = data['group']['sort']['aggregate']
    assert agg['column']['type'] == 'list<item: int32>'
    assert all(latitude > 1000 for latitude in agg['columns']['latitude']['values'])
    assert all(77 > longitude > -119 for longitude in agg['columns']['longitude']['values'])
    data = client.execute("""{ scan(columns: {name: "zipcode", cast: "bool"})
        { group(by: ["state"], aggregate: {list: {name: "zipcode"}}) { slice(length: 3) {
        scan(columns: [{alias: "a", list: {any: {name: "zipcode"}}}, {alias: "b", list: {all: {name: "zipcode"}}}]) {
        a: column(name: "a") { ... on BooleanColumn { values } }
        b: column(name: "b") { ... on BooleanColumn { values } }
        column(name: "zipcode") { type } } } } } }""")
    assert data['scan']['group']['slice']['scan'] == {
        'a': {'values': [True, True, True]},
        'b': {'values': [True, True, True]},
        'column': {'type': 'list<item: bool>'},
    }
    data = client.execute("""{ sc: group(by: ["state", "county"]) { length }
        cs: group(by: ["county", "state"]) { length } }""")
    assert data['sc']['length'] == data['cs']['length'] == 3216


def test_flatten(client):
    data = client.execute(
        '{ group(by: "state", aggregate: {list: {name: "city"}}) { flatten { columns { city { type } } } } }'
    )
    assert data == {'group': {'flatten': {'columns': {'city': {'type': 'string'}}}}}
    data = client.execute(
        """{ group(by: "state", aggregate: {list: {name: "city"}}) { flatten(indices: "idx") { columns { city { type } }
        column(name: "idx") { ... on LongColumn { unique { values counts } } } } } }"""
    )
    idx = data['group']['flatten']['column']['unique']
    assert idx['values'] == list(range(52))
    assert sum(idx['counts']) == 41700
    assert idx['counts'][0] == 2205


def test_aggregate(client):
    data = client.execute(
        """{ group(by: ["state"] counts: "c", aggregate: {first: [{name: "county"}]
        countDistinct: [{name: "city", alias: "cd"}]}) { slice(length: 3) {
        c: column(name: "c") { ... on LongColumn { values } }
        cd: column(name: "cd") { ... on LongColumn { values } }
        columns { state { values } county { values } } } } }"""
    )
    assert data['group']['slice'] == {
        'c': {'values': [2205, 176, 703]},
        'cd': {'values': [1612, 99, 511]},
        'columns': {
            'state': {'values': ['NY', 'PR', 'MA']},
            'county': {'values': ['Suffolk', 'Adjuntas', 'Hampden']},
        },
    }
    data = client.execute(
        """{ group(by: ["state", "county"], aggregate: {list: {name: "city"}, min: {name: "city", alias: "first"}}) {
        aggregate(max: {name: "city", alias: "last"}) { slice(length: 3) {
        first: column(name: "first") { ... on StringColumn { values } }
        last: column(name: "last") { ... on StringColumn { values } } } } } }"""
    )
    assert data['group']['aggregate']['slice'] == {
        'first': {'values': ['Amagansett', 'Adjuntas', 'Aguada']},
        'last': {'values': ['Yaphank', 'Adjuntas', 'Aguada']},
    }


def test_runs(client):
    data = client.execute("""{ runs(by: ["state"]) { aggregate { length columns { state { values } }
        column(name: "county") { type } } } }""")
    agg = data['runs']['aggregate']
    assert agg['length'] == 66
    assert agg['columns']['state']['values'][:3] == ['NY', 'PR', 'MA']
    assert agg['column']['type'] == 'list<item: string>'
    data = client.execute("""{ sort(by: ["state", "longitude"]) {
        runs(by: ["state"], split: [{name: "longitude", gt: 1.0}]) {
        length columns { state { values } }
        column(name: "longitude") { type } } } }""")
    groups = data['sort']['runs']
    assert groups['length'] == 62
    assert groups['columns']['state']['values'][:7] == ['AK'] * 7
    assert groups['column']['type'] == 'list<item: double>'
    data = client.execute("""{ runs(split: [{name: "state", lt: null}]) {
        length column(name: "state") {
        ... on ListColumn {values { ... on StringColumn { values } } } } } }""")
    assert data['runs']['length'] == 34
    assert data['runs']['column']['values'][0]['values'] == (['NY'] * 2) + (['PR'] * 176)
    data = client.execute("""{ runs(by: ["state"]) { sort(by: ["state"]) {
        columns { state { values } } } } }""")
    assert data['runs']['sort']['columns']['state']['values'][-2:] == ['WY', 'WY']
    data = client.execute("""{ runs(by: ["state"]) { slice(offset: 2, length: 2) {
        aggregate(count: {name: "zipcode", alias: "c"}) {
        column(name: "c") { ... on LongColumn { values } } columns { state { values } } } } } }""")
    agg = data['runs']['slice']['aggregate']
    assert agg['column']['values'] == [701, 91]
    assert agg['columns']['state']['values'] == ['MA', 'RI']
    data = client.execute("""{ runs(by: ["state"]) {
        apply(list: {filter: {gt: [{name: "zipcode"}, {value: 90000}]}}) {
        column(name: "zipcode") { type } } } }""")
    assert data['runs']['apply']['column']['type'] == 'large_list<item: int32>'
    data = client.execute("""{ runs(by: ["state"], counts: "c") { filter(state: {eq: "NY"}) {
        column(name: "c") { ... on LongColumn { values } } columns { state { values } } } } }""")
    agg = data['runs']['filter']
    assert agg['column']['values'] == [2, 1, 2202]
    assert agg['columns']['state']['values'] == ['NY'] * 3
    data = client.execute('{ runs(split: {name: "zipcode", lt: 0}) { length } }')
    assert data == {'runs': {'length': 1}}
    data = client.execute('{ runs(split: {name: "zipcode", lt: null}) { length } }')
    assert data == {'runs': {'length': 1}}


def test_rows(client):
    with pytest.raises(ValueError, match="out of bounds"):
        client.execute('{ row(index: 100000) { zipcode } }')
    data = client.execute('{ row { state } }')
    assert data == {'row': {'state': 'NY'}}
    data = client.execute('{ row(index: -1) { state } }')
    assert data == {'row': {'state': 'AK'}}
