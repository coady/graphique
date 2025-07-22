import pytest


def test_slice(client):
    data = client.execute('{ count slice(limit: 3) { columns { zipcode { values } } } }')
    assert data == {'count': 41700, 'slice': {'columns': {'zipcode': {'values': [501, 544, 601]}}}}
    data = client.execute('{ slice(offset: 1) { columns { zipcode { values } } } }')
    zipcodes = data['slice']['columns']['zipcode']['values']
    assert zipcodes[0] == 544
    assert len(zipcodes) == 41699
    data = client.execute('{ columns { zipcode { count } } }')
    assert data['columns']['zipcode']['count'] == 41700
    data = client.execute('{ columns { zipcode { count(mode: "only_null") } } }')
    assert data['columns']['zipcode']['count'] == 0
    data = client.execute('{ cache { count } }')
    assert data == {'cache': {'count': 41700}}


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
    data = client.execute('{ scan(filter: {isFinite: {name: "longitude"}}) { count } }')
    assert data == {'scan': {'count': 41700}}
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
        scan(filter: {gt: [{utf8: {length: {name: "city"}}}, {value: 23}]}) { count } } }""")
    assert data == {'filter': {'scan': {'count': 1}}}
    data = client.execute(
        '{ scan(columns: {utf8: {swapcase: {name: "city"}}, alias: "city"}) { row { city } } }'
    )
    assert data == {'scan': {'row': {'city': 'hOLTSVILLE'}}}
    data = client.execute(
        '{ scan(columns: {utf8: {capitalize: {name: "state"}}, alias: "state"}) { row { state } } }'
    )
    assert data == {'scan': {'row': {'state': 'Ny'}}}
    data = client.execute('{ scan(filter: {utf8: {isLower: {name: "city"}}}) { count } }')
    assert data == {'scan': {'count': 0}}
    data = client.execute('{ scan(filter: {utf8: {isTitle: {name: "city"}}}) { count } }')
    assert data == {'scan': {'count': 41700}}
    data = client.execute(
        """{ scan(columns: {alias: "city", substring: {match: {name: "city"}, pattern: "Mountain"}})
        { scan(filter: {name: "city"}) { count } } }"""
    )
    assert data == {'scan': {'scan': {'count': 88}}}
    data = client.execute(
        """{ scan(filter: {substring: {match: {name: "city"}, pattern: "mountain", ignoreCase: true}})
        { count } }"""
    )
    assert data == {'scan': {'count': 88}}
    data = client.execute(
        """{ scan(filter: {substring: {match: {name: "city"}, pattern: "^Mountain", regex: true}})
        { count } }"""
    )
    assert data == {'scan': {'count': 42}}
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
        '{ scan(columns: {alias: "state", utf8: {ltrim: {name: "state"}}}) { count } }'
    )
    assert data == {'scan': {'count': 41700}}
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
    data = client.execute('{ filter { count } }')
    assert data == {'filter': {'count': 41700}}
    data = client.execute('{ filter(zipcode: {eq: 501}) { columns { zipcode { values } } } }')
    assert data == {'filter': {'columns': {'zipcode': {'values': [501]}}}}
    data = client.execute('{ filter(zipcode: {ne: 501}) { count } }')
    assert data['filter']['count'] == 41699

    data = client.execute('{ filter(zipcode: {ge: 99929}) { columns { zipcode { values } } } }')
    assert data == {'filter': {'columns': {'zipcode': {'values': [99929, 99950]}}}}
    data = client.execute('{ filter(zipcode: {lt: 601}) { columns { zipcode { values } } } }')
    assert data == {'filter': {'columns': {'zipcode': {'values': [501, 544]}}}}
    data = client.execute(
        '{ filter(zipcode: {gt: 501, le: 601}) { columns { zipcode { values } } } }'
    )
    assert data == {'filter': {'columns': {'zipcode': {'values': [544, 601]}}}}

    data = client.execute('{ filter(zipcode: {eq: []}) { count } }')
    assert data == {'filter': {'count': 0}}
    data = client.execute('{ filter(zipcode: {eq: [0]}) { count } }')
    assert data == {'filter': {'count': 0}}
    data = client.execute(
        '{ filter(zipcode: {eq: [501, 601]}) { columns { zipcode { values } } } }'
    )
    assert data == {'filter': {'columns': {'zipcode': {'values': [501, 601]}}}}


def test_filter(client):
    data = client.execute('{ filter { count } }')
    assert data['filter']['count'] == 41700
    data = client.execute('{ filter(city: {eq: "Mountain View"}) { count } }')
    assert data['filter']['count'] == 11
    data = client.execute('{ filter(state: {ne: "CA"}) { count } }')
    assert data['filter']['count'] == 39053
    data = client.execute('{ filter(city: {eq: "Mountain View"}, state: {le: "CA"}) { count } }')
    assert data['filter']['count'] == 7
    data = client.execute('{ filter(state: {eq: null}) { columns { state { values } } } }')
    assert data['filter']['columns']['state']['values'] == []
    data = client.execute(
        '{ scan(filter: {le: [{abs: {name: "longitude"}}, {value: 66}]}) { count } }'
    )
    assert data['scan']['count'] == 30
    with pytest.raises(ValueError, match="optional, not nullable"):
        client.execute('{ filter(city: {le: null}) { count } }')


def test_scan(client):
    data = client.execute('{ scan(filter: {eq: [{name: "county"}, {name: "city"}]}) { count } }')
    assert data['scan']['count'] == 2805
    data = client.execute("""{ scan(filter: {or: [{eq: [{name: "state"}, {name: "county"}]},
        {eq: [{name: "county"}, {name: "city"}]}]}) { count } }""")
    assert data['scan']['count'] == 2805
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
        { scan(filter: {gt: [{name: "product"}, {value: 0}]}) { count } } }"""
    )
    assert data['scan']['scan']['count'] == 0
    data = client.execute(
        '{ scan(columns: {name: "zipcode", cast: "float"}) { column(name: "zipcode") { type } } }'
    )
    assert data['scan']['column']['type'] == 'float'
    data = client.execute(
        '{ scan(filter: {inv: {eq: [{name: "state"}, {value: "CA"}]}}) { count } }'
    )
    assert data == {'scan': {'count': 39053}}
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


def test_order(client):
    with pytest.raises(ValueError, match="is required"):
        client.execute('{ order { columns { state { values } } } }')
    data = client.execute('{ order(by: ["state"]) { columns { state { values } } } }')
    assert data['order']['columns']['state']['values'][0] == 'AK'
    data = client.execute('{ order(by: "-state") { columns { state { values } } } }')
    assert data['order']['columns']['state']['values'][0] == 'WY'
    data = client.execute('{ order(by: ["state"], limit: 1) { columns { state { values } } } }')
    assert data['order']['columns']['state']['values'] == ['AK']
    data = client.execute('{ order(by: "-state", limit: 1) { columns { state { values } } } }')
    assert data['order']['columns']['state']['values'] == ['WY']
    data = client.execute('{ order(by: ["state", "county"]) { columns { county { values } } } }')
    assert data['order']['columns']['county']['values'][0] == 'Aleutians East'
    data = client.execute(
        """{ order(by: ["-state", "-county"], limit: 1) { columns { county { values } } } }"""
    )
    assert data['order']['columns']['county']['values'] == ['Weston']
    data = client.execute('{ order(by: ["state"], limit: 2) { columns { state { values } } } }')
    assert data['order']['columns']['state']['values'] == ['AK', 'AK']
    data = client.execute(
        """{ group(by: ["state"], aggregate: {list: {name: "county"}}) { apply(list: {sort: {by: ["county"]}})
        { project(columns: {array: {value: {name: "county"}, offset: 0}, alias: "county"}) { row { state county } } } } }"""
    )
    assert data['group']['apply']['project']['row'] == {'state': 'NY', 'county': 'Albany'}
    data = client.execute(
        """{ group(by: ["state"], aggregate: {list: {name: "county"}}) { apply(list: {sort: {by: ["-county"], length: 1}})
        { project(columns: {array: {slice: {name: "county"}, limit: 0}, alias: "county"}) { row { state } } } } }"""
    )
    assert data['group']['apply']['project']['row'] == {'state': 'NY'}
    data = client.execute(
        """{ group(by: ["state"], aggregate: {list: {name: "county"}}) { apply(list: {sort: {by: "county", length: 2}})
        { row { state } column(name: "county") { ... on ListColumn { value { count } } } } } }"""
    )
    assert data['group']['apply'] == {'row': {'state': 'NY'}, 'column': {'value': {'count': 2}}}


def test_group(client):
    with pytest.raises(ValueError, match="list"):
        client.execute('{ group(by: ["state"]) { tables { count } } }')
    with pytest.raises(ValueError, match="cannot represent"):
        client.execute('{ group(by: "state", aggregate: {list: {name: "city"}}) { row { city } } }')
    data = client.execute(
        """{ group(by: ["state"], ordered: true, aggregate: {list: {name: "county"}}) { count tables { count
        columns { state { values } county { min max } } }
        project(columns: {array: {length: {name: "county"}}, alias: "c"}) {
        column(name: "c") { ... on LongColumn { values } } } } }"""
    )
    assert len(data['group']['tables']) == data['group']['count'] == 52
    table = data['group']['tables'][0]
    assert table['count'] == data['group']['project']['column']['values'][0] == 2205
    assert set(table['columns']['state']['values']) == {'NY'}
    assert table['columns']['county'] == {'min': 'Albany', 'max': 'Yates'}
    data = client.execute(
        """{ group(by: ["state", "county"], counts: "counts", aggregate: {list: {name: "city"}}) {
        scan(filter: {gt: [{name: "counts"}, {value: 200}]}) {
        project(columns: [{array: {mins: {name: "city"}}, alias: "min"}, {array: {maxs: {name: "city"}}, alias: "max"}]) {
        min: column(name: "min") { ... on StringColumn { values } }
        max: column(name: "max") { ... on StringColumn { values } } } } } }"""
    )
    agg = data['group']['scan']['project']
    assert agg['min']['values'] == ['Naval Anacost Annex', 'Alsip', 'Alief', 'Acton']
    assert agg['max']['values'] == ['Washington Navy Yard', 'Worth', 'Webster', 'Woodland Hills']
    data = client.execute(
        """{ group(by: ["state", "county"], counts: "c", aggregate: {list: [{name: "zipcode"}, {name: "latitude"}, {name: "longitude"}]}) {
        order(by: ["-c"], limit: 4) { project(columns: [{alias: "latitude", array: {sums: {name: "latitude"}}}, {alias: "longitude", array: {means: {name: "longitude"}}}]) {
        columns { latitude { values } longitude { values } }
        column(name: "zipcode") { type } } } } }"""
    )
    agg = data['group']['order']['project']
    assert agg['column']['type'] == 'list<l: int32>'
    assert all(latitude > 1000 for latitude in agg['columns']['latitude']['values'])
    assert all(77 > longitude > -119 for longitude in agg['columns']['longitude']['values'])
    data = client.execute("""{ scan(columns: {name: "zipcode", cast: "bool"})
        { group(by: ["state"], aggregate: {list: {name: "zipcode"}}) { slice(limit: 3) {
        project(columns: [{alias: "a", array: {anys: {name: "zipcode"}}}, {alias: "b", array: {alls: {name: "zipcode"}}}]) {
        a: column(name: "a") { ... on BooleanColumn { values } }
        b: column(name: "b") { ... on BooleanColumn { values } }
        column(name: "zipcode") { type } } } } } }""")
    assert data['scan']['group']['slice']['project'] == {
        'a': {'values': [True, True, True]},
        'b': {'values': [True, True, True]},
        'column': {'type': 'list<l: bool>'},
    }
    data = client.execute("""{ sc: group(by: ["state", "county"]) { count }
        cs: group(by: ["county", "state"]) { count } }""")
    assert data['sc']['count'] == data['cs']['count'] == 3216


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


def test_runs(client):
    data = client.execute("""{ runs(by: ["state"]) { project(columns: []) { count columns { state { values } }
        column(name: "county") { type } } } }""")
    agg = data['runs']['project']
    assert agg['count'] == 66
    assert agg['columns']['state']['values'][:3] == ['NY', 'PR', 'MA']
    assert agg['column']['type'] == 'list<l: string>'
    data = client.execute("""{ order(by: ["state", "longitude"]) {
        runs(by: ["state"], split: [{name: "longitude", gt: 1.0}]) {
        count columns { state { values } }
        column(name: "longitude") { type } } } }""")
    groups = data['order']['runs']
    assert groups['count'] == 62
    assert groups['columns']['state']['values'][:7] == ['AK'] * 7
    assert groups['column']['type'] == 'list<item: double>'
    data = client.execute("""{ runs(split: [{name: "state", lt: null}]) {
        count column(name: "state") {
        ... on ListColumn {values { ... on StringColumn { values } } } } } }""")
    assert data['runs']['count'] == 34
    assert data['runs']['column']['values'][0]['values'] == (['NY'] * 2) + (['PR'] * 176)
    data = client.execute("""{ runs(by: ["state"]) { order(by: ["state"]) {
        columns { state { values } } } } }""")
    assert data['runs']['order']['columns']['state']['values'][-2:] == ['WY', 'WY']
    data = client.execute("""{ runs(by: ["state"]) { slice(offset: 2, limit: 2) {
        project(columns: {array: {length: {name: "zipcode"}}, alias: "c"}) {
        column(name: "c") { ... on LongColumn { values } } columns { state { values } } } } } }""")
    agg = data['runs']['slice']['project']
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
    data = client.execute('{ runs(split: {name: "zipcode", lt: 0}) { count } }')
    assert data == {'runs': {'count': 1}}
    data = client.execute('{ runs(split: {name: "zipcode", lt: null}) { count } }')
    assert data == {'runs': {'count': 1}}


def test_rows(client):
    with pytest.raises(ValueError, match="out of bounds"):
        client.execute('{ row(index: 100000) { zipcode } }')
    data = client.execute('{ row { state } }')
    assert data == {'row': {'state': 'NY'}}
    data = client.execute('{ row(index: -1) { state } }')
    assert data == {'row': {'state': 'AK'}}
