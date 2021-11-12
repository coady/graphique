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
    data = client.execute('{ columns { zipcode { count(notEqual: null) } } }')
    assert data['columns']['zipcode']['count'] == 41700
    data = client.execute('{ columns { zipcode { count(equal: null) } } }')
    assert data['columns']['zipcode']['count'] == 0
    data = client.execute('{ slice(length: 0) { columns { zipcode { any all } } } }')
    assert data['slice']['columns']['zipcode'] == {'any': None, 'all': None}


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
    data = client.execute('{ columns { zipcode { truthy: count count(equal: 0) } } }')
    zipcodes = data['columns']['zipcode']
    assert zipcodes['truthy'] == 41700
    assert zipcodes['count'] == 0
    data = client.execute('{ columns { zipcode { count(equal: 501) } } }')
    zipcodes = data['columns']['zipcode']
    assert zipcodes['count'] == 1
    data = client.execute('{ columns { zipcode { sort desc: sort(reverse: true)} } }')
    zipcodes = data['columns']['zipcode']
    assert zipcodes['sort'][0] == zipcodes['desc'][-1] == 501
    assert zipcodes['sort'][-1] == zipcodes['desc'][0] == 99950
    data = client.execute(
        '''{ column(name: "zipcode", apply: {subtract: "zipcode"})
        { ... on IntColumn { unique { values } } } }'''
    )
    assert data == {'column': {'unique': {'values': [0]}}}


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
    data = client.execute('{ columns { latitude { truthy: count count(equal: 0.0) } } }')
    latitudes = data['columns']['latitude']
    assert latitudes['truthy'] == 41700
    assert latitudes['count'] == 0
    data = client.execute('{ columns { latitude { quantile(q: [0.5]) } } }')
    (quantile,) = data['columns']['latitude']['quantile']
    assert quantile == pytest.approx(39.12054)
    data = client.execute(
        '''{ column(name: "latitude", apply: {minElementWise: "longitude"})
        { ... on FloatColumn { min } } }'''
    )
    assert data['column']['min'] == pytest.approx(-174.213333)
    data = client.execute(
        f'''{{ apply(int: {{name: "latitude", digitize: {list(range(90))}}})
        {{ columns {{ latitude {{ min max unique {{ length }} }} }} }} }}'''
    )
    latitudes = data['apply']['columns']['latitude']
    assert latitudes == {'min': 18.0, 'max': 72.0, 'unique': {'length': 52}}
    data = client.execute(
        '''{ slice(length: 1) { columns { latitude { logb(base: 3) { values } } } } }'''
    )
    assert data['slice']['columns']['latitude']['logb']['values'] == [pytest.approx(3.376188)]
    data = client.execute(
        '''{ filter(on: {float: {name: "latitude", isFinite: true}})
        { apply(float: {name: "latitude", round: true}) { row { latitude } } } }'''
    )
    assert data == {'filter': {'apply': {'row': {'latitude': 41.0}}}}
    data = client.execute(
        '''{ slice(length: 1) { columns { latitude { round(ndigits: 1) { values } } } } }'''
    )
    assert data == {'slice': {'columns': {'latitude': {'round': {'values': [40.8]}}}}}
    data = client.execute(
        '''{ slice(length: 1) { columns { latitude { round(multiple: 10) { values } } } } }'''
    )
    assert data == {'slice': {'columns': {'latitude': {'round': {'values': [40.0]}}}}}
    with pytest.raises(ValueError, match="only one"):
        client.execute('{ columns { latitude { round(ndigits: 1, multiple: 10) { length } } } }')


def test_strings(client):
    data = client.execute(
        '''{ columns {
        state { values unique { values counts } }
        county { unique { length values } }
        city { min max sort(length: 1), desc: sort(reverse: true, length: 1) }
    } }'''
    )
    states = data['columns']['state']
    assert len(states['values']) == 41700
    assert len(states['unique']['values']) == 52
    assert sum(states['unique']['counts']) == 41700
    counties = data['columns']['county']
    assert len(counties['unique']['values']) == counties['unique']['length'] == 1920
    cities = data['columns']['city']
    assert [cities['min']] == cities['sort'] == ['Aaronsburg']
    assert [cities['max']] == cities['desc'] == ['Zwolle']
    data = client.execute('{ columns { state { truthy: count count(equal: "") } } }')
    states = data['columns']['state']
    assert states['truthy'] == 41700
    assert states['count'] == 0
    data = client.execute('{ columns { state { count(equal: "CA") } } }')
    assert data == {'columns': {'state': {'count': 2647}}}
    data = client.execute(
        '''{ filter(query: {state: {equal: "CA"}}) {
        apply(string: {name: "city", utf8Length: true, alias: "size"}) {
        filter(on: {int: {name: "size", greater: 23}}) { length }
        column(name: "size") { ... on IntColumn { max } } } } }'''
    )
    assert data == {'filter': {'apply': {'filter': {'length': 1}, 'column': {'max': 24}}}}
    data = client.execute('{ apply(string: {name: "city", utf8Swapcase: true}) { row { city } } }')
    assert data == {'apply': {'row': {'city': 'hOLTSVILLE'}}}


def test_string_methods(client):
    data = client.execute(
        '''{ group(by: ["city"]) { columns { city { split {
        element { ... on StringColumn { count(equal: "New") } }
        count { ... on LongColumn { max } } } } } } }'''
    )
    cities = data['group']['columns']['city']
    assert cities['split'] == {'element': {'count': 177}, 'count': {'max': 6}}
    data = client.execute(
        '''{ group(by: ["city"]) { columns { city {
        split(pattern: "-", maxSplits: 1, reverse: true) {
        count { unique { values counts } } } } } } }'''
    )
    cities = data['group']['columns']['city']
    cities['split']['count']['unique'] == {'values': [1, 2], 'counts': [18718, 1]}
    data = client.execute(
        '''{ columns { city { split(pattern: "-", maxSplits: 1, regex: true)
        { count { values } } } } }'''
    )
    assert all(count > 0 for count in data['columns']['city']['split']['count']['values'])
    data = client.execute(
        '{ columns { state { utf8Trim { values } utf8Ltrim { values } utf8Rtrim { values } } } }'
    )
    states = data['columns']['state']
    assert 'CA' in states['utf8Trim']['values']
    assert 'CA' in states['utf8Ltrim']['values']
    assert 'CA' in states['utf8Rtrim']['values']
    data = client.execute(
        '''{ columns { state { utf8Trim(characters: "C") { values }
        utf8Ltrim(characters: "C") { values } utf8Rtrim(characters: "A") { values } } } }'''
    )
    states = data['columns']['state']
    assert 'A' in states['utf8Trim']['values']
    assert 'A' in states['utf8Ltrim']['values']
    assert 'C' in states['utf8Rtrim']['values']
    data = client.execute(
        '''{ columns { state { utf8Center(width: 4, padding: "_") { values }
        utf8Lpad(width: 3) { values } utf8Rpad(width: 3) { values } } } }'''
    )
    states = data['columns']['state']
    assert all(len(state.split('_')) == 3 for state in states['utf8Center']['values'])
    assert all(state.startswith(' ') for state in states['utf8Lpad']['values'])
    assert all(state.endswith(' ') for state in states['utf8Rpad']['values'])
    data = client.execute(
        '''{ columns { state { utf8ReplaceSlice(start: 0, stop: 2, replacement: "")
        { unique { values } } } } }'''
    )
    assert data['columns']['state']['utf8ReplaceSlice']['unique']['values'] == ['']
    data = client.execute(
        '''{ columns { state { replaceSubstring(pattern: "C", replacement: "A") { values } } } }'''
    )
    assert 'AA' in data['columns']['state']['replaceSubstring']['values']
    data = client.execute(
        '''{ group(by: "state") { column(name: "city") { ... on ListColumn
        { stringJoin(separator: ",") { ... on StringColumn { values } } } } } }'''
    )
    assert ','.join(['New York'] * 3) in data['group']['column']['stringJoin']['values'][0]


def test_search(client):
    data = client.execute('{ index search { length } }')
    assert data['index'] == ['zipcode']
    assert data['search']['length'] == 41700
    data = client.execute('{ search(zipcode: {equal: 501}) { columns { zipcode { values } } } }')
    assert data == {'search': {'columns': {'zipcode': {'values': [501]}}}}
    data = client.execute('{ search(zipcode: {notEqual: 501}) { length } }')
    assert data['search']['length'] == 41699

    data = client.execute(
        '{ search(zipcode: {greaterEqual: 99929}) { columns { zipcode { values } } } }'
    )
    assert data == {'search': {'columns': {'zipcode': {'values': [99929, 99950]}}}}
    data = client.execute('{ search(zipcode: {less: 601}) { columns { zipcode { values } } } }')
    assert data == {'search': {'columns': {'zipcode': {'values': [501, 544]}}}}
    data = client.execute(
        '{ search(zipcode: {greater: 501, lessEqual: 601}) { columns { zipcode { values } } } }'
    )
    assert data == {'search': {'columns': {'zipcode': {'values': [544, 601]}}}}

    data = client.execute('{ search(zipcode: {isIn: []}) { length } }')
    assert data == {'search': {'length': 0}}
    data = client.execute('{ search(zipcode: {isIn: [0]}) { length } }')
    assert data == {'search': {'length': 0}}
    data = client.execute(
        '{ search(zipcode: {isIn: [501, 601]}) { columns { zipcode { values } } } }'
    )
    assert data == {'search': {'columns': {'zipcode': {'values': [501, 601]}}}}
    with pytest.raises(ValueError, match="optional, not nullable"):
        client.execute('{ search(zipcode: null) { length } }')


def test_filter(client):
    data = client.execute('{ filter(query: {}) { length } }')
    assert data['filter']['length'] == 41700
    data = client.execute('{ filter(query: {city: {equal: "Mountain View"}}) { length } }')
    assert data['filter']['length'] == 11
    data = client.execute('{ filter(query: {state: {notEqual: "CA"}}) { length } }')
    assert data['filter']['length'] == 39053
    data = client.execute(
        '{ filter(query: {city: {equal: "Mountain View"}, state: {lessEqual: "CA"}}) { length } }'
    )
    assert data['filter']['length'] == 7
    data = client.execute(
        '{ filter(query: {state: {equal: null}}) { columns { state { values } } } }'
    )
    assert data['filter']['columns']['state']['values'] == []
    data = client.execute('{ filter(query: {}, invert: true) { length } }')
    assert data['filter']['length'] == 41700
    data = client.execute('{ filter(query: {state: {equal: "CA"}}, invert: true) { length } }')
    assert data['filter']['length'] == 39053
    data = client.execute(
        '{ filter(on: {string: {name: "city", matchSubstring: "Mountain"}}) { length } }'
    )
    assert data['filter']['length'] == 88
    data = client.execute(
        '''{ filter(on: {string: {name: "city", matchSubstring: "mountain", ignoreCase: true}})
        { length } }'''
    )
    assert data['filter']['length'] == 88
    data = client.execute(
        '''{ filter(on: {string: {name: "city", matchSubstring: "^mountain",
        ignoreCase: true, regex: true}}) { length } }'''
    )
    assert data['filter']['length'] == 42
    data = client.execute(
        '{ filter(on: {string: {name: "county", apply: {equal: "city"}}}) { length } }'
    )
    assert data['filter']['length'] == 2805
    data = client.execute('{ filter(on: {string: {name: "city", utf8IsLower: true}}) { length } }')
    assert data['filter']['length'] == 0
    data = client.execute('{ filter(on: {string: {name: "city", utf8IsTitle: true}}) { length } }')
    assert data['filter']['length'] == 41700
    data = client.execute('{ filter(on: {string: {name: "city"}}) { length } }')
    assert data['filter']['length'] == 41700
    data = client.execute(
        '''{apply(float: {name: "longitude", abs: true})
        { filter(on: {float: {name: "longitude", less: 66}}) { length } } }'''
    )
    assert data['apply']['filter']['length'] == 30
    with pytest.raises(ValueError, match="optional, not nullable"):
        client.execute('{ filter(on: {string: {name: "city", apply: {equal: null}}}) { length } }')
    data = client.execute(
        '''{ filter(on: {string: [{name: "state", apply: {equal: "county"}},
        {name: "county", apply: {equal: "city"}}]}, reduce: OR) { length } }'''
    )
    assert data['filter']['length'] == 2805


def test_apply(client):
    data = client.execute(
        '{ apply(int: {name: "zipcode", add: "zipcode"}) { columns { zipcode { min } } } }'
    )
    assert data['apply']['columns']['zipcode']['min'] == 1002
    data = client.execute(
        '''{ apply(int: {name: "zipcode", subtract: "zipcode"})
        { columns { zipcode { unique { values } } } } }'''
    )
    assert data['apply']['columns']['zipcode']['unique']['values'] == [0]
    data = client.execute(
        '''{ apply(float: {name: "latitude", multiply: "longitude", alias: "product"})
        { filter(on: {float: {name: "product", greater: 0}}) { length } } }'''
    )
    assert data['apply']['filter']['length'] == 0
    data = client.execute(
        '''{ apply(float: {name: "longitude", maxElementWise: "latitude"})
        { columns { longitude { min } } } }'''
    )
    assert data['apply']['columns']['longitude']['min'] == pytest.approx(17.963333)
    data = client.execute(
        '''{ apply(float: {name: "latitude", minElementWise: "longitude"})
        { columns { latitude { max } } } }'''
    )
    assert data['apply']['columns']['latitude']['max'] == pytest.approx(-65.301389)
    data = client.execute(
        '''{ apply(int: {name: "zipcode", cast: "float"})
        { column(name: "zipcode") { type } } }'''
    )
    assert data['apply']['column']['type'] == 'float'
    data = client.execute(
        '''{ apply(string: {name: "city", findSubstring: "mountain"})
        { column(name: "city") { ... on IntColumn { unique { values } } } } }'''
    )
    assert data['apply']['column']['unique']['values'] == [-1]
    data = client.execute(
        '''{ apply(string: {name: "city", countSubstring: "mountain", ignoreCase: true})
        { column(name: "city") { ... on IntColumn { unique { values } } } } }'''
    )
    assert data['apply']['column']['unique']['values'] == [0, 1]
    data = client.execute(
        '''{ apply(string: {name: "state", binaryJoinElementWise: ["county", "city"]})
        { columns { state { values } } } }'''
    )
    assert data['apply']['columns']['state']['values'][0] == 'NYHoltsvilleSuffolk'


def test_sort(client):
    with pytest.raises(ValueError, match="is required"):
        client.execute('{ sort { columns { state { values } } } }')
    data = client.execute('{ sort(by: ["state"]) { columns { state { values } } } }')
    assert data['sort']['columns']['state']['values'][0] == 'AK'
    data = client.execute('{ sort(by: ["state"], reverse: true) { columns { state { values } } } }')
    assert data['sort']['columns']['state']['values'][0] == 'WY'
    data = client.execute('{ sort(by: ["state"], length: 1) { columns { state { values } } } }')
    assert data['sort']['columns']['state']['values'] == ['AK']
    data = client.execute(
        '{ sort(by: ["state"], reverse: true, length: 1) { columns { state { values } } } }'
    )
    assert data['sort']['columns']['state']['values'] == ['WY']
    data = client.execute('{ sort(by: ["state", "county"]) { columns { county { values } } } }')
    assert data['sort']['columns']['county']['values'][0] == 'Aleutians East'
    data = client.execute(
        '''{ sort(by: ["state", "county"], reverse: true, length: 1)
        { columns { county { values } } } }'''
    )
    assert data['sort']['columns']['county']['values'] == ['Weston']
    data = client.execute('{ sort(by: ["state"], length: 2) { columns { state { values } } } }')
    assert data['sort']['columns']['state']['values'] == ['AK', 'AK']


def test_group(client):
    with pytest.raises(ValueError, match="is required"):
        client.execute('{ group { length } }')
    with pytest.raises(ValueError, match="list"):
        client.execute('{ group(by: ["state"]) { tables { length } } }')
    data = client.execute(
        '''{ group(by: ["state"]) { length tables { length
        columns { state { values } county { min max } } }
        aggregate(valueLength: {name: "county", alias: "c"}) {
        column(name: "c") { ... on LongColumn { values } } } } }'''
    )
    assert len(data['group']['tables']) == data['group']['length'] == 52
    table = data['group']['tables'][0]
    assert table['length'] == data['group']['aggregate']['column']['values'][0] == 2205
    assert set(table['columns']['state']['values']) == {'NY'}
    assert table['columns']['county'] == {'min': 'Albany', 'max': 'Yates'}
    data = client.execute(
        '''{ group(by: ["state", "county"], reverse: true, length: 3) {
        aggregate(first: [{name: "city", alias: "f"}], last: [{name: "city", alias: "l"}]) {
        f: column(name: "f") { ... on StringColumn { values } }
        l: column(name: "l") { ... on StringColumn { values } } } } }'''
    )
    assert data['group']['aggregate']['f']['values'] == ['Meyers Chuck', 'Ketchikan', 'Sitka']
    assert data['group']['aggregate']['l']['values'] == ['Point Baker', 'Ketchikan', 'Sitka']
    data = client.execute(
        '''{ group(by: ["state", "county"], reverse: true, counts: "counts") {
        filter(on: {int: {name: "counts", greaterEqual: 200}}) {
        aggregate(min: [{name: "city", alias: "min"}], max: [{name: "city", alias: "max"}]) {
        min: column(name: "min") { ... on StringColumn { values } }
        max: column(name: "max") { ... on StringColumn { values } } } } } }'''
    )
    agg = data['group']['filter']['aggregate']
    assert agg['min']['values'] == ['Acton', 'Alief', 'Alsip', 'Naval Anacost Annex']
    assert agg['max']['values'] == ['Woodland Hills', 'Webster', 'Worth', 'Washington Navy Yard']
    data = client.execute(
        '''{ group(by: ["state", "county"], counts: "c") {
        sort(by: ["c"], reverse: true, length: 4) {
        aggregate(sum: [{name: "latitude"}], mean: [{name: "longitude"}]) {
        columns { latitude { values } longitude { values } }
        column(name: "zipcode") { ... on ListColumn { count { values } } } } } } }'''
    )
    agg = data['group']['sort']['aggregate']
    assert agg['column']['count']['values'] == [525, 284, 242, 219]
    assert all(latitude > 1000 for latitude in agg['columns']['latitude']['values'])
    assert all(77 > longitude > -119 for longitude in agg['columns']['longitude']['values'])
    data = client.execute(
        '''{ group(by: ["state"], length: 3) {
        aggregate(any: [{name: "latitude"}], all: [{name: "longitude"}]) {
        lat: column(name: "latitude") { ... on BooleanColumn { values } }
        lng: column(name: "longitude") { ... on BooleanColumn { values } } } } }'''
    )
    assert data['group']['aggregate'] == {
        'lat': {'values': [True, True, True]},
        'lng': {'values': [True, True, True]},
    }
    data = client.execute(
        '''{ sc: group(by: ["state", "county"]) { length }
        cs: group(by: ["county", "state"]) { length } }'''
    )
    assert data['sc']['length'] == data['cs']['length'] == 3216
    data = client.execute(
        '''{ sc: group(by: ["state", "county"]) { length }
        cs: group(by: ["county", "state"]) { length } }'''
    )
    assert data['sc']['length'] == data['cs']['length'] == 3216


def test_partition(client):
    with pytest.raises(ValueError, match="is required"):
        client.execute('{ group { length } }')
    data = client.execute(
        '''{ partition(by: ["state"]) { aggregate { length columns { state { values } }
        column(name: "county") { ... on ListColumn { count { values } } } } } }'''
    )
    agg = data['partition']['aggregate']
    assert agg['length'] == 66
    assert agg['columns']['state']['values'][:3] == ['NY', 'PR', 'MA']
    assert agg['column']['count']['values'][:3] == [2, 176, 701]
    data = client.execute(
        '''{ sort(by: ["state", "longitude"]) {
        partition(by: ["state", "longitude"], diffs: [{name: "longitude", greater: 1.0}]) {
        length columns { state { values } }
        column(name: "longitude") { ... on ListColumn { count { values } } } } } }'''
    )
    groups = data['sort']['partition']
    assert groups['length'] == 62
    assert groups['columns']['state']['values'][:7] == ['AK'] * 7
    assert groups['column']['count']['values'][:7] == [1, 1, 5, 232, 1, 32, 1]
    data = client.execute(
        '''{ partition(by: ["state"], diffs: [{name: "state", less: null}]) {
        length column(name: "state") {
        ... on ListColumn {values { ... on StringColumn { values } } } } } }'''
    )
    assert data['partition']['length'] == 34
    assert data['partition']['column']['values'][0]['values'] == (['NY'] * 2) + (['PR'] * 176)
    data = client.execute(
        '''{ partition(by: ["state"]) { sort(by: ["state"]) {
        columns { state { values } } } } }'''
    )
    assert data['partition']['sort']['columns']['state']['values'][-2:] == ['WY', 'WY']
    data = client.execute(
        '''{ partition(by: ["state"]) { slice(offset: 2, length: 2) {
        aggregate(count: {name: "zipcode", alias: "c"}) {
        column(name: "c") { ... on LongColumn { values } } columns { state { values } } } } } }'''
    )
    agg = data['partition']['slice']['aggregate']
    assert agg['column']['values'] == [701, 91]
    assert agg['columns']['state']['values'] == ['MA', 'RI']
    data = client.execute(
        '''{ partition(by: ["state"]) { filter(query: {zipcode: {greater: 90000}}) {
        column(name: "zipcode") { ... on ListColumn { count { values } } } } } }'''
    )
    counts = data['partition']['filter']['column']['count']['values']
    assert len(counts) == 66
    assert counts.count(0) == 61
    data = client.execute(
        '''{ partition(by: ["state"], counts: "c") {
        filter(on: {string: {name: "state", equal: "NY"}}) {
        column(name: "c") { ... on LongColumn { values } } columns { state { values } } } } }'''
    )
    agg = data['partition']['filter']
    assert agg['column']['values'] == [2, 1, 2202]
    assert agg['columns']['state']['values'] == ['NY'] * 3


def test_rows(client):
    with pytest.raises(ValueError, match="out of bounds"):
        client.execute('{ row(index: 100000) { zipcode } }')
    data = client.execute('{ row { state } }')
    assert data == {'row': {'state': 'NY'}}
    data = client.execute('{ row(index: -1) { state } }')
    assert data == {'row': {'state': 'AK'}}
    data = client.execute('{ min(by: ["latitude"]) { row { state } } }')
    assert data == {'min': {'row': {'state': 'PR'}}}
    data = client.execute('{ max(by: ["latitude", "longitude"]) { row { state } } }')
    assert data == {'max': {'row': {'state': 'AK'}}}
