import sys
import pytest


def test_camel(aliasclient):
    data = aliasclient.execute('{ index schema { names } }')
    assert data == {'index': ['snakeId', 'camelId'], 'schema': {'names': ['snakeId', 'camelId']}}
    data = aliasclient.execute('{ row { snakeId } columns { snakeId { type } } }')
    assert data == {'row': {'snakeId': 1}, 'columns': {'snakeId': {'type': 'int64'}}}
    data = aliasclient.execute('{ filter(snakeId: {eq: 1}) { index length } }')
    assert data == {'filter': {'index': [], 'length': 1}}
    data = aliasclient.execute('{ filter(camelId: {eq: 1}) { index length } }')
    assert data == {'filter': {'index': [], 'length': 1}}


def test_snake(executor):
    data = executor('{ index schema { names } }')
    assert data['index'] == ['snake_id', 'camelId']
    assert 'snake_id' in data['schema']['names']
    data = executor('{ row { snake_id } columns { snake_id { type } } }')
    assert data == {'row': {'snake_id': 1}, 'columns': {'snake_id': {'type': 'int64'}}}
    data = executor('{ filter(snake_id: {eq: 1}) { index length } }')
    assert data == {'filter': {'index': ['camelId'], 'length': 1}}
    data = executor('{ filter(camelId: {eq: 1}) { index length } }')
    assert data == {'filter': {'index': [], 'length': 1}}


def test_columns(executor):
    def execute(query):
        return executor(f'{{ columns {query} }}')['columns']

    for name in ('uint8', 'int8', 'uint16', 'int16', 'int32'):
        assert execute(f'{{ {name} {{ values }} }}') == {name: {'values': [0, None]}}
        assert execute(f'{{ {name} {{ index(value: 0) }} }}') == {name: {'index': 0}}
        data = execute(f'{{ {name} {{ dropNull {{ length }} }} }}')
        assert data == {name: {'dropNull': {'length': 1}}}
        data = execute(f'{{ {name} {{ fillNull(value: 1) {{ values }} }} }}')
        assert data == {name: {'fillNull': {'values': [0, 1]}}}
        assert execute(f'{{ {name} {{ type }} }}') == {name: {'type': name}}
        assert execute(f'{{ {name} {{ min max }} }}')
    for name in ('uint32', 'uint64', 'int64'):
        assert execute(f'{{ {name} {{ values }} }}') == {name: {'values': [0, None]}}
        assert execute(f'{{ {name} {{ index(value: 0) }} }}') == {name: {'index': 0}}
        data = execute(f'{{ {name} {{ dropNull {{ length }} }} }}')
        assert data == {name: {'dropNull': {'length': 1}}}
        data = execute(f'{{ {name} {{ fillNull(value: 1) {{ values }} }} }}')
        assert data == {name: {'fillNull': {'values': [0, 1]}}}
        assert execute(f'{{ {name} {{ min max }} }}')

    for name in ('float', 'double'):
        assert execute(f'{{ {name} {{ values }} }}') == {name: {'values': [0.0, None]}}
        assert execute(f'{{ {name} {{ index(value: 0.0) }} }}') == {name: {'index': 0}}
        data = execute(f'{{ {name} {{ dropNull {{ length }} }} }}')
        assert data == {name: {'dropNull': {'length': 1}}}
        data = execute(f'{{ {name} {{ fillNull(value: 1.0) {{ values }} }} }}')
        assert data == {name: {'fillNull': {'values': [0.0, 1.0]}}}
        assert execute(f'{{ {name} {{ min max }} }}')
    assert execute('{ decimal { values } }') == {'decimal': {'values': ['0', None]}}
    assert execute('{ decimal { min max } }')

    for name in ('date32', 'date64'):
        assert execute(f'{{ {name} {{ values }} }}') == {name: {'values': ['1970-01-01', None]}}
        assert execute(f'{{ {name} {{ index(value: "1970-01-01") }} }}') == {name: {'index': 0}}
        data = execute(f'{{ {name} {{ dropNull {{ length }} }} }}')
        assert data == {name: {'dropNull': {'length': 1}}}
        data = execute(f'{{ {name} {{ fillNull(value: "1970-01-02") {{ values }} }} }}')
        assert data == {name: {'fillNull': {'values': ['1970-01-01', '1970-01-02']}}}
        assert execute(f'{{ {name} {{ min max }} }}')

    data = execute('{ timestamp { values } }')
    assert data == {'timestamp': {'values': ['1970-01-01T00:00:00', None]}}
    data = execute(f'{{ {name} {{ dropNull {{ length }} }} }}')
    assert data == {name: {'dropNull': {'length': 1}}}
    data = execute('{ timestamp { fillNull(value: "1970-01-02T00:00:00") { values } } }')
    assert data == {
        'timestamp': {'fillNull': {'values': ['1970-01-01T00:00:00', '1970-01-02T00:00:00']}}
    }
    assert execute('{ timestamp { index(value: "1970-01-01") } }') == {'timestamp': {'index': 0}}
    assert execute('{ timestamp { min max } }')
    data = execute(
        '''{ timestamp { floorTemporal(unit: "day") { values }
        roundTemporal(unit: "day") { values } ceilTemporal(unit: "day") { values } } }'''
    )
    for name in ('floorTemporal', 'roundTemporal', 'ceilTemporal'):
        data['timestamp'][name] == {'values': ['1970-01-01T00:00:00', None]}

    for name in ('time32', 'time64'):
        assert execute(f'{{ {name} {{ values }} }}') == {name: {'values': ['00:00:00', None]}}
        assert execute(f'{{ {name} {{ index(value: "00:00:00") }} }}') == {name: {'index': 0}}
        data = execute(f'{{ {name} {{ dropNull {{ length }} }} }}')
        assert data == {name: {'dropNull': {'length': 1}}}
        data = execute(f'{{ {name} {{ fillNull(value: "00:00:01") {{ values }} }} }}')
        assert data == {name: {'fillNull': {'values': ['00:00:00', '00:00:01']}}}
        assert execute(f'{{ {name} {{ min max }} }}')

    for name in ('binary', 'string'):
        assert execute(f'{{ {name} {{ values }} }}') == {name: {'values': ['', None]}}
        assert execute(f'{{ {name} {{ index(value: "") }} }}') == {name: {'index': 0}}
        data = execute(f'{{ {name} {{ dropNull {{ length }} }} }}')
        assert data == {name: {'dropNull': {'length': 1}}}
        data = execute(f'{{ {name} {{ fillNull(value: "") {{ values }} }} }}')
        assert data == {name: {'fillNull': {'values': ['', '']}}}

    data = execute(
        '{ binary { binaryReplaceSlice(start: 0, stop: 1, replacement: "") { values } } }'
    )
    assert data == {'binary': {'binaryReplaceSlice': {'values': ['', None]}}}
    assert execute('{ string { type } }') == {
        'string': {'type': 'dictionary<values=string, indices=int32, ordered=0>'}
    }
    assert execute('{ string { min max } }')


def test_boolean(executor):
    def execute(query):
        return executor(f'{{ columns {query} }}')['columns']

    assert execute('{ bool { values } }') == {'bool': {'values': [False, None]}}
    assert execute('{ bool { index(value: false) } }') == {'bool': {'index': 0}}
    assert execute('{ bool { index(value: false, start: 1, end: 2) } }') == {'bool': {'index': -1}}
    assert execute('{ bool { type } }') == {'bool': {'type': 'bool'}}
    assert execute('{ bool { unique { length } } }') == {'bool': {'unique': {'length': 2}}}
    assert execute('{ bool { any all } }') == {'bool': {'any': False, 'all': False}}

    data = executor(
        '{ apply(boolean: {name: "bool", xor: "bool"}) { columns { bool { values } } } }'
    )
    assert data == {'apply': {'columns': {'bool': {'values': [False, None]}}}}
    data = executor(
        '''{ apply(boolean: {name: "bool", andNot: "bool", kleene: true})
        { columns { bool { values } } } }'''
    )
    assert data == {'apply': {'columns': {'bool': {'values': [False, None]}}}}


def test_numeric(executor):
    for name in ('int32', 'int64', 'float'):
        data = executor(f'{{ columns {{ {name} {{ add(value: 1) {{ sum product }} }} }} }}')
        assert data == {'columns': {name: {'add': {'sum': 1, 'product': 1}}}}
        data = executor(f'{{ columns {{ {name} {{ subtract(value: 1) {{ sum }} }} }} }}')
        assert data == {'columns': {name: {'subtract': {'sum': 1}}}}
        data = executor(f'{{ columns {{ {name} {{ multiply(value: 1) {{ sum }} }} }} }}')
        assert data == {'columns': {name: {'multiply': {'sum': 0}}}}
        with pytest.raises(ValueError):
            executor(f'{{ columns {{ {name} {{ divide(value: 1) {{ sum }} }} }} }}')
        data = executor(f'{{ columns {{ {name} {{ power(base: 2) {{ values }} }} }} }}')
        assert data == {'columns': {name: {'power': {'values': [1, None]}}}}
        data = executor(f'{{ columns {{ {name} {{ power(exponent: 2) {{ values }} }} }} }}')
        assert data == {'columns': {name: {'power': {'values': [0, None]}}}}
        with pytest.raises(ValueError):
            executor(f'{{ columns {{ {name} {{ power {{ values }} }} }} }}')
        with pytest.raises(ValueError):
            executor(f'{{ columns {{ {name} {{ power(base: 1, exponent: 1) {{ values }} }} }} }}')

        data = executor(f'{{ columns {{ {name} {{ minElementWise(value: -1) {{ sum }} }} }} }}')
        assert data == {'columns': {name: {'minElementWise': {'sum': -2}}}}
        data = executor(f'{{ columns {{ {name} {{ maxElementWise(value: 1) {{ sum }} }} }} }}')
        assert data == {'columns': {name: {'maxElementWise': {'sum': 2}}}}
        data = executor(f'{{ columns {{ {name} {{ mean stddev variance }} }} }}')
        assert data == {'columns': {name: {'mean': 0.0, 'stddev': 0.0, 'variance': 0.0}}}
        data = executor(f'{{ columns {{ {name} {{ mode {{ values }} }} }} }}')
        assert data == {'columns': {name: {'mode': {'values': [0]}}}}
        data = executor(f'{{ columns {{ {name} {{ mode(length: 2) {{ counts }} }} }} }}')
        assert data == {'columns': {name: {'mode': {'counts': [1]}}}}
        data = executor(f'{{ columns {{ {name} {{ quantile }} }} }}')
        assert data == {'columns': {name: {'quantile': [0.0]}}}
        data = executor(f'{{ columns {{ {name} {{ tdigest }} }} }}')
        assert data == {'columns': {name: {'tdigest': [0.0]}}}

    data = executor(
        '''{ apply(int: [{name: "int32", fillNull: -1, alias: "i"}])
        { column(name: "i") { type ... on IntColumn { values } } } }'''
    )
    assert data == {'apply': {'column': {'type': 'int32', 'values': [0, -1]}}}
    data = executor('{ columns { float { add(value: 2.0) { divide(value: 1.0) { sum } } } } }')
    assert data == {'columns': {'float': {'add': {'divide': {'sum': 0.5}}}}}
    data = executor(
        '''{ column(name: "float", apply: {minElementWise: "int32", maxElementWise: "int32",
        power: "int32"}) { ... on FloatColumn { values } } }'''
    )
    assert data == {'column': {'values': [1.0, None]}}
    data = executor('{ column(name: "float", cast: "int32") { type } }')
    assert data == {'column': {'type': 'int32'}}
    data = executor(
        '''{ apply(int: {name: "int32", negate: true, checked: true})
        { columns { int32 { values } } } }'''
    )
    assert data == {'apply': {'columns': {'int32': {'values': [0, None]}}}}
    data = executor(
        '''{ apply(float: {name: "float", coalesce: "int32"})
        { columns { float { values } } } }'''
    )
    assert data == {'apply': {'columns': {'float': {'values': [0.0, None]}}}}
    data = executor('{ column(name: "float", apply: {coalesce: "int32"}) { type } }')
    assert data == {'column': {'type': 'float'}}
    data = executor(
        '{ apply(int: {name: "int32", bitWiseNot: true}) { columns { int32 { values } } } }'
    )
    assert data == {'apply': {'columns': {'int32': {'values': [-1, None]}}}}
    data = executor(
        '{ apply(int: {name: "int32", bitWiseOr: "int64"}) { columns { int32 { values } } } }'
    )
    assert data == {'apply': {'columns': {'int32': {'values': [0, None]}}}}
    data = executor(
        '{ column(name: "int32", apply: {atan2: "float"}) { ... on FloatColumn { values } } }'
    )
    assert data == {'column': {'values': [0.0, None]}}


@pytest.mark.skipif(sys.version_info < (3, 9), reason="requires python3.9 or pytz")
def test_datetime(executor):
    for name in ('timestamp', 'date32'):
        data = executor(
            f'''{{ apply(datetime: {{name: "{name}", year: true, alias: "year"}})
            {{ column(name: "year") {{ ... on LongColumn {{ values }} }} }} }}'''
        )
        assert data == {'apply': {'column': {'values': [1970, None]}}}
        data = executor(
            f'''{{ apply(datetime: {{name: "{name}", quarter: true, alias: "quarter"}})
            {{ column(name: "quarter") {{ ... on LongColumn {{ values }} }} }} }}'''
        )
        assert data == {'apply': {'column': {'values': [1, None]}}}
        data = executor(
            f'''{{ column(name: "{name}", apply: {{yearsBetween: "{name}"}})
            {{ ... on LongColumn {{ values }} }} }}'''
        )
        assert data == {'column': {'values': [0, None]}}
        data = executor(
            f'''{{ columns {{ {name}
            {{ between(unit: "years", start: "1969-01-01") {{ values }} }} }} }}'''
        )
        assert data == {'columns': {name: {'between': {'values': [1, None]}}}}
    data = executor('{ columns { date32 { strftime { strptime { values } } } } }')
    dates = data['columns']['date32']['strftime']['strptime']['values']
    assert dates == ['1970-01-01T00:00:00', None]
    for name in ('timestamp', 'time32'):
        data = executor(
            f'''{{ apply(datetime: {{name: "{name}", hour: true, alias: "hour"}})
            {{ column(name: "hour") {{ ... on LongColumn {{ values }} }} }} }}'''
        )
        assert data == {'apply': {'column': {'values': [0, None]}}}
        data = executor(
            f'''{{ apply(datetime: {{name: "{name}", subsecond: true, alias: "subsecond"}})
            {{ column(name: "subsecond") {{ ... on FloatColumn {{ values }} }} }} }}'''
        )
        assert data == {'apply': {'column': {'values': [0.0, None]}}}
        data = executor(
            f'''{{ column(name: "{name}", apply: {{hoursBetween: "{name}"}})
            {{ ... on LongColumn {{ values }} }} }}'''
        )
        assert data == {'column': {'values': [0, None]}}
    data = executor('{ columns { time32 { between(unit: "hours", end: "01:00:00") { values } } } }')
    assert data == {'columns': {'time32': {'between': {'values': [1, None]}}}}
    with pytest.raises(ValueError):
        executor('{ columns { time64 { between(unit: "hours") { values } } } }')
    data = executor(
        '''{ apply(datetime: {name: "timestamp", fillNullForward: true}) { columns
        { timestamp { assumeTimezone(timezone: "UTC") { values } } } } }'''
    )
    dates = data['apply']['columns']['timestamp']['assumeTimezone']['values']
    assert dates == ['1970-01-01T00:00:00+00:00', '1970-01-01T00:00:00+00:00']


def test_duration(executor):
    data = executor(
        '''{ apply(datetime: [{name: "timestamp", fillNull: "0001-01-01"}])
        { columns { timestamp { values subtract(value: "0001-01-01") { values } } } } }'''
    )
    column = data['apply']['columns']['timestamp']
    assert column['values'] == ['1970-01-01T00:00:00', '0001-01-01T00:00:00']
    assert column['subtract'] == {'values': [-62135596800.0, 0.0]}
    data = executor(
        '''{ scan(columns: {alias: "diff", sub: [{name: "timestamp"}, {name: "timestamp"}]})
        { column(name: "diff") { ... on DurationColumn { values } } } }'''
    )
    assert data == {'scan': {'column': {'values': [0.0, None]}}}
    data = executor(
        '''{ partition(by: ["timestamp"] diffs: [{name: "timestamp", gt: 0.0}]) { length } }'''
    )
    assert data == {'partition': {'length': 1}}


def test_list(executor):
    data = executor('{ columns { list { length type } } }')
    assert data == {'columns': {'list': {'length': 2, 'type': 'list<item: int32>'}}}
    data = executor('{ columns { list { values { length } } } }')
    assert data == {'columns': {'list': {'values': [{'length': 3}, None]}}}
    data = executor('{ row { list { ... on IntColumn { values } } } }')
    assert data == {'row': {'list': {'values': [0, 1, 2]}}}
    data = executor('{ row(index: -1) { list { ... on IntColumn { values } } } }')
    assert data == {'row': {'list': None}}
    data = executor('{ columns { list { mode { flatten { ... on IntColumn { values } } } } } }')
    assert data == {'columns': {'list': {'mode': {'flatten': {'values': [0]}}}}}
    data = executor(
        '''{ columns { list { mode(n: 2) { flatten { ... on IntColumn { values } } } } } }'''
    )
    assert data == {'columns': {'list': {'mode': {'flatten': {'values': [0, 1]}}}}}
    data = executor(
        '''{ columns { list { quantile(q: [0.25, 0.75]) { flatten { ... on FloatColumn
        { values } } } } } }'''
    )
    column = data['columns']['list']
    assert column == {'quantile': {'flatten': {'values': [0.5, 1.5, None, None]}}}
    data = executor(
        '''{ columns { list { tdigest(q: [0.25, 0.75]) { flatten { ... on FloatColumn
        { values } } } } } }'''
    )
    column = data['columns']['list']
    assert column == {'tdigest': {'flatten': {'values': [0.0, 2.0, None, None]}}}

    data = executor(
        '''{ columns { list { count { values } countDistinct { values } valueLength { values }
        first: element { ... on IntColumn { values } }
        last: element(index: -1) { ... on IntColumn { values } }
        min { ... on IntColumn { values } } max { ... on IntColumn { values } }
        sum { ... on LongColumn { values } } product { ... on LongColumn { values } } mean { values }
        stddev { values } variance { values } } } }'''
    )
    assert data['columns']['list'] == {
        'count': {'values': [3, None]},
        'countDistinct': {'values': [3, None]},
        'valueLength': {'values': [3, None]},
        'first': {'values': [0, None]},
        'last': {'values': [2, None]},
        'min': {'values': [0, None]},
        'max': {'values': [2, None]},
        'sum': {'values': [3, None]},
        'product': {'values': [0, None]},
        'mean': {'values': [1.0, None]},
        'stddev': {'values': [pytest.approx((2 / 3) ** 0.5), None]},
        'variance': {'values': [pytest.approx(2 / 3), None]},
    }
    data = executor('{ columns { list { distinct { valueLength { values } } } } }')
    assert data['columns']['list'] == {'distinct': {'valueLength': {'values': [3, None]}}}
    data = executor(
        '''{ aggregate(distinct: {name: "list", mode: "only_null"})
        { columns { list { flatten { length } } } } }'''
    )
    assert data['aggregate']['columns']['list'] == {'flatten': {'length': 0}}
    data = executor(
        '''{ apply(list: {filter: {ne: [{name: "list"}, {int: 1}]}}) {
        columns { list { values { ... on IntColumn { values } } } } } }'''
    )
    column = data['apply']['columns']['list']
    assert column == {'values': [{'values': [0, 2]}, {'values': []}]}
    data = executor(
        '''{ apply(list: {name: "list", mode: true}) {
        column(name: "list") { ... on IntColumn { values } } } }'''
    )
    assert data['apply']['column']['values'] == [0, None]
    data = executor(
        '''{ aggregate(stddev: {name: "list"}, variance: {name: "list", alias: "var"}) {
        column(name: "list") { ... on FloatColumn { values } }
        var: column(name: "var") { ... on FloatColumn { values } } } }'''
    )
    assert data['aggregate']['column']['values'] == [pytest.approx((2 / 3) ** 0.5), None]
    assert data['aggregate']['var']['values'] == [pytest.approx((2 / 3)), None]
    data = executor(
        '''{ partition(by: "int32") { column(name: "binary") { ... on ListColumn
        { binaryJoin(separator: " ") { ... on Base64Column { values } } } } } }'''
    )
    assert data == {'partition': {'column': {'binaryJoin': {'values': [None]}}}}


def test_struct(executor):
    data = executor('{ columns { struct { names column(name: "x") { length } } } }')
    assert data == {'columns': {'struct': {'names': ['x', 'y'], 'column': {'length': 2}}}}


def test_dictionary(executor):
    data = executor('{ column(name: "string") { length } }')
    assert data == {'column': {'length': 2}}
    data = executor(
        '''{ group(by: ["camelId"]) { column(name: "string") {
        ... on ListColumn { distinct { count { values } } } } } }'''
    )
    assert data == {'group': {'column': {'distinct': {'count': {'values': [1, 0]}}}}}
    data = executor(
        '''{ group(by: ["string"]) { tables {
        columns { string { values } } column(name: "camelId") { length } } } }'''
    )
    assert data['group']['tables'] == [
        {'columns': {'string': {'values': ['']}}, 'column': {'length': 1}},
        {'columns': {'string': {'values': [None]}}, 'column': {'length': 1}},
    ]
    data = executor(
        '''{ group(by: ["camelId"]) { aggregate(countDistinct: {name: "string"}) {
        column(name: "string") { ... on LongColumn { values } } } } }'''
    )
    assert data == {'group': {'aggregate': {'column': {'values': [1, 0]}}}}
    data = executor(
        '{ apply(string: [{name: "string", fillNull: ""}]) { columns { string { values } } } }'
    )
    assert data == {'apply': {'columns': {'string': {'values': ['', '']}}}}


def test_selections(executor):
    data = executor('{ slice { length } slice { sort(by: "snake_id") { length } } }')
    assert data == {'slice': {'length': 2, 'sort': {'length': 2}}}


def test_conditions(executor):
    data = executor('{ column(name: "bool", apply: {ifElse: ["int32", "float"]}) { type } }')
    assert data == {'column': {'type': 'float'}}
    with pytest.raises(ValueError, match="must be BOOL"):
        executor('{ column(name: "struct", apply: {caseWhen: ["int32", "float"]}) { type } }')


def test_long(executor):
    with pytest.raises(ValueError, match="Long cannot represent value"):
        executor('{ filter(int64: {eq: 0.0}) { length } }')
