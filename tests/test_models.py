import pytest


def test_camel(aliasclient):
    data = aliasclient.execute('{ schema { index names } }')
    assert data == {'schema': {'index': [], 'names': ['snakeId', 'camelId']}}
    data = aliasclient.execute('{ row { snakeId } columns { snakeId { type } } }')
    assert data == {'row': {'snakeId': 1}, 'columns': {'snakeId': {'type': 'int64'}}}
    data = aliasclient.execute('{ filter(snakeId: {eq: 1}) { length } }')
    assert data == {'filter': {'length': 1}}
    data = aliasclient.execute('{ filter(camelId: {eq: 1}) { length } }')
    assert data == {'filter': {'length': 1}}


def test_snake(executor):
    data = executor('{ schema { names } }')
    assert 'snake_id' in data['schema']['names']
    data = executor('{ row { snake_id } columns { snake_id { type } } }')
    assert data == {'row': {'snake_id': 1}, 'columns': {'snake_id': {'type': 'int64'}}}
    data = executor('{ filter(snake_id: {eq: 1}) { length } }')
    assert data == {'filter': {'length': 1}}
    data = executor('{ filter(camelId: {eq: 1}) { length } }')
    assert data == {'filter': {'length': 1}}


def test_columns(executor):
    def execute(query):
        return executor(f'{{ columns {query} }}')['columns']

    for name in ('uint8', 'int8', 'uint16', 'int16', 'int32'):
        assert execute(f'{{ {name} {{ values }} }}') == {name: {'values': [0, None]}}
        assert execute(f'{{ {name} {{ index(value: 0) }} }}') == {name: {'index': 0}}
        data = execute(f'{{ {name} {{ dropNull }} }}')
        assert data == {name: {'dropNull': [0]}}
        assert execute(f'{{ {name} {{ type }} }}') == {name: {'type': name}}
        assert execute(f'{{ {name} {{ min max }} }}')
    for name in ('uint32', 'uint64', 'int64'):
        assert execute(f'{{ {name} {{ values }} }}') == {name: {'values': [0, None]}}
        assert execute(f'{{ {name} {{ index(value: 0) }} }}') == {name: {'index': 0}}
        data = execute(f'{{ {name} {{ dropNull }} }}')
        assert data == {name: {'dropNull': [0]}}
        assert execute(f'{{ {name} {{ min max }} }}')

    for name in ('float', 'double'):
        assert execute(f'{{ {name} {{ values }} }}') == {name: {'values': [0.0, None]}}
        assert execute(f'{{ {name} {{ index(value: 0.0) }} }}') == {name: {'index': 0}}
        data = execute(f'{{ {name} {{ dropNull }} }}')
        assert data == {name: {'dropNull': [0.0]}}
        assert execute(f'{{ {name} {{ min max }} }}')

    for name in ('date32', 'date64'):
        assert execute(f'{{ {name} {{ values }} }}') == {name: {'values': ['1970-01-01', None]}}
        assert execute(f'{{ {name} {{ index(value: "1970-01-01") }} }}') == {name: {'index': 0}}
        data = execute(f'{{ {name} {{ dropNull }} }}')
        assert data == {name: {'dropNull': ['1970-01-01']}}
        assert execute(f'{{ {name} {{ min max }} }}')
        assert execute(f'{{ {name} {{ first last }} }}')

    data = execute('{ timestamp { values } }')
    assert data == {'timestamp': {'values': ['1970-01-01T00:00:00', None]}}
    data = execute(f'{{ {name} {{ dropNull }} }}')
    assert data == {name: {'dropNull': ['1970-01-01']}}
    assert execute('{ timestamp { index(value: "1970-01-01") } }') == {'timestamp': {'index': 0}}
    assert execute('{ timestamp { min max } }')

    for name in ('time32', 'time64'):
        assert execute(f'{{ {name} {{ values }} }}') == {name: {'values': ['00:00:00', None]}}
        assert execute(f'{{ {name} {{ index(value: "00:00:00") }} }}') == {name: {'index': 0}}
        data = execute(f'{{ {name} {{ dropNull }} }}')
        assert data == {name: {'dropNull': ['00:00:00']}}
        assert execute(f'{{ {name} {{ min max }} }}')

    for name in ('binary', 'string'):
        assert execute(f'{{ {name} {{ values }} }}') == {name: {'values': ['', None]}}
        assert execute(f'{{ {name} {{ index(value: "") }} }}') == {name: {'index': 0}}
        data = execute(f'{{ {name} {{ dropNull }} }}')
        assert data == {name: {'dropNull': ['']}}
        data = execute(f'{{ {name} {{ fillNull(value: "") }} }}')
        assert data == {name: {'fillNull': ['', '']}}

    assert execute('{ string { type } }') == {
        'string': {'type': 'dictionary<values=string, indices=int32, ordered=0>'}
    }
    assert execute('{ string { min max } }')
    assert execute('{ string { first last } }') == {'string': {'first': '', 'last': ''}}


def test_boolean(executor):
    def execute(query):
        return executor(f'{{ columns {query} }}')['columns']

    assert execute('{ bool { values } }') == {'bool': {'values': [False, None]}}
    assert execute('{ bool { index(value: false) } }') == {'bool': {'index': 0}}
    assert execute('{ bool { index(value: false, start: 1, end: 2) } }') == {'bool': {'index': -1}}
    assert execute('{ bool { type } }') == {'bool': {'type': 'bool'}}
    assert execute('{ bool { unique { length } } }') == {'bool': {'unique': {'length': 2}}}
    assert execute('{ bool { any all } }') == {'bool': {'any': False, 'all': False}}
    assert execute('{ bool { indicesNonzero } }') == {'bool': {'indicesNonzero': []}}

    data = executor('{ scan(filter: {xor: [{name: "bool"}, {inv: {name: "bool"}}]}) { length } }')
    assert data == {'scan': {'length': 1}}
    data = executor(
        """{ scan(columns: {alias: "bool", andNot: [{inv: {name: "bool"}}, {name: "bool"}], kleene: true})
        { columns { bool { values } } } }"""
    )
    assert data == {'scan': {'columns': {'bool': {'values': [True, None]}}}}


def test_decimal(executor):
    def execute(query):
        return executor(f'{{ columns {query} }}')['columns']

    assert execute('{ decimal { values } }') == {'decimal': {'values': ['0', None]}}
    assert execute('{ decimal { min max } }')
    assert execute('{ decimal { indicesNonzero } }') == {'decimal': {'indicesNonzero': []}}
    assert execute('{ decimal { index(value: 0) } }')
    data = executor(
        '{ sort(by: "decimal", nullPlacement: "at_start") { columns { decimal { values } } } }'
    )
    assert data == {'sort': {'columns': {'decimal': {'values': [None, '0']}}}}
    data = executor('{ rank(by: "decimal") { columns { decimal { values } } } }')
    assert data == {'rank': {'columns': {'decimal': {'values': ['0']}}}}
    data = executor(
        '{ rank(by: "-decimal", nullPlacement: "at_start") { columns { decimal { values } } } }'
    )
    assert data == {'rank': {'columns': {'decimal': {'values': [None]}}}}


def test_numeric(executor):
    for name in ('int32', 'int64', 'float'):
        data = executor(f'{{ columns {{ {name} {{ mean stddev variance }} }} }}')
        assert data == {'columns': {name: {'mean': 0.0, 'stddev': 0.0, 'variance': 0.0}}}
        data = executor(f'{{ columns {{ {name} {{ mode {{ values }} }} }} }}')
        assert data == {'columns': {name: {'mode': {'values': [0]}}}}
        data = executor(f'{{ columns {{ {name} {{ mode(n: 2) {{ counts }} }} }} }}')
        assert data == {'columns': {name: {'mode': {'counts': [1]}}}}
        data = executor(f'{{ columns {{ {name} {{ quantile }} }} }}')
        assert data == {'columns': {name: {'quantile': [0.0]}}}
        data = executor(f'{{ columns {{ {name} {{ tdigest }} }} }}')
        assert data == {'columns': {name: {'tdigest': [0.0]}}}
        data = executor(f'{{ columns {{ {name} {{ product }} }} }}')
        assert data == {'columns': {name: {'product': 0.0}}}
        data = executor(f'{{ columns {{ {name} {{ indicesNonzero }} }} }}')
        assert data == {'columns': {name: {'indicesNonzero': []}}}

    data = executor("""{ scan(columns: {alias: "int32", elementWise: {min: {name: "int32"}}}) {
        columns { int32 { values } } } }""")
    assert data == {'scan': {'columns': {'int32': {'values': [0, None]}}}}
    data = executor('{ column(name: "float", cast: "int32") { type } }')
    assert data == {'column': {'type': 'int32'}}
    data = executor("""{ scan(columns: {alias: "int32", negate: {checked: true, name: "int32"}}) {
        columns { int32 { values } } } }""")
    assert data == {'scan': {'columns': {'int32': {'values': [0, None]}}}}
    data = executor(
        """{ scan(columns: {alias: "float", coalesce: [{name: "float"}, {name: "int32"}]}) {
        columns { float { values } } } }"""
    )
    assert data == {'scan': {'columns': {'float': {'values': [0.0, None]}}}}
    data = executor("""{ scan(columns: {bitWise: {not: {name: "int32"}}, alias: "int32"}) {
        columns { int32 { values } } } }""")
    assert data == {'scan': {'columns': {'int32': {'values': [-1, None]}}}}
    data = executor(
        """{ scan(columns: {bitWise: {or: [{name: "int32"}, {name: "int64"}]}, alias: "int64"}) {
        columns { int64 { values } } } }"""
    )
    assert data == {'scan': {'columns': {'int64': {'values': [0, None]}}}}


def test_datetime(executor):
    for name in ('timestamp', 'date32'):
        data = executor(
            f"""{{ scan(columns: {{alias: "year", temporal: {{year: {{name: "{name}"}}}}}})
            {{ column(name: "year") {{ ... on LongColumn {{ values }} }} }} }}"""
        )
        assert data == {'scan': {'column': {'values': [1970, None]}}}
        data = executor(
            f"""{{ scan(columns: {{alias: "quarter", temporal: {{quarter: {{name: "{name}"}}}}}})
            {{ column(name: "quarter") {{ ... on LongColumn {{ values }} }} }} }}"""
        )
        assert data == {'scan': {'column': {'values': [1, None]}}}
        data = executor(f"""{{ scan(columns: {{alias: "{name}",
            temporal: {{yearsBetween: [{{name: "{name}"}}, {{name: "{name}"}}]}}}})
            {{ column(name: "{name}") {{ ... on LongColumn {{ values }} }} }} }}""")
        assert data == {'scan': {'column': {'values': [0, None]}}}
    data = executor(
        """{ scan(columns: {alias: "timestamp", temporal: {strftime: {name: "timestamp"}}}) {
        column(name: "timestamp") { type } } }"""
    )
    assert data == {'scan': {'column': {'type': 'string'}}}
    for name in ('timestamp', 'time32'):
        data = executor(
            f"""{{ scan(columns: {{alias: "hour", temporal: {{hour: {{name: "{name}"}}}}}})
            {{ column(name: "hour") {{ ... on LongColumn {{ values }} }} }} }}"""
        )
        assert data == {'scan': {'column': {'values': [0, None]}}}
        data = executor(
            f"""{{ scan(columns: {{alias: "subsecond", temporal: {{subsecond: {{name: "{name}"}}}}}})
            {{ column(name: "subsecond") {{ ... on FloatColumn {{ values }} }} }} }}"""
        )
        assert data == {'scan': {'column': {'values': [0.0, None]}}}
        data = executor(f"""{{ scan(columns: {{alias: "hours",
            temporal: {{hoursBetween: [{{name: "{name}"}}, {{name: "{name}"}}]}}}})
            {{ column(name: "hours") {{ ... on LongColumn {{ values }} }} }} }}""")
        assert data == {'scan': {'column': {'values': [0, None]}}}
    with pytest.raises(ValueError):
        executor('{ columns { time64 { between(unit: "hours") { values } } } }')
    data = executor(
        """{ scan(columns: {alias: "timestamp", temporal: {assumeTimezone: {name: "timestamp"}, timezone: "UTC"}}) {
        columns { timestamp { values } } } }"""
    )
    dates = data['scan']['columns']['timestamp']['values']
    assert dates == ['1970-01-01T00:00:00+00:00', None]
    data = executor(
        """{ scan(columns: {alias: "time32", temporal: {round: {name: "time32"}, unit: "hour"}}) {
        columns { time32 { values } } } }"""
    )
    assert data == {'scan': {'columns': {'time32': {'values': ['00:00:00', None]}}}}


def test_duration(executor):
    data = executor(
        """{ scan(columns: {alias: "diff", checked: true, subtract: [{name: "timestamp"}, {name: "timestamp"}]})
        { column(name: "diff") { ... on DurationColumn { unique { values } } } } }"""
    )
    assert data == {'scan': {'column': {'unique': {'values': ['PT0S', None]}}}}
    data = executor('{ runs(split: [{name: "timestamp", gt: 0.0}]) { length } }')
    assert data == {'runs': {'length': 1}}
    data = executor("""{ scan(columns: {alias: "diff", temporal:
        {monthDayNanoIntervalBetween: [{name: "timestamp"}, {name: "timestamp"}]}})
        { column(name: "diff") { ... on DurationColumn { values } } } }""")
    assert data == {'scan': {'column': {'values': ['P0MT0S', None]}}}


def test_list(executor):
    data = executor('{ columns { list { length type } } }')
    assert data == {'columns': {'list': {'length': 2, 'type': 'list<item: int32>'}}}
    data = executor('{ columns { list { values { length } } } }')
    assert data == {'columns': {'list': {'values': [{'length': 3}, None]}}}
    data = executor('{ columns { list { dropNull { length } } } }')
    assert data == {'columns': {'list': {'dropNull': [{'length': 3}]}}}
    data = executor('{ row { list { ... on IntColumn { values } } } }')
    assert data == {'row': {'list': {'values': [0, 1, 2]}}}
    data = executor('{ row(index: -1) { list { ... on IntColumn { values } } } }')
    assert data == {'row': {'list': None}}
    data = executor("""{ aggregate(approximateMedian: {name: "list"}) {
        column(name: "list") { ... on FloatColumn { values } } } }""")
    assert data == {'aggregate': {'column': {'values': [1.0, None]}}}
    data = executor("""{ aggregate(tdigest: {name: "list", q: [0.25, 0.75]}) {
        columns { list { flatten { ... on FloatColumn { values } } } } } }""")
    column = data['aggregate']['columns']['list']
    assert column == {'flatten': {'values': [0.0, 2.0]}}

    data = executor("""{ apply(list: {quantile: {name: "list", q: 0.5}}) {
        columns { list { flatten { ... on FloatColumn { values } } } } } }""")
    assert data == {'apply': {'columns': {'list': {'flatten': {'values': [1.0, None]}}}}}
    data = executor("""{ apply(list: {index: {name: "list", value: 1}}) {
        column(name: "list") { ... on LongColumn { values } } } }""")
    assert data == {'apply': {'column': {'values': [1, -1]}}}
    data = executor(
        """{ scan(columns: {list: {element: [{name: "list"}, {value: 1}]}, alias: "value"}) {
        column(name: "value") { ... on IntColumn { values } } } }"""
    )
    assert data == {'scan': {'column': {'values': [1, None]}}}
    data = executor("""{ scan(columns: {list: {slice: {name: "list"}, stop: 1}, alias: "value"}) {
        column(name: "value") { ... on ListColumn { flatten { ... on IntColumn { values } } } } } }""")
    assert data == {'scan': {'column': {'flatten': {'values': [0]}}}}
    data = executor("""{ aggregate(distinct: {name: "list", mode: "only_null"})
        { columns { list { flatten { length } } } } }""")
    assert data['aggregate']['columns']['list'] == {'flatten': {'length': 0}}
    data = executor("""{ apply(list: {filter: {ne: [{name: "list"}, {value: 1}]}}) {
        columns { list { values { ... on IntColumn { values } } } } } }""")
    column = data['apply']['columns']['list']
    assert column == {'values': [{'values': [0, 2]}, None]}
    data = executor('{ apply(list: {mode: {name: "list"}}) { column(name: "list") { type } } }')
    assert data['apply']['column']['type'] == 'large_list<item: struct<mode: int32, count: int64>>'
    data = executor(
        """{ aggregate(stddev: {name: "list"}, variance: {name: "list", alias: "var", ddof: 1}) {
        column(name: "list") { ... on FloatColumn { values } }
        var: column(name: "var") { ... on FloatColumn { values } } } }"""
    )
    assert data['aggregate']['column']['values'] == [pytest.approx((2 / 3) ** 0.5), None]
    assert data['aggregate']['var']['values'] == [1, None]
    data = executor(
        """{ runs(by: "int32") { scan(columns: {binary: {join: [{name: "binary"}, {base64: ""}]}, alias: "binary"}) {
        column(name: "binary") { ... on Base64Column { values } } } } }"""
    )
    assert data == {'runs': {'scan': {'column': {'values': [None]}}}}
    data = executor('{ columns { list { value { type } } } }')
    assert data == {'columns': {'list': {'value': {'type': 'int32'}}}}
    data = executor('{ tables { column(name: "list") { type } } }')
    assert data == {'tables': [{'column': {'type': 'int32'}}, None]}
    data = executor(
        '{ apply(list: {rank: {by: "list"}}) { columns { list { values { length } } } } }'
    )
    assert data == {'apply': {'columns': {'list': {'values': [{'length': 1}, None]}}}}
    data = executor(
        '{ apply(list: {rank: {by: "list", max: 2}}) { columns { list { flatten { ... on IntColumn { values } } } } } }'
    )
    assert data == {'apply': {'columns': {'list': {'flatten': {'values': [0, 1]}}}}}


def test_struct(executor):
    data = executor('{ columns { struct { names column(name: "x") { length } } } }')
    assert data == {'columns': {'struct': {'names': ['x', 'y'], 'column': {'length': 2}}}}
    data = executor("""{ scan(columns: {alias: "leaf", name: ["struct", "x"]}) {
        column(name: "leaf") { ... on IntColumn { values } } } }""")
    assert data == {'scan': {'column': {'values': [0, None]}}}
    data = executor('{ column(name: ["struct", "x"]) { type } }')
    assert data == {'column': {'type': 'int32'}}
    data = executor('{ row { struct } columns { struct { value } } }')
    assert data['row']['struct'] == data['columns']['struct']['value'] == {'x': 0, 'y': None}
    with pytest.raises(ValueError, match="must be BOOL"):
        executor(
            '{ scan(filter: {caseWhen: [{name: "struct"}, {name: "int32"}, {name: "float"}]}) {type } }'
        )


def test_dictionary(executor):
    data = executor('{ column(name: "string") { length } }')
    assert data == {'column': {'length': 2}}
    data = executor("""{ group(by: ["string"], aggregate: {list: {name: "camelId"}}) { tables {
        columns { string { values } } column(name: "camelId") { length } } } }""")
    assert data['group']['tables'] == [
        {'columns': {'string': {'values': ['']}}, 'column': {'length': 1}},
        {'columns': {'string': {'values': [None]}}, 'column': {'length': 1}},
    ]
    data = executor("""{ group(by: ["camelId"], aggregate: {countDistinct: {name: "string"}}) {
        column(name: "string") { ... on LongColumn { values } } } }""")
    assert data == {'group': {'column': {'values': [1, 0]}}}
    data = executor(
        """{ scan(columns: {alias: "string", coalesce: [{name: "string"}, {value: ""}]}) {
        columns { string { values } } } }"""
    )
    assert data == {'scan': {'columns': {'string': {'values': ['', '']}}}}


def test_selections(executor):
    data = executor('{ slice { length } slice { sort(by: "snake_id") { length } } }')
    assert data == {'slice': {'length': 2, 'sort': {'length': 2}}}
    data = executor('{ dropNull { length } }')
    assert data == {'dropNull': {'length': 2}}
    data = executor('{ dropNull { columns { float { values } } } }')
    assert data == {'dropNull': {'columns': {'float': {'values': [0.0]}}}}


def test_conditions(executor):
    data = executor(
        """{ scan(columns: {alias: "bool", ifElse: [{name: "bool"}, {name: "int32"}, {name: "float"}]}) {
        column(name: "bool") { type } } }"""
    )
    assert data == {'scan': {'column': {'type': 'float'}}}
    with pytest.raises(ValueError, match="no kernel"):
        executor("""{ scan(columns: {alias: "bool",
            ifElse: [{name: "struct"}, {name: "int32"}, {name: "float"}]}) { type } }""")


def test_long(executor):
    with pytest.raises(ValueError, match="Long cannot represent value"):
        executor('{ filter(int64: {eq: 0.0}) { length } }')


def test_base64(executor):
    data = executor("""{ scan(columns: {alias: "binary", binary: {length: {name: "binary"}}}) {
        column(name: "binary") { ...on IntColumn { values } } } }""")
    assert data == {'scan': {'column': {'values': [0, None]}}}
    data = executor(
        """{ scan(columns: {alias: "binary", binary: {repeat: [{name: "binary"}, {value: 2}]}}) {
        columns { binary { values } } } }"""
    )
    assert data == {'scan': {'columns': {'binary': {'values': ['', None]}}}}
    data = executor(
        '{ apply(fillNullForward: {name: "binary"}) { columns { binary { values } } } }'
    )
    assert data == {'apply': {'columns': {'binary': {'values': ['', '']}}}}
    data = executor(
        """{ scan(columns: {alias: "binary", coalesce: [{name: "binary"}, {base64: "Xw=="}]}) {
        columns { binary { values } } } }"""
    )
    assert data == {'scan': {'columns': {'binary': {'values': ['', 'Xw==']}}}}
    data = executor("""{ scan(columns: {alias: "binary", binary: {joinElementWise: [
        {name: "binary"}, {name: "binary"}, {base64: "Xw=="}], nullHandling: "replace"}}) {
        columns { binary { values } } } }""")
    assert data == {'scan': {'columns': {'binary': {'values': ['Xw==', 'Xw==']}}}}
    data = executor("""{ scan(columns: {alias: "binary", binary: {replaceSlice: {name: "binary"}
        start: 0, stop: 1, replacement: "Xw=="}}) { columns { binary { values } } } }""")
    assert data == {'scan': {'columns': {'binary': {'values': ['Xw==', None]}}}}
    data = executor('{ scan(filter: {eq: [{name: "binary"}, {base64: "Xw=="}]}) { length } }')
    assert data == {'scan': {'length': 0}}
