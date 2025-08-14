import pytest


def test_columns(executor):
    def execute(query):
        return executor(f'{{ columns {query} }}')['columns']

    for name in ('uint8', 'int8', 'uint16', 'int16', 'int32'):
        assert execute(f'{{ {name} {{ values }} }}') == {name: {'values': [0, None]}}
        data = execute(f'{{ {name} {{ dropNull }} }}')
        assert data == {name: {'dropNull': [0]}}
        assert execute(f'{{ {name} {{ type }} }}') == {name: {'type': name}}
        assert execute(f'{{ {name} {{ min max }} }}')
    for name in ('uint32', 'uint64', 'int64'):
        assert execute(f'{{ {name} {{ values }} }}') == {name: {'values': [0, None]}}
        data = execute(f'{{ {name} {{ dropNull }} }}')
        assert data == {name: {'dropNull': [0]}}
        assert execute(f'{{ {name} {{ min max }} }}')

    for name in ('float', 'double'):
        assert execute(f'{{ {name} {{ values }} }}') == {name: {'values': [0.0, None]}}
        data = execute(f'{{ {name} {{ dropNull }} }}')
        assert data == {name: {'dropNull': [0.0]}}
        assert execute(f'{{ {name} {{ min max }} }}')

    for name in ('date32', 'date64'):
        assert execute(f'{{ {name} {{ values }} }}') == {name: {'values': ['1970-01-01', None]}}
        data = execute(f'{{ {name} {{ dropNull }} }}')
        assert data == {name: {'dropNull': ['1970-01-01']}}
        assert execute(f'{{ {name} {{ min max }} }}')
        assert execute(f'{{ {name} {{ first last }} }}')

    data = execute('{ timestamp { values } }')
    assert data == {'timestamp': {'values': ['1970-01-01T00:00:00', None]}}
    data = execute(f'{{ {name} {{ dropNull }} }}')
    assert data == {name: {'dropNull': ['1970-01-01']}}
    assert execute('{ timestamp { min max } }')

    for name in ('time32', 'time64'):
        assert execute(f'{{ {name} {{ values }} }}') == {name: {'values': ['00:00:00', None]}}
        data = execute(f'{{ {name} {{ dropNull }} }}')
        assert data == {name: {'dropNull': ['00:00:00']}}
        assert execute(f'{{ {name} {{ min max }} }}')

    for name in ('binary', 'string'):
        assert execute(f'{{ {name} {{ values }} }}') == {name: {'values': ['', None]}}
        data = execute(f'{{ {name} {{ dropNull }} }}')
        assert data == {name: {'dropNull': ['']}}
        data = execute(f'{{ {name} {{ fillNull(value: "") }} }}')
        assert data == {name: {'fillNull': ['', '']}}


def test_boolean(executor):
    def execute(query):
        return executor(f'{{ columns {query} }}')['columns']

    assert execute('{ bool { values } }') == {'bool': {'values': [False, None]}}
    assert execute('{ bool { type } }') == {'bool': {'type': 'bool'}}
    assert execute('{ bool { any all } }') == {'bool': {'any': False, 'all': False}}

    data = executor('{ filter(where: {xor: [{name: "bool"}, {inv: {name: "bool"}}]}) { count } }')
    assert data == {'filter': {'count': 1}}


def test_decimal(executor):
    def execute(query):
        return executor(f'{{ columns {query} }}')['columns']

    assert execute('{ decimal { values } }') == {'decimal': {'values': ['0', None]}}
    assert execute('{ decimal { min max } }')


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

    data = executor('{ column(name: "float", cast: "int32") { type } }')
    assert data == {'column': {'type': 'int32'}}
    data = executor("""{ project(columns: {alias: "float", coalesce: [{name: "float"}, {name: "int32"}]})
        { columns { float { values } } } }""")
    assert data == {'project': {'columns': {'float': {'values': [0.0, None]}}}}


def test_datetime(executor):
    for name in ('timestamp', 'date32'):
        data = executor(
            f"""{{ project(columns: {{alias: "year", temporal: {{year: {{name: "{name}"}}}}}})
            {{ column(name: "year") {{ ... on IntColumn {{ values }} }} }} }}"""
        )
        assert data == {'project': {'column': {'values': [1970, None]}}}
        data = executor(
            f"""{{ project(columns: {{alias: "quarter", temporal: {{quarter: {{name: "{name}"}}}}}})
            {{ column(name: "quarter") {{ ... on IntColumn {{ values }} }} }} }}"""
        )
        assert data == {'project': {'column': {'values': [1, None]}}}
        data = executor(f"""{{ project(columns: {{alias: "{name}",
            temporal: {{truncate: {{name: "{name}"}}, unit: "D"}}}})
            {{ column(name: "{name}") {{ type }} }} }}""")
        assert data['project']['column']['type'] in ('timestamp[us]', 'date32[day]')
    data = executor("""{ project(columns: {alias: "timestamp", temporal: {strftime: {name: "timestamp"}, formatStr: "%Y"}})
        { column(name: "timestamp") { type } } }""")
    assert data == {'project': {'column': {'type': 'string'}}}
    for name in ('timestamp', 'time32'):
        data = executor(
            f"""{{ project(columns: {{alias: "hour", temporal: {{hour: {{name: "{name}"}}}}}})
            {{ column(name: "hour") {{ ... on IntColumn {{ values }} }} }} }}"""
        )
        assert data == {'project': {'column': {'values': [0, None]}}}


def test_duration(executor):
    data = executor("""{ project(columns: {alias: "diff", sub: [{name: "timestamp"}, {name: "timestamp"}]})
        { schema { types } } }""")
    assert "interval('s')" in data['project']['schema']['types']
    data = executor("""{ project(columns: {alias: "diff", temporal:
        {delta: [{name: "timestamp"}, {name: "timestamp"}], unit: "day"}})
        { column(name: "diff") { ... on LongColumn { values } } } }""")
    assert data == {'project': {'column': {'values': [0, None]}}}


def test_list(executor):
    data = executor('{ columns { list { count type } } }')
    assert data == {'columns': {'list': {'count': 1, 'type': 'list<l: int32>'}}}
    data = executor('{ columns { list { dropNull { count } } } }')
    assert data == {'columns': {'list': {'dropNull': [{'count': 3}]}}}
    data = executor('{ row { list { ... on IntColumn { values } } } }')
    assert data == {'row': {'list': {'values': [0, 1, 2]}}}
    data = executor('{ row(index: -1) { list { ... on IntColumn { values } } } }')
    assert data == {'row': {'list': None}}

    data = executor("""{ project(columns: {array: {index: [{name: "list"}, {value: 1}]}, alias: "list"}) {
        column(name: "list") { ... on LongColumn { values } } } }""")
    assert data == {'project': {'column': {'values': [1, None]}}}
    data = executor("""{ project(columns: {array: {unique: {name: "list"}}, alias: "list"})
        { columns { list { flatten { count } } } } }""")
    assert data['project']['columns']['list'] == {'flatten': {'count': 3}}
    data = executor(
        '{ project(columns: {array: {modes: {name: "list"}}, alias: "list"}) { column(name: "list") { type } } }'
    )
    assert data['project']['column']['type'] == 'int32'
    data = executor('{ columns { list { value { type } } } }')
    assert data == {'columns': {'list': {'value': {'type': 'int32'}}}}


def test_struct(executor):
    data = executor('{ columns { struct { names column(name: "x") { count } } } }')
    assert data == {'columns': {'struct': {'names': ['x', 'y'], 'column': {'count': 1}}}}
    data = executor("""{ project(columns: {alias: "leaf", name: ["struct", "x"]})
        { column(name: "leaf") { ... on IntColumn { values } } } }""")
    assert data == {'project': {'column': {'values': [0, None]}}}
    data = executor('{ column(name: ["struct", "x"]) { type } }')
    assert data == {'column': {'type': 'int32'}}
    data = executor('{ row { struct } columns { struct { value } } }')
    assert data['row']['struct'] == data['columns']['struct']['value'] == {'x': 0, 'y': None}


def test_selections(executor):
    data = executor('{ slice { count } slice { order(by: "snake_id") { count } } }')
    assert data == {'slice': {'count': 2, 'order': {'count': 2}}}
    data = executor('{ dropNull { count } }')
    assert data == {'dropNull': {'count': 1}}
    data = executor('{ dropNull { columns { float { values } } } }')
    assert data == {'dropNull': {'columns': {'float': {'values': [0.0]}}}}


def test_conditions(executor):
    data = executor("""{ project(columns: {alias: "bool", ifelse: [{name: "bool"}, {name: "int32"}, {name: "float"}]})
        { column(name: "bool") { type } } }""")
    assert data == {'project': {'column': {'type': 'float'}}}


def test_long(executor):
    with pytest.raises(ValueError, match="Long cannot represent value"):
        executor('{ filter(int64: {eq: 0.0}) { length } }')


def test_base64(executor):
    data = executor("""{ project(columns: {alias: "binary", coalesce: [{name: "binary"}, {base64: "Xw=="}]})
        { columns { binary { values } } } }""")
    assert data == {'project': {'columns': {'binary': {'values': ['', 'Xw==']}}}}
    data = executor('{ filter(where: {eq: [{name: "binary"}, {base64: "Xw=="}]}) { count } }')
    assert data == {'filter': {'count': 0}}
