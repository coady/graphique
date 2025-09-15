import pytest


def test_columns(executor):
    def execute(query):
        return executor(f'{{ columns {query} }}')['columns']

    assert execute('{ int32 { values } }') == {'int32': {'values': [0, None]}}
    data = execute('{ int32 { fillNull(value: 1) } }')
    assert data == {'int32': {'fillNull': [0, 1]}}
    assert execute('{ int32 { type } }') == {'int32': {'type': 'int32'}}
    assert execute('{ int32 { min max } }')

    assert execute('{ int64 { values } }') == {'int64': {'values': [0, None]}}
    data = execute('{ int64 { fillNull(value: 1) } }')
    assert data == {'int64': {'fillNull': [0, 1]}}
    assert execute('{ int64 { min max } }')

    assert execute('{ float64 { values } }') == {'float64': {'values': [0.0, None]}}
    assert execute('{ float64 { min max } }')

    assert execute('{ date { values } }') == {'date': {'values': ['1970-01-01', None]}}
    data = execute('{ date { fillNull(value: "1971-01-01") } }')
    assert data == {'date': {'fillNull': ['1970-01-01', '1971-01-01']}}
    assert execute('{ date { min max } }')
    assert execute('{ date { first last } }')

    data = execute('{ timestamp { values } }')
    assert data == {'timestamp': {'values': ['1970-01-01T00:00:00', None]}}
    assert execute('{ timestamp { min max } }')

    assert execute('{ time { values } }') == {'time': {'values': ['00:00:00', None]}}
    assert execute('{ time { min max } }')

    for name in ('bytes', 'string'):
        assert execute(f'{{ {name} {{ values }} }}') == {name: {'values': ['', None]}}
        assert execute(f'{{ {name} {{ dropNull }} }}') == {name: {'dropNull': ['']}}
        data = execute(f'{{ {name} {{ fillNull(value: "") }} }}')
        assert data == {name: {'fillNull': ['', '']}}


def test_boolean(executor):
    def execute(query):
        return executor(f'{{ columns {query} }}')['columns']

    assert execute('{ boolean { values } }') == {'boolean': {'values': [False, None]}}
    assert execute('{ boolean { type } }') == {'boolean': {'type': 'boolean'}}
    assert execute('{ boolean { any all } }') == {'boolean': {'any': False, 'all': False}}

    data = executor("""{ filter(where: {xor: [{name: "boolean"}, {inv: {name: "boolean"}}]})
        { count } }""")
    assert data == {'filter': {'count': 1}}
    data = executor('{ fillNull(name: "boolean", value: true) { columns { boolean { values } } } }')
    assert data == {'fillNull': {'columns': {'boolean': {'values': [False, True]}}}}


def test_decimal(executor):
    def execute(query):
        return executor(f'{{ columns {query} }}')['columns']

    assert execute('{ decimal { values } }') == {'decimal': {'values': ['0', None]}}
    assert execute('{ decimal { min max } }')


def test_numeric(executor):
    for name in ('int32', 'int64', 'float64'):
        data = executor(f'{{ columns {{ {name} {{ mean std var }} }} }}')
        assert data == {'columns': {name: {'mean': 0.0, 'std': None, 'var': None}}}
        data = executor(f'{{ columns {{ {name} {{ mode }} }} }}')
        assert data == {'columns': {name: {'mode': 0}}}
        data = executor(f'{{ columns {{ {name} {{ quantile(approx: true) }} }} }}')
        assert data == {'columns': {name: {'quantile': [0.0]}}}

    data = executor('{ column(name: "float64", cast: "int32") { type } }')
    assert data == {'column': {'type': 'int32'}}
    data = executor("""{ project(columns: {alias: "float64", coalesce: [{name: "float64"}, {name: "int32"}]})
        { columns { float64 { values } } } }""")
    assert data == {'project': {'columns': {'float64': {'values': [0.0, None]}}}}
    data = executor("""{ fillNull(name: ["int32", "float64"], value: 1)
        { columns { int32 { values } float64 { values } } } }""")
    assert data == {
        'fillNull': {'columns': {'int32': {'values': [0, 1]}, 'float64': {'values': [0, 1]}}}
    }


def test_datetime(executor):
    for name in ('timestamp', 'date'):
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
        assert data['project']['column']['type'] in ('timestamp(6)', 'date')
    data = executor("""{ project(columns: {alias: "timestamp", temporal: {strftime: {name: "timestamp"}, formatStr: "%Y"}})
        { column(name: "timestamp") { type } } }""")
    assert data == {'project': {'column': {'type': 'string'}}}
    for name in ('timestamp', 'time'):
        data = executor(
            f"""{{ project(columns: {{alias: "hour", temporal: {{hour: {{name: "{name}"}}}}}})
            {{ column(name: "hour") {{ ... on IntColumn {{ values }} }} }} }}"""
        )
        assert data == {'project': {'column': {'values': [0, None]}}}
    data = executor("""{ filter(where: {gt: [{name: "date"}, {scalar: {date: "1970-01-01"}}]}) 
        { count } }""")
    assert data == {'filter': {'count': 0}}
    data = executor("""{ filter(where: {gt: [{name: "timestamp"}, {scalar: {datetime: "1970-01-01"}}]}) 
        { count } }""")
    assert data == {'filter': {'count': 0}}


def test_duration(executor):
    data = executor("""{ project(columns: {alias: "diff", sub: [{name: "timestamp"}, {name: "timestamp"}]})
        { schema { types } } }""")
    assert "interval('s')" in data['project']['schema']['types']
    data = executor("""{ project(columns: {alias: "diff", temporal:
        {delta: [{name: "timestamp"}, {name: "timestamp"}], unit: "day"}})
        { column(name: "diff") { ... on BigIntColumn { values } } } }""")
    assert data == {'project': {'column': {'values': [0, None]}}}
    data = executor("""{ project(columns: {alias: "date", add: [{name: "date"}, {scalar: {duration: "P1D"}}]}) 
        { columns { date { values } } } }""")
    assert data == {'project': {'columns': {'date': {'values': ["1970-01-02", None]}}}}


def test_array(executor):
    data = executor('{ columns { array { count type } } }')
    assert data == {'columns': {'array': {'count': 1, 'type': 'array<int32>'}}}
    data = executor('{ row { array { ... on IntColumn { values } } } }')
    assert data == {'row': {'array': {'values': [0, 1, 2]}}}
    data = executor('{ row(index: -1) { array { ... on IntColumn { values } } } }')
    assert data == {'row': {'array': {'values': []}}}

    data = executor("""{ project(columns: {array: {index: [{name: "array"}, {value: 1}]}, alias: "list"}) {
        column(name: "list") { ... on BigIntColumn { values } } } }""")
    assert data == {'project': {'column': {'values': [1, None]}}}
    data = executor("""{ project(columns: {array: {unique: {name: "array"}}, alias: "a"})
        { unnest(name: "a") { column(name: "a") { ... on IntColumn { values } } } } }""")
    assert set(data['project']['unnest']['column']['values']) == {0, 1, 2}
    data = executor("""{ project(columns: {array: {modes: {name: "array"}}, alias: "list"})
        { column(name: "list") { type } } }""")
    assert data == {'project': {'column': {'type': 'int32'}}}
    data = executor("""{ project(columns: {array: {value: {name: "array"}, offset: -1}, alias: "a"})
        { column(name: "a") { ... on IntColumn { values } } } }""")
    assert data == {'project': {'column': {'values': [2, None]}}}


def test_struct(executor):
    data = executor('{ columns { struct { names } } }')
    assert data == {'columns': {'struct': {'names': ['x', 'y']}}}
    data = executor("""{ project(columns: {alias: "leaf", name: ["struct", "x"]})
        { column(name: "leaf") { ... on IntColumn { values } } } }""")
    assert data == {'project': {'column': {'values': [0, None]}}}
    data = executor('{ column(name: ["struct", "x"]) { type } }')
    assert data == {'column': {'type': 'int32'}}
    data = executor('{ row { struct } columns { struct { first } } }')
    assert data['row']['struct'] == data['columns']['struct']['first'] == {'x': 0, 'y': None}


def test_conditions(executor):
    data = executor("""{ project(columns: {alias: "bool", ifelse: [{name: "boolean"}, {name: "int32"}, {name: "float64"}]})
        { column(name: "bool") { type } } }""")
    assert data == {'project': {'column': {'type': 'float64'}}}


def test_bigint(executor):
    with pytest.raises(ValueError, match="BigInt cannot represent value"):
        executor('{ filter(int64: {eq: 0.0}) { type } }')


def test_base64(executor):
    data = executor("""{ project(columns: {alias: "bytes", coalesce: [{name: "bytes"}, {scalar: {base64: "Xw=="}}]})
        { columns { bytes { values } } } }""")
    assert data == {'project': {'columns': {'bytes': {'values': ['', 'Xw==']}}}}
    data = executor("""{ filter(where: {eq: [{name: "bytes"}, {scalar: {base64: "Xw=="}}]})
        { count } }""")
    assert data == {'filter': {'count': 0}}
    data = executor("""{ fillNull(name: "bytes", scalar: {base64: ""})
        { columns { bytes { values } } } }""")
    assert data == {'fillNull': {'columns': {'bytes': {'values': ['', '']}}}}
