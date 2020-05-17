from typing import List, Optional
import pyarrow as pa
import pyarrow.parquet as pq
import strawberry.asgi
from starlette.applications import Starlette
from .core import Column as C, Table as T
from .settings import DEBUG, INDEX, MMAP, PARQUET_PATH

table = pq.read_table(PARQUET_PATH, memory_map=MMAP)
types = T.types(table)
index = list(INDEX) or T.index(table)


def selections(node):
    """Return tree of field name selections."""
    nodes = getattr(node.selection_set, 'selections', [])
    return {node.name.value: selections(node) for node in nodes}


def __init__(self, **kwargs):  # workaround for default inputs being overridden
    for name, value in kwargs.items():
        if value is not None:
            setattr(self, name, value)


@strawberry.input
class Equals:
    __annotations__ = {name: Optional[types[name]] for name in index}
    locals().update(dict.fromkeys(index))
    __init__ = __init__


@strawberry.input
class IsIn:
    __annotations__ = {name: Optional[List[types[name]]] for name in index}  # type: ignore
    locals().update(dict.fromkeys(index))
    __init__ = __init__


ranges = {}
namespace = {
    'lower': None,
    'upper': None,
    'include_lower': True,
    'include_upper': False,
    '__init__': __init__,
}
for cls in {types[name] for name in index}:
    name = cls.__name__.capitalize() + 'Range'
    namespace['__annotations__'] = {  # type: ignore
        'lower': Optional[cls],
        'upper': Optional[cls],
        'include_lower': Optional[bool],
        'include_upper': Optional[bool],
    }
    ranges[cls] = strawberry.input(type(name, (), namespace))


@strawberry.input
class Range:
    __annotations__ = {name: Optional[ranges[types[name]]] for name in index}
    locals().update(dict.fromkeys(index))
    __init__ = __init__


def unique(self, info):
    if 'counts' in selections(*info.field_nodes):
        values, counts = C.value_counts(self.array).flatten()
    else:
        values, counts = C.unique(self.array), pa.array([])
    return values.to_pylist(), counts.to_pylist()


@strawberry.type
class IntUnique:
    """unique ints"""

    values: List[int]
    counts: List[int]

    @strawberry.field
    def length(self) -> int:
        """number of rows"""
        return len(self.values)


@strawberry.type
class IntColumn:
    """column of ints"""

    def __init__(self, array):
        self.array = array

    @strawberry.field
    def values(self) -> List[int]:
        """list of values"""
        return self.array.to_pylist()

    @strawberry.field
    def sum(self) -> int:
        """sum of values"""
        return C.sum(self.array)

    @strawberry.field
    def min(self) -> int:
        """min of columns"""
        return C.min(self.array)

    @strawberry.field
    def max(self) -> int:
        """max of columns"""
        return C.max(self.array)

    @strawberry.field
    def unique(self, info) -> IntUnique:
        """unique values and counts"""
        return IntUnique(*unique(self, info))


@strawberry.type
class FloatColumn:
    """column of floats"""

    def __init__(self, array):
        self.array = array

    @strawberry.field
    def values(self) -> List[float]:
        """list of values"""
        return self.array.to_pylist()

    @strawberry.field
    def sum(self) -> float:
        """sum of values"""
        return C.sum(self.array)

    @strawberry.field
    def min(self) -> float:
        """min of columns"""
        return C.min(self.array)

    @strawberry.field
    def max(self) -> float:
        """max of columns"""
        return C.max(self.array)


@strawberry.type
class StringUnique:
    """unique strings"""

    values: List[str]
    counts: List[int]
    length = IntUnique.length


@strawberry.type
class StringColumn:
    """column of strings"""

    def __init__(self, array):
        self.array = array

    @strawberry.field
    def values(self) -> List[str]:
        """list of values"""
        return self.array.to_pylist()

    @strawberry.field
    def min(self) -> str:
        """min of columns"""
        return C.min(self.array)

    @strawberry.field
    def max(self) -> str:
        """max of columns"""
        return C.max(self.array)

    @strawberry.field
    def unique(self, info) -> StringUnique:
        """unique values and counts"""
        return StringUnique(*unique(self, info))


column_map = {
    int: IntColumn,
    float: FloatColumn,
    str: StringColumn,
}


def resolver(name):
    column = column_map[types[name]]

    def method(self) -> column:
        return column(self.table[name])

    return strawberry.field(method, description=column.__doc__)


@strawberry.type
class Columns:
    locals().update({name: resolver(name) for name in types})

    def __init__(self, table):
        self.table = table


@strawberry.type
class Table:
    def __init__(self, table):
        self.table = table

    @strawberry.field
    def length(self) -> int:
        """number of rows"""
        return len(self.table)

    @strawberry.field
    def slice(self, offset: int = 0, length: int = None) -> Columns:
        """Return table slice."""
        return Columns(self.table.slice(offset, length))


@strawberry.type
class Indexed(Table):
    def __init__(self, table):
        self.table = table

    @strawberry.field
    def search(
        self, info, equals: Equals = Equals(), isin: IsIn = IsIn(), range: Range = Range(),
    ) -> Table:
        f"""Return table with matching values for index: {index}.
        The values are matched in index order.
        Only one `range` or `isin` query is allowed, and applied last.
        """
        names = list(isin.__dict__) + list(range.__dict__)
        if len(names) > 1:
            raise ValueError(f"only one multi-valued selection allowed: {names}")
        names = list(equals.__dict__) + names
        if names != index[: len(names)]:
            raise ValueError(f"{names} is not a prefix of index: {index}")
        table = self.table
        for name in equals.__dict__:
            table = T.isin(table, name, getattr(equals, name))
        for name in isin.__dict__:
            table = T.isin(table, name, *getattr(isin, name))
        for name in range.__dict__:
            table = T.range(table, name, **getattr(range, name).__dict__)
        return Table(table)


Query = Indexed if index else Table
schema = strawberry.Schema(query=Query)
app = Starlette(debug=DEBUG)
app.add_route('/graphql', strawberry.asgi.GraphQL(schema, root_value=Query(table), debug=DEBUG))
