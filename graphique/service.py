from typing import List, Optional
import graphql
import pyarrow as pa
import pyarrow.parquet as pq
import strawberry.asgi
from starlette.applications import Starlette
from strawberry.utils.arguments import is_unset
from .core import Column as C, Table as T
from .settings import DEBUG, INDEX, MMAP, PARQUET_PATH

table = pq.read_table(PARQUET_PATH, memory_map=MMAP)
types = T.types(table)
indexed = list(INDEX) or T.index(table)
ops = 'equal', 'less', 'less_equal', 'greater', 'greater_equal'


def selections(node):
    """Return tree of field name selections."""
    nodes = getattr(node.selection_set, 'selections', [])
    return {node.name.value: selections(node) for node in nodes}


@strawberry.input
class IntQuery:
    __annotations__ = dict.fromkeys(ops, Optional[int])
    isin: Optional[List[int]]


@strawberry.input
class FloatQuery:
    __annotations__ = dict.fromkeys(ops, Optional[float])
    isin: Optional[List[float]]


@strawberry.input
class StringQuery:
    __annotations__ = dict.fromkeys(ops, Optional[str])
    isin: Optional[List[str]]


query_map = {
    int: IntQuery,
    float: FloatQuery,
    str: StringQuery,
}
query_map = {
    name: graphql.GraphQLArgument(query_map[types[name]].graphql_type)  # type: ignore
    for name in types
}


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


def convert(arg, default=None):
    return default if is_unset(arg) else arg


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
        """Return table slice with column fields."""
        return Columns(self.table.slice(offset, convert(length)))


@strawberry.type
class IndexedTable(Table):
    def __init__(self, table):
        self.table = table

    @strawberry.field
    def index(self) -> List[str]:
        """indexed columns"""
        return indexed

    @strawberry.field
    def search(self, **queries) -> Table:
        """Return table with matching values for compound `index`.
        Queries must be a prefix of the `index`.
        Only one non-equal query is allowed, and applied last.
        """
        table = self.table
        for name in indexed:
            query = queries.pop(name, None)
            if query is None:
                break
            if 'equal' in query:
                table = T.isin(table, name, query.pop('equal'))
            if query and queries:  # pragma: no cover
                raise ValueError(f"non-equal query for {name} not last; have {queries} remaining")
            if 'isin' in query:
                table = T.isin(table, name, *query['isin'])
            if 'less' in query:
                table = T.range(table, name, upper=query['less'])
            if 'lessEqual' in query:
                table = T.range(table, name, upper=query['lessEqual'], include_upper=True)
            if 'greater' in query:
                table = T.range(table, name, lower=query['greater'], include_lower=False)
            if 'greaterEqual' in query:
                table = T.range(table, name, lower=query['greaterEqual'])
        if queries:  # pragma: no cover
            raise ValueError(f"expected query for {name}; have {queries} remaining")
        return Table(table)

    search.graphql_type.args.update({name: query_map[name] for name in indexed})


Query = IndexedTable if indexed else Table
schema = strawberry.Schema(query=Query)
app = Starlette(debug=DEBUG)
app.add_route('/graphql', strawberry.asgi.GraphQL(schema, root_value=Query(table), debug=DEBUG))
