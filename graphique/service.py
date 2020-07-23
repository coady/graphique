from typing import List, Optional
import graphql
import pyarrow as pa
import pyarrow.parquet as pq
import strawberry.asgi
from starlette.applications import Starlette
from strawberry.utils.str_converters import to_camel_case
from .core import Column as C, Table as T
from .models import Long, column_map, query_map, resolvers, selections, type_map
from .settings import COLUMNS, DEBUG, DICTIONARIES, INDEX, MMAP, PARQUET_PATH

table = pq.read_table(
    PARQUET_PATH, COLUMNS, memory_map=MMAP, use_legacy_dataset=True, read_dictionary=DICTIONARIES
)
indexed = T.index(table) if INDEX is None else list(INDEX)
types = {name: type_map[tp.id] for name, tp in T.types(table).items()}
to_snake_case = {to_camel_case(name): name for name in types}.__getitem__
query_map = {
    to_camel_case(name): graphql.GraphQLArgument(query_map[tp].graphql_type)  # type: ignore
    for name, tp in types.items()
}


def resolver(name):
    column = column_map[types[name]]

    def method(self) -> column:
        return column(self.table[name])

    return strawberry.field(method, description=column.__doc__)


@strawberry.type
class Columns:
    """fields for each column"""

    locals().update({name: resolver(name) for name in types})

    def __init__(self, table):
        self.table = table


@strawberry.type
class Row:
    """scalar fields"""

    __annotations__ = {name: Optional[types[name]] for name in types}
    locals().update(dict.fromkeys(types))


@strawberry.type
class Table:
    """a column-oriented table"""

    def __init__(self, table):
        self.table = table

    @strawberry.field
    def length(self) -> Long:
        """number of rows"""
        return len(self.table)  # type: ignore

    @strawberry.field
    def columns(self) -> Columns:
        """fields for each column"""
        return Columns(self.table)

    @strawberry.field
    def row(self, info, index: Long = 0) -> Row:  # type: ignore
        """Return scalar values at index."""
        names = map(to_snake_case, selections(*info.field_nodes))
        return Row(**{name: self.table[name][index].as_py() for name in names})  # type: ignore

    @strawberry.field
    def slice(self, offset: Long = 0, length: Long = None) -> 'Table':  # type: ignore
        """Return table slice."""
        return Table(self.table.slice(offset, length))

    @strawberry.field
    def group(self, by: List[str], reverse: bool = False, length: Long = None) -> List['Table']:
        """Return tables grouped by columns, with stable ordering."""
        tables = T.grouped(self.table, *by, reverse=reverse, length=length)
        return list(map(Table, tables))

    @strawberry.field
    def unique(self, by: List[str], reverse: bool = False) -> 'Table':
        """Return table of first or last occurrences grouped by columns, with stable ordering."""
        by = list(map(to_snake_case, by))
        tables = [T.unique(table, by[-1], reverse) for table in T.grouped(self.table, *by[:-1])]
        return Table(pa.concat_tables(tables[::-1] if reverse else tables))

    @strawberry.field
    def sort(self, by: List[str], reverse: bool = False, length: Long = None) -> 'Table':
        """Return table slice sorted by specified columns.
        Optimized for a single column with fixed length.
        """
        indices = T.argsort(self.table, *map(to_snake_case, by), reverse=reverse, length=length)
        return Table(self.table.take(indices))

    @strawberry.field
    def min(self, by: List[str]) -> 'Table':
        """Return table with minimum values per column."""
        return Table(T.matched(self.table, C.min, *map(to_snake_case, by)))

    @strawberry.field
    def max(self, by: List[str]) -> 'Table':
        """Return table with maximum values per column."""
        return Table(T.matched(self.table, C.max, *map(to_snake_case, by)))

    @strawberry.field
    def filter(self, **queries) -> 'Table':
        """Return table with rows which match all queries."""
        predicates = {to_snake_case(name): resolvers.predicate(**queries[name]) for name in queries}
        return Table(T.filtered(self.table, predicates, invert=False))

    @strawberry.field
    def exclude(self, **queries) -> 'Table':
        """Return table with rows which don't match all queries; inverse of filter."""
        predicates = {to_snake_case(name): resolvers.predicate(**queries[name]) for name in queries}
        return Table(T.filtered(self.table, predicates, invert=True))


Table.filter.graphql_type.args.update(query_map)
Table.exclude.graphql_type.args.update(query_map)


@strawberry.type
class IndexedTable(Table):
    """a table sorted by a composite index"""

    def __init__(self, table):
        self.table = table

    @strawberry.field
    def index(self) -> List[str]:
        """indexed columns"""
        return list(map(to_camel_case, indexed))

    @strawberry.field
    def search(self, **queries) -> Table:
        """Return table with matching values for compound `index`.
        Queries must be a prefix of the `index`.
        Only one non-equal query is allowed, and applied last.
        """
        table = self.table
        for name in indexed:
            query = queries.pop(to_camel_case(name), None)
            if query is None:
                break
            if 'equal' in query:
                table = T.isin(table, name, query.pop('equal'))
            if query and queries:  # pragma: no cover
                raise ValueError(f"non-equal query for {name} not last; have {queries} remaining")
            if 'notEqual' in query:
                table = T.not_equal(table, name, query['notEqual'])
            if 'isin' in query:
                table = T.isin(table, name, *query['isin'])
            lower, upper = query.get('greater'), query.get('less')
            includes = {'include_lower': False, 'include_upper': False}
            if 'greaterEqual' in query and (lower is None or query['greaterEqual'] > lower):
                lower, includes['include_lower'] = query['greaterEqual'], True
            if 'lessEqual' in query and (upper is None or query['lessEqual'] > upper):
                upper, includes['include_upper'] = query['lessEqual'], True
            if {lower, upper} != {None}:
                table = T.range(table, name, lower, upper, **includes)
        if queries:  # pragma: no cover
            raise ValueError(f"expected query for {name}; have {queries} remaining")
        return Table(table)

    search.graphql_type.args.update({name: query_map[name] for name in map(to_camel_case, indexed)})


Query = IndexedTable if indexed else Table
schema = strawberry.Schema(query=Query)
app = Starlette(debug=DEBUG)
app.add_route('/graphql', strawberry.asgi.GraphQL(schema, root_value=Query(table), debug=DEBUG))
