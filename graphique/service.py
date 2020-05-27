from typing import List
import graphql
import pyarrow.parquet as pq
import strawberry.asgi
from starlette.applications import Starlette
from .core import Column as C, Table as T
from .models import Long, column_map, query_map, resolvers, type_map
from .settings import DEBUG, INDEX, MMAP, PARQUET_PATH

table = pq.read_table(PARQUET_PATH, memory_map=MMAP)
indexed = list(INDEX) or T.index(table)
types = {name: type_map[tp.id] for name, tp in T.types(table).items()}


query_map = {
    name: graphql.GraphQLArgument(query_map[types[name]].graphql_type)  # type: ignore
    for name in types
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
class Table:
    """a column-oriented table"""

    def __init__(self, table):
        self.table = table

    @strawberry.field
    def length(self) -> Long:
        """number of rows"""
        return len(self.table)  # type: ignore

    @strawberry.field
    def slice(self, offset: Long = 0, length: Long = None) -> Columns:  # type: ignore
        """Return table slice with column fields."""
        # ARROW-8911: slicing an empty table may crash
        return Columns(self.table and self.table.slice(offset, length))

    @strawberry.field
    def filter(self, **queries) -> 'Table':
        """Return table with matching rows."""
        table = self.table
        for name, query in queries.items():
            table = table.filter(C.mask(table[name], resolvers.predicate(**query)))
        return Table(table)


Table.filter.graphql_type.args.update(query_map)


@strawberry.type
class IndexedTable(Table):
    """a table sorted by a composite index"""

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

    search.graphql_type.args.update({name: query_map[name] for name in indexed})


Query = IndexedTable if indexed else Table
schema = strawberry.Schema(query=Query)
app = Starlette(debug=DEBUG)
app.add_route('/graphql', strawberry.asgi.GraphQL(schema, root_value=Query(table), debug=DEBUG))
