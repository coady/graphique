import itertools
from typing import List
import graphql
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import strawberry.asgi
from starlette.applications import Starlette
from strawberry.utils.str_converters import to_snake_case
from .core import Column as C, Table as T, flatten
from .models import Long, column_map, query_map, resolvers, type_map
from .settings import COLUMNS, DEBUG, DICTIONARIES, INDEX, MMAP, PARQUET_PATH

table = pq.read_table(PARQUET_PATH, COLUMNS, memory_map=MMAP, read_dictionary=DICTIONARIES)
indexed = T.index(table) if INDEX is None else list(INDEX)
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
    def columns(self) -> Columns:
        """fields for each column"""
        return Columns(self.table)

    @strawberry.field
    def slice(self, offset: Long = 0, length: Long = None) -> 'Table':  # type: ignore
        """Return table slice."""
        # ARROW-8911: slicing an empty table may crash
        return Table(self.table and self.table.slice(offset, length))

    @strawberry.field
    def group(  # type: ignore
        self, by: List[str], reverse: bool = False, length: Long = None
    ) -> List['Table']:
        """Return tables grouped by specified columns.
        Optimized for a single column.
        Groups are sorted and have stable ordering within a group.
        """
        groups = flatten(T.arggroupby(self.table, *map(to_snake_case, by)), reverse=reverse)
        columns = [pa.concat_arrays(column.chunks) for column in self.table.columns]
        for indices in itertools.islice(groups, length):
            yield Table(
                T.from_arrays([col.take(indices) for col in columns], self.table.column_names)
            )

    @strawberry.field
    def unique(self, by: List[str], reverse: bool = False) -> 'Table':
        """Return table of first or last occurrences grouped by specified columns.
        Optimized for a single column.
        """
        indices = T.argunique(self.table, *map(to_snake_case, by), reverse=reverse)
        return Table(T.from_pydict(T.apply(self.table, lambda col: np.take(col, indices))))

    @strawberry.field
    def sort(self, by: List[str], reverse: bool = False, length: Long = None) -> 'Table':
        """Return table slice sorted by specified columns.
        Optimized for a single column with fixed length.
        """
        indices = T.argsort(self.table, *map(to_snake_case, by), reverse=reverse, length=length)
        return Table(T.from_pydict(T.apply(self.table, lambda col: np.take(col, indices))))

    @strawberry.field
    def filter(self, **queries) -> 'Table':
        """Return table with rows which match all queries."""
        if not queries:
            return self
        predicates = {name: resolvers.predicate(**queries[name]) for name in queries}
        return Table(table.filter(T.mask(self.table, **predicates)))

    @strawberry.field
    def exclude(self, **queries) -> 'Table':
        """Return table with rows which don't match all queries; inverse of filter."""
        if not queries:
            return self
        predicates = {name: resolvers.predicate(**queries[name]) for name in queries}
        return Table(table.filter(C.mask(T.mask(self.table, **predicates), np.invert)))


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
