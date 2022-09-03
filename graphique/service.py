"""
GraphQL service and top-level resolvers.
"""
from typing import List, Optional, no_type_check
import pyarrow as pa
import pyarrow.dataset as ds
import strawberry
from strawberry.types import Info
from .core import Column as C, Table as T
from .inputs import Expression, Query as QueryInput, default_field
from .middleware import Dataset, GraphQL
from .models import Column, doc_field
from .scalars import Long, type_map
from .settings import COLUMNS, DEBUG, DICTIONARIES, FEDERATED, FILTERS, INDEX, PARQUET_PATH

format = ds.ParquetFileFormat(read_options={'dictionary_columns': DICTIONARIES})
root = dataset = ds.dataset(PARQUET_PATH, format=format, partitioning='hive')
aliases = dict(map(reversed, COLUMNS.items())) if isinstance(COLUMNS, dict) else {}  # type: ignore
types = {
    aliases.get(field.name, field.name): type_map[C.scalar_type(field).id]
    for field in dataset.schema
}


@strawberry.type(description="fields for each column")
class Columns:
    __annotations__ = {name: Column.type_map[types[name]] for name in types}  # type: ignore
    locals().update({name: default_field(name=name) for name in __annotations__})


@strawberry.type(description="scalar fields")
class Row:
    __annotations__ = {
        name: Optional[Column if types[name] is list else types[name]]
        for name in types
        if types[name] is not dict
    }
    locals().update({name: default_field(name=name) for name in __annotations__})


@strawberry.type(description="a column-oriented table")
class Table(Dataset):
    @doc_field
    def columns(self, info: Info) -> Columns:
        """fields for each column"""
        table = self.select(info)
        columns = {name: Columns.__annotations__[name](table[name]) for name in table.column_names}
        return Columns(**columns)

    @doc_field
    def row(self, info: Info, index: Long = 0) -> Row:
        """Return scalar values at index."""
        table = self.select(info, index + 1 if index >= 0 else None)
        row = {}
        for name in table.column_names:
            scalar = table[name][index]
            row[name] = (
                Column.fromscalar(scalar) if isinstance(scalar, pa.ListScalar) else scalar.as_py()
            )
        return Row(**row)

    @no_type_check
    @QueryInput.resolve_types(types)
    def filter(self, info: Info, **queries) -> 'Table':
        """Return table with rows which match all queries.

        See `scan(filter: ...)` for more advanced queries.
        """
        return self.scan(info, filter=Expression.from_query(**queries))


@strawberry.type(description="a table sorted by a composite index")
class IndexedTable(Table):
    index: List[str] = strawberry.field(default=tuple(INDEX), description="the composite index")

    @QueryInput.resolve_types({name: types[name] for name in INDEX})
    def search(self, info: Info, **queries) -> Table:
        """Return table with matching values for composite `index`.

        Queries must be a prefix of the `index`.
        Only one inequality query is allowed, and must be last.
        """
        for name in queries:
            if queries[name] is None:
                raise TypeError(f"`{name}` is optional, not nullable")
        queries = {name: dict(queries[name]) for name in queries}
        if isinstance(self.table, ds.Dataset):
            return self.scan(info, filter=Expression.from_query(**queries))
        table = self.select(info)
        for name in self.index:
            if name not in queries:
                break
            query = queries.pop(name)
            if 'eq' in query:
                table = T.is_in(table, name, *query.pop('eq'))
            if query and queries:
                raise ValueError(f"inequality query for {name} not last; have {queries} remaining")
            if 'ne' in query:
                table = T.not_equal(table, name, query['ne'])
            lower, upper = query.get('gt'), query.get('lt')
            includes = {'include_lower': False, 'include_upper': False}
            if 'ge' in query and (lower is None or query['ge'] > lower):
                lower, includes['include_lower'] = query['ge'], True
            if 'le' in query and (upper is None or query['le'] > upper):
                upper, includes['include_upper'] = query['le'], True
            if {lower, upper} != {None}:
                table = T.range(table, name, lower, upper, **includes)
        if queries:
            raise ValueError(f"expected query for {name}; have {queries} remaining")
        return Table(table)


if isinstance(COLUMNS, dict):
    COLUMNS = {alias: ds.field(name) for alias, name in COLUMNS.items()}
if FILTERS is not None:
    root = dataset.to_table(columns=COLUMNS, filter=Expression.from_query(**FILTERS).to_arrow())
    for name in INDEX:
        assert not root[name].null_count, f"binary search requires non-null columns: {name}"
elif COLUMNS:
    root = dataset.scanner(columns=COLUMNS)

Query = IndexedTable if INDEX else Table
app = GraphQL(Query(root), debug=DEBUG, federated=FEDERATED)
