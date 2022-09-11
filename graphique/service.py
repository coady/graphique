"""
GraphQL service and top-level resolvers.
"""
from typing import Optional
import pyarrow as pa
import pyarrow.dataset as ds
import strawberry
from strawberry.types import Info
from .core import Column as C, Table as T
from .inputs import Expression, Query as QueryInput, default_field
from .middleware import Dataset, GraphQL
from .models import Column, doc_field
from .scalars import Long, type_map
from .settings import COLUMNS, DEBUG, DICTIONARIES, FEDERATED, FILTERS, PARQUET_PATH

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
        name: Optional[Column if types[name] is list else types[name]] for name in types
    }
    locals().update({name: default_field(name=name) for name in __annotations__})


@strawberry.type(description="a column-oriented table")
class Table(Dataset):
    __init__ = Dataset.__init__

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

    @QueryInput.resolve_types(types)
    def filter(self, info: Info, **queries) -> 'Table':
        """Return table with rows which match all queries.

        See `scan(filter: ...)` for more advanced queries.
        """
        table = self.table
        search = isinstance(table, pa.Table) and info.path.prev is None
        for name in self.schema().index if search else []:
            assert not table[name].null_count, f"binary search requires non-null columns: {name}"
            query = dict(queries.pop(name))
            if not query:
                break
            if 'eq' in query:
                table = T.is_in(table, name, *query.pop('eq'))
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
            if query:
                break
        self = type(self)(table)
        expr = Expression.from_query(**queries)
        return self if expr.to_arrow() is None else self.scan(info, filter=expr)


if isinstance(COLUMNS, dict):
    COLUMNS = {alias: ds.field(name) for alias, name in COLUMNS.items()}
if FILTERS is not None:
    root = dataset.to_table(columns=COLUMNS, filter=Expression.from_query(**FILTERS).to_arrow())
elif COLUMNS:
    root = dataset.scanner(columns=COLUMNS)

app = GraphQL(Table(root), debug=DEBUG, federated=FEDERATED)
