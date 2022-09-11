"""
ASGI GraphQL utilities.
"""
from datetime import datetime
from typing import Mapping, Optional, Union
import pyarrow as pa
import pyarrow.dataset as ds
import strawberry.asgi
from strawberry.types import Info
from .core import Column as C, Table as T
from .inputs import Expression, Query, default_field
from .interface import Dataset
from .models import Column, doc_field
from .scalars import Long, scalar_map, type_map

Root = Union[ds.Dataset, ds.Scanner, pa.Table]


class TimingExtension(strawberry.extensions.Extension):
    def on_request_start(self):
        self.start = datetime.now()

    def on_request_end(self):
        end = datetime.now()
        print(f"[{end.replace(microsecond=0)}]: {end - self.start}")


class GraphQL(strawberry.asgi.GraphQL):
    """ASGI GraphQL object with root value(s).

    Args:
        root_value: a mapping indicates federation and names will be prefixed,
            otherwise a single root value is attached to the Query type
        debug: enable timing extension
    """

    def __init__(self, root: Union[Root, Mapping[str, Root]], debug: bool = False, **kwargs):
        Schema = strawberry.Schema
        if isinstance(root, Mapping):
            Schema = strawberry.federation.Schema
            root_value = federated(**root)
        else:
            root_value = implement(root)
        schema = Schema(
            type(root_value),
            types=Column.type_map.values(),  # type: ignore
            extensions=[TimingExtension] * bool(debug),
            scalar_overrides=scalar_map,
        )
        super().__init__(schema, debug=debug, **kwargs)
        self.root_value = root_value

    async def get_root_value(self, request):
        return self.root_value


def federated(**roots: Root):
    """Return root Query with multiple datasets implemented."""
    root_values = {name: implement(roots[name], name.title()) for name in roots}
    annotations = {name: type(root_values[name]) for name in root_values}
    Query = type('Query', (), {'__annotations__': annotations})
    return strawberry.type(Query)(**root_values)


def implement(root: Root, prefix: str = ''):
    """Return type which extends the Dataset interface with knowledge of the schema."""
    schema = root.projected_schema if isinstance(root, ds.Scanner) else root.schema
    types = {field.name: type_map[C.scalar_type(field).id] for field in schema}
    TypeName = prefix + 'Table'

    namespace = {name: default_field(name=name) for name in types}
    annotations = {name: Column.type_map[types[name]] for name in types}  # type: ignore
    cls = type(prefix + 'Columns', (), dict(namespace, __annotations__=annotations))
    Columns = strawberry.type(cls, description="fields for each column")

    namespace = {name: default_field(name=name) for name in types}
    annotations = {name: Optional[Column if cls is list else cls] for name, cls in types.items()}
    cls = type(prefix + 'Row', (), dict(namespace, __annotations__=annotations))
    Row = strawberry.type(cls, description="scalar fields")

    @strawberry.type(name=TypeName, description="a column-oriented table")
    class Table(Dataset):
        __init__ = Dataset.__init__

        @doc_field
        def columns(self, info: Info) -> Columns:  # type: ignore
            """fields for each column"""
            table = self.select(info)
            columns = {
                name: Columns.__annotations__[name](table[name]) for name in table.column_names
            }
            return Columns(**columns)

        @doc_field
        def row(self, info: Info, index: Long = 0) -> Row:  # type: ignore
            """Return scalar values at index."""
            table = self.select(info, index + 1 if index >= 0 else None)
            row = {}
            for name in table.column_names:
                scalar = table[name][index]
                row[name] = (
                    Column.fromscalar(scalar)
                    if isinstance(scalar, pa.ListScalar)
                    else scalar.as_py()
                )
            return Row(**row)

        @Query.resolve_types(types)
        def filter(self, info: Info, **queries) -> TypeName:  # type: ignore
            """Return table with rows which match all queries.

            See `scan(filter: ...)` for more advanced queries.
            """
            table = self.table
            search = isinstance(table, pa.Table) and info.path.prev is None
            for name in self.schema().index if search else []:
                assert not table[name].null_count, f"search requires non-null column: {name}"
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

    globals()[TypeName] = Table  # hack namespace to resolve recursive type
    return Table(root)
