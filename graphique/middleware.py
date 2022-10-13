"""
ASGI GraphQL utilities.
"""
from datetime import datetime
from typing import Iterable, Mapping, Optional, Union
import pyarrow as pa
import pyarrow.dataset as ds
import strawberry.asgi
from strawberry.utils.str_converters import to_camel_case
from strawberry.types import Info
from .core import Column as C
from .inputs import Filter, default_field
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
        root_value: a single root is attached as the Query type
            a mapping attaches roots to each field name, and enables federation
        keys: a mapping of optional federation keys for multiple roots
        debug: enable timing extension
    """

    def __init__(
        self,
        root: Union[Root, Mapping[str, Root]],
        keys: Mapping[str, Iterable] = {},
        debug: bool = False,
        **kwargs,
    ):
        options = dict(
            types=Column.type_map.values(),  # type: ignore
            extensions=[TimingExtension] * bool(debug),
            scalar_overrides=scalar_map,
        )
        if isinstance(root, Mapping):
            root_value = federated(root, keys)
            options['enable_federation_2'] = True
            schema = strawberry.federation.Schema(type(root_value), **options)
        else:
            assert not keys, "federation keys required named roots"
            root_value = implemented(root)
            schema = strawberry.Schema(type(root_value), **options)
        super().__init__(schema, debug=debug, **kwargs)
        self.root_value = root_value

    async def get_root_value(self, request):
        return self.root_value


def federated(roots: Mapping[str, Root], keys: Mapping[str, Iterable] = {}):
    """Return root Query with multiple datasets implemented."""
    root_values = {name: implemented(roots[name], name, keys.get(name, ())) for name in roots}
    annotations = {name: type(root_values[name]) for name in root_values}
    Query = type('Query', (), {'__annotations__': annotations})
    return strawberry.type(Query)(**root_values)


def implemented(root: Root, name: str = '', keys: Iterable = ()):
    """Return type which extends the Dataset interface with knowledge of the schema."""
    schema = root.projected_schema if isinstance(root, ds.Scanner) else root.schema
    types = {field.name: type_map[C.scalar_type(field).id] for field in schema}
    prefix = to_camel_case(name.title())
    TypeName = prefix + 'Table'

    namespace = {name: default_field(name=name) for name in types}
    annotations = {name: Column.type_map[types[name]] for name in types}  # type: ignore
    cls = type(prefix + 'Columns', (), dict(namespace, __annotations__=annotations))
    Columns = strawberry.type(cls, description="fields for each column")

    namespace = {name: default_field(name=name) for name in types}
    annotations = {name: Optional[Column if cls is list else cls] for name, cls in types.items()}
    cls = type(prefix + 'Row', (), dict(namespace, __annotations__=annotations))
    Row = strawberry.type(cls, description="scalar fields")

    options = dict(name=TypeName, description="a column-oriented table")
    if name:
        decorator = strawberry.federation.type(keys=keys, **options)
    else:
        decorator = strawberry.type(**options)

    @decorator
    class Table(Dataset):
        __init__ = Dataset.__init__
        field = name

        @doc_field
        def columns(self, info: Info) -> Columns:  # type: ignore
            """fields for each column"""
            return Columns(**super().columns(info))

        @doc_field
        def row(self, info: Info, index: Long = 0) -> Row:  # type: ignore
            """Return scalar values at index."""
            return Row(**super().row(info, index))

        @Filter.resolve_types(types)
        def filter(self, info: Info, **queries) -> TypeName:  # type: ignore
            """Return table with rows which match all queries.

            See `scan(filter: ...)` for more advanced queries.
            """
            return super().filter(info, **queries)

    globals()[TypeName] = Table  # hack namespace to resolve recursive type
    return Table(root)
