"""
ASGI GraphQL utilities.
"""
from datetime import datetime
from typing import Iterable, Mapping, Optional, Union
import pyarrow as pa
import pyarrow.dataset as ds
import strawberry.asgi
from strawberry import UNSET
from strawberry.utils.str_converters import to_camel_case
from strawberry.types import Info
from .core import Column as C
from .inputs import Filter
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
    """ASGI GraphQL app with root value(s).

    Args:
        root: root dataset to attach as the Query type
        debug: enable timing extension
        **kwargs: additional `asgi.GraphQL` options
    """

    options = dict(types=Column.registry.values(), scalar_overrides=scalar_map)

    def __init__(self, root: Root, debug: bool = False, **kwargs):
        options = dict(self.options, extensions=[TimingExtension] * bool(debug))
        if type(root).__name__ == 'Query':
            self.root_value = root
            options['enable_federation_2'] = True  # type: ignore
            schema = strawberry.federation.Schema(type(self.root_value), **options)
        else:
            self.root_value = implemented(root)
            schema = strawberry.Schema(type(self.root_value), **options)
        super().__init__(schema, debug=debug, **kwargs)

    async def get_root_value(self, request):
        return self.root_value

    @classmethod
    def federated(cls, roots: Mapping[str, Root], keys: Mapping[str, Iterable] = {}, **kwargs):
        """Construct GraphQL app with multiple federated datasets.

        Args:
            roots: mapping of field names to root datasets
            keys: mapping of optional federation keys for each root
            **kwargs: additional `asgi.GraphQL` options
        """
        root_values = {name: implemented(roots[name], name, keys.get(name, ())) for name in roots}
        annotations = {name: type(root_values[name]) for name in root_values}
        Query = type('Query', (), {'__annotations__': annotations})
        return cls(strawberry.type(Query)(**root_values), **kwargs)


def implemented(root: Root, name: str = '', keys: Iterable = ()):
    """Return type which extends the Dataset interface with knowledge of the schema."""
    schema = root.projected_schema if isinstance(root, ds.Scanner) else root.schema
    types = {field.name: type_map[C.scalar_type(field).id] for field in schema}
    prefix = to_camel_case(name.title())

    namespace = {name: strawberry.field(default=UNSET, name=name) for name in types}
    annotations = {name: Column.registry[types[name]] for name in types}  # type: ignore
    cls = type(prefix + 'Columns', (), dict(namespace, __annotations__=annotations))
    Columns = strawberry.type(cls, description="fields for each column")

    namespace = {name: strawberry.field(default=UNSET, name=name) for name in types}
    annotations = {name: Optional[Column if cls is list else cls] for name, cls in types.items()}
    cls = type(prefix + 'Row', (), dict(namespace, __annotations__=annotations))
    Row = strawberry.type(cls, description="scalar fields")

    class Table(Dataset):
        __init__ = Dataset.__init__
        field = name
        filter = Filter.resolve_types(types)(Dataset.filter)

        @doc_field
        def columns(self, info: Info) -> Columns:  # type: ignore
            """fields for each column"""
            return Columns(**super().columns(info))

        @doc_field
        def row(self, info: Info, index: Long = 0) -> Row:  # type: ignore
            """Return scalar values at index."""
            row = super().row(info, index)
            for name, value in row.items():
                if isinstance(value, Column) and Column not in Row.__annotations__[name].__args__:
                    raise TypeError(f"Field `{name}` cannot represent `Column` value")
            return Row(**row)

    Table.filter.base_resolver.type_annotation = Table
    options = dict(name=prefix + 'Table', description="a column-oriented table")
    if name:
        return strawberry.federation.type(Table, keys=keys, **options)(root)
    return strawberry.type(Table, **options)(root)
