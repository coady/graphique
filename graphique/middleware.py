"""
ASGI GraphQL utilities.
"""
from datetime import timedelta
from typing import Iterable, Mapping, Optional
import pyarrow.dataset as ds
import strawberry.asgi
from strawberry import UNSET
from strawberry.extensions import tracing
from strawberry.utils.str_converters import to_camel_case
from strawberry.types import Info
from .core import Column as C
from .inputs import Filter
from .interface import Dataset, Root
from .models import Column, doc_field
from .scalars import Long, scalar_map, type_map


class MetricsExtension(tracing.ApolloTracingExtension):
    """Human-readable metrics from apollo tracing."""

    def get_results(self) -> dict:
        tracing = super().get_results()['tracing']
        resolvers = []
        for resolver in tracing['execution']['resolvers']:  # pragma: no cover
            resolvers.append({'path': resolver['path'], 'duration': self.duration(resolver)})
        metrics = {'duration': self.duration(tracing), 'execution': {'resolvers': resolvers}}
        return {'metrics': metrics}

    @staticmethod
    def duration(data: dict) -> Optional[str]:
        return data['duration'] and str(timedelta(microseconds=data['duration'] / 1e3))


class ContextExtension(strawberry.extensions.SchemaExtension):
    """Adds registered context keys to extensions."""

    keys = ('deprecations',)

    def get_results(self) -> dict:
        context = self.execution_context.context
        return {key: context[key] for key in self.keys if key in context}


class GraphQL(strawberry.asgi.GraphQL):
    """ASGI GraphQL app with root value(s).

    Args:
        root: root dataset to attach as the Query type
        debug: enable timing extension
        **kwargs: additional `asgi.GraphQL` options
    """

    options = dict(types=Column.registry.values(), scalar_overrides=scalar_map)
    extensions = ContextExtension, MetricsExtension

    def __init__(self, root: Root, debug: bool = False, **kwargs):
        options = dict(self.options, extensions=self.extensions if debug else [])
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
