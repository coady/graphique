"""
ASGI GraphQL utilities.
"""

import warnings
from collections.abc import Iterable, Mapping
from datetime import timedelta
from keyword import iskeyword
from typing import Optional
import pyarrow.dataset as ds
import strawberry.asgi
from strawberry import Info, UNSET
from strawberry.extensions import tracing
from strawberry.utils.str_converters import to_camel_case
from .core import Column as C
from .inputs import Filter
from .interface import Dataset, Root
from .models import Column, doc_field
from .scalars import Long, scalar_map, type_map


class MetricsExtension(tracing.ApolloTracingExtension):
    """Human-readable metrics from apollo tracing."""

    def get_results(self) -> dict:
        tracing = super().get_results()['tracing']
        metrics = self.execution_context.context.get('metrics', {})
        resolvers = []
        for resolver in tracing['execution']['resolvers']:  # pragma: no cover
            path = tuple(resolver['path'])
            resolvers.append({'path': path, 'duration': self.duration(resolver)})
            resolvers[-1].update(metrics.get(path, {}))
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
    extensions = (ContextExtension,)

    def __init__(self, root: Root, debug: bool = False, **kwargs):
        options: dict = dict(self.options, extensions=self.extensions)
        if debug:  # pragma: no cover
            options['extensions'] += (MetricsExtension,)
        if type(root).__name__ == 'Query':
            self.root_value = root
            options['enable_federation_2'] = True
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
    types = {name: types[name] for name in types if name.isidentifier() and not iskeyword(name)}
    if invalid := set(schema.names) - set(types):
        warnings.warn(f'invalid field names: {invalid}')
    prefix = to_camel_case(name.title())

    namespace = {name: strawberry.field(default=UNSET, name=name) for name in types}
    annotations = {name: Optional[Column.registry[types[name]]] for name in types}
    cls = type(prefix + 'Columns', (), dict(namespace, __annotations__=annotations))
    Columns = strawberry.type(cls, description="fields for each column")

    namespace = {name: strawberry.field(default=UNSET, name=name) for name in types}
    annotations = {name: Optional[Column if cls is list else cls] for name, cls in types.items()}
    cls = type(prefix + 'Row', (), dict(namespace, __annotations__=annotations))
    Row = strawberry.type(cls, description="scalar fields")

    class Table(Dataset):
        __init__ = Dataset.__init__
        field = name

        def columns(self, info: Info) -> Columns:  # type: ignore
            """fields for each column"""
            return Columns(**super().columns(info))

        def row(self, info: Info, index: Long = 0) -> Optional[Row]:  # type: ignore
            """Return scalar values at index."""
            row = super().row(info, index)
            for name, value in row.items():
                if isinstance(value, Column) and types[name] is not list:
                    raise TypeError(f"Field `{name}` cannot represent `Column` value")
            return Row(**row)

    if types:
        for field in ('filter', 'columns', 'row'):
            setattr(Table, field, doc_field(getattr(Table, field)))
        Table.filter.type = Table
        Table.filter.base_resolver.arguments = list(Filter.resolve_args(types))
    options = dict(name=prefix + 'Table', description="a dataset with a derived schema")
    if name:
        return strawberry.federation.type(Table, keys=keys, **options)(root)
    return strawberry.type(Table, **options)(root)
