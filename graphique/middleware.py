"""
ASGI GraphQL utilities.
"""

import warnings
from collections.abc import Iterable, Mapping
from datetime import timedelta

import ibis
import strawberry.asgi
import strawberry.schema.config
from strawberry import UNSET, Info
from strawberry.extensions import tracing
from strawberry.utils.str_converters import to_camel_case

from .inputs import Filter
from .interface import Dataset, Source, ibis_schema
from .models import Column, doc_field
from .scalars import BigInt, scalar_map, schema_types


class MetricsExtension(tracing.ApolloTracingExtension):
    """Human-readable metrics from apollo tracing."""

    def get_results(self) -> dict:
        tracing = super().get_results()["tracing"]
        metrics = self.execution_context.context.get("metrics", {})
        resolvers = []
        for resolver in tracing["execution"]["resolvers"]:  # pragma: no cover
            path = tuple(resolver["path"])
            resolvers.append({"path": path, "duration": self.duration(resolver)})
            resolvers[-1].update(metrics.get(path, {}))
        metrics = {"duration": self.duration(tracing), "execution": {"resolvers": resolvers}}
        return {"metrics": metrics}

    @staticmethod
    def duration(data: dict) -> str | None:
        return data["duration"] and str(timedelta(microseconds=data["duration"] / 1e3))


class GraphQL(strawberry.asgi.GraphQL):
    """ASGI GraphQL app with root value(s).

    Args:
        root: root dataset, Query type, or instance to attach as the Query root
        extensions: additional extensions to enable
        **kwargs: additional `asgi.GraphQL` options
    """

    options = dict(
        types=Column.registry.values(),
        config=strawberry.schema.config.StrawberryConfig(scalar_map=scalar_map),
    )

    def __init__(self, root: Source | type | object, extensions: Iterable = (), **kwargs):
        self.root_value, Schema = root, strawberry.federation.Schema
        if isinstance(root, Source):
            self.root_value, Schema = implement(ibis_schema(root))(source=root), strawberry.Schema
        elif isinstance(root, type):  # pragma: no branch
            self.root_value = root_value(root)
        schema = Schema(type(self.root_value), extensions=extensions, **self.options)
        super().__init__(schema, **kwargs)

    async def get_root_value(self, request):
        return self.root_value

    @classmethod
    def federated(cls, roots: Mapping[str, Source], keys: Mapping[str, Iterable] = {}, **kwargs):
        """Deprecated: construct GraphQL app with multiple federated datasets.

        Create a `Query` class with typed fields using `implement` instead. See customize docs.
        """
        warnings.warn("use a Query class with attributes instead", DeprecationWarning)
        root_values = {
            name: implement(ibis_schema(value), name, keys.get(name, ()))(source=value)
            for name, value in roots.items()
        }
        return cls(type("Query", (), root_values), **kwargs)


def root_value(cls: type):
    """Convert a `Query` class to be a root value instance with typed fields.

    ```python
    class Query:
        name = source
        ...

    @strawberry.type
    class Query:
        name: Table
        ...

    Query(name=Table(source=source), ...)
    ```
    """
    data = {}
    for name, value in cls.__dict__.items():
        if isinstance(value, Source):
            value = implement(ibis_schema(value), name)(source=value)
        if isinstance(value, Dataset):
            data[name] = value
    annotations = {name: type(data[name]) for name in data}
    cls = type(cls.__name__, cls.__bases__, {"__annotations__": annotations})
    return strawberry.type(cls)(**data)


def implement(schema: ibis.Schema, name: str = "", keys: Iterable = ()) -> type[Dataset]:
    """Create `Table` type which implements `Dataset` interface with schema-aware fields.

    Args:
        schema: ibis schema
        name: optional name of the dataset, prefixed to the type names
        keys: keys for federation
    """
    types = dict(schema_types(schema))
    prefix = to_camel_case(name.title())
    Columns = columns_type(types, prefix)
    Row = row_type(types, prefix)

    class Table(Dataset):
        field = name

        def columns(self, info: Info) -> Columns:  # type: ignore
            """fields for each column"""
            return Columns(**super().columns(info))

        def row(self, info: Info, index: BigInt = 0) -> Row | None:  # type: ignore
            """Return scalar values at index."""
            row = super().row(info, index)
            for name, value in row.items():
                if isinstance(value, Column) and types[name] is not list:
                    raise TypeError(f"Field `{name}` cannot represent `Column` value")
            return Row(**row)

    if types:
        for field in ("filter", "columns", "row"):
            setattr(Table, field, doc_field(getattr(Table, field)))
        Table.filter.type = Table
        Table.filter.base_resolver.arguments = list(Filter.resolve_args(types))
    options = dict(name=prefix + "Table", description="a dataset with a derived schema")
    if name:
        return strawberry.federation.type(Table, keys=keys, **options)
    return strawberry.type(Table, **options)


def row_type(types: Mapping, prefix: str = "") -> type:
    """Return typed `Row`."""
    namespace = {name: strawberry.field(default=UNSET, name=name) for name in types}
    annotations = {name: (Column if cls is list else cls) | None for name, cls in types.items()}
    cls = type(prefix + "Row", (), dict(namespace, __annotations__=annotations))
    return strawberry.type(cls, description="scalar fields")


def columns_type(types: Mapping, prefix: str = "") -> type:
    """Return typed `Columns`."""
    namespace = {name: strawberry.field(default=UNSET, name=name) for name in types}
    annotations = {name: Column.registry[types[name]] | None for name in types}
    cls = type(prefix + "Columns", (), dict(namespace, __annotations__=annotations))
    return strawberry.type(cls, description="fields for each column")
