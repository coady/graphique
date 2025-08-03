"""
Primary Dataset interface.

Doesn't require knowledge of the schema.
"""

# mypy: disable-error-code=valid-type
import itertools
from collections.abc import Iterable, Iterator, Mapping, Sized
from typing import TypeAlias, no_type_check
import ibis
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import strawberry.asgi
from strawberry import Info
from typing_extensions import Self
from .core import Nodes, Parquet, Table as T, order_key
from .inputs import Expression, Filter, Aggregates, IProjection, Projection, links
from .models import Column, doc_field
from .scalars import Long

Source: TypeAlias = ds.Dataset | Nodes | ibis.Table | pa.Table


def references(field) -> Iterator:
    """Generate every possible column reference from strawberry `SelectedField`."""
    if isinstance(field, str):
        yield field.lstrip('-')
    elif isinstance(field, Iterable):
        for value in field:
            yield from references(value)
        if isinstance(field, Mapping):
            for value in field.values():
                yield from references(value)
    else:
        for name in ('name', 'arguments', 'selections'):
            yield from references(getattr(field, name, []))


@strawberry.type(description="dataset schema")
class Schema:
    names: list[str] = strawberry.field(description="field names")
    types: list[str] = strawberry.field(
        description="[arrow types](https://arrow.apache.org/docs/python/api/datatypes.html), corresponding to `names`"
    )
    partitioning: list[str] = strawberry.field(description="partition keys")


@strawberry.interface(description="an arrow dataset or ibis table")
class Dataset:
    def __init__(self, source: Source):
        self.source = source

    def references(self, info: Info, level: int = 0) -> set:
        """Return set of every possible future column reference."""
        fields = info.selected_fields
        for _ in range(level):
            fields = itertools.chain(*[field.selections for field in fields])
        return set(itertools.chain(*map(references, fields))) & set(self.schema().names)

    def select(self, info: Info) -> Source:
        """Return source with only the columns necessary to proceed."""
        names = list(self.references(info))
        if len(names) >= len(self.schema().names):
            return self.source
        if isinstance(self.source, pa.Table):
            return self.source.select(names)
        if isinstance(self.source, ibis.Table):
            projection = {} if names else {'_': ibis.row_number()}
            return self.source.select(names, **projection)
        return Nodes.scan(self.source, names)

    def to_table(self, info: Info, length: int | None = None) -> pa.Table:
        """Return table with only the rows and columns necessary to proceed."""
        source = self.select(info)
        if isinstance(source, pa.Table):
            return source
        if isinstance(source, ibis.Table):
            return source.head(length).to_pyarrow()
        return source.to_table() if length is None else source.head(length)

    def to_ibis(self, info: Info) -> ibis.Table:
        """Return table with only the rows and columns necessary to proceed."""
        if isinstance(self.source, ibis.Table):
            return self.source
        if isinstance(self.source, ds.Dataset):  # pragma: no cover
            return Parquet.to_table(self.source)
        return ibis.memtable(self.to_table(info))

    @classmethod
    @no_type_check
    def resolve_reference(cls, info: Info, **keys) -> Self:
        """Return table from federated keys."""
        self = getattr(info.root_value, cls.field)
        queries = {name: Filter(eq=[keys[name]]) for name in keys}
        return self.filter(info, **queries)

    def columns(self, info: Info) -> dict:
        """fields for each column"""
        table = self.to_table(info)
        return {name: Column.cast(table[name]) for name in table.schema.names}

    def row(self, info: Info, index: int = 0) -> dict:
        """Return scalar values at index."""
        table = self.to_table(info, index + 1 if index >= 0 else None)
        row = {}
        for name in table.schema.names:
            scalar = table[name][index]
            columnar = isinstance(scalar, pa.ListScalar)
            row[name] = Column.fromscalar(scalar) if columnar else scalar.as_py()
        return row

    def filter(self, info: Info, **queries: Filter) -> Self:
        """Return table with rows which match all queries.

        See `scan(filter: ...)` for more advanced queries.
        """
        expr = Expression.from_query(**queries)
        source = Parquet.filter(self.source, expr.to_arrow())
        return self.scan(info, filter=expr) if source is None else type(self)(source)

    @doc_field
    def type(self) -> str:
        """[arrow type](https://arrow.apache.org/docs/python/api/dataset.html#classes)"""
        return type(self.source).__name__

    @doc_field
    def schema(self) -> Schema:
        """dataset schema"""
        source = self.source
        schema = source.schema() if isinstance(source, ibis.Table) else source.schema
        partitioning = getattr(source, 'partitioning', None)
        return Schema(
            names=schema.names,
            types=schema.types,
            partitioning=partitioning.schema.names if partitioning else [],
        )  # type: ignore

    @doc_field
    def optional(self) -> Self | None:
        """Nullable field to stop error propagation, enabling partial query results.

        Will be replaced by client controlled nullability.
        """
        return self

    @doc_field
    def count(self) -> Long:
        """number of rows"""
        if isinstance(self.source, ibis.Table):
            return self.source.count().to_pyarrow().as_py()
        return len(self.source) if isinstance(self.source, Sized) else self.source.count_rows()

    @doc_field
    def any(self, info: Info, length: Long = 1) -> bool:
        """Return whether there are at least `length` rows.

        May be significantly faster than `length` for out-of-core data.
        """
        table = self.to_table(info, length)
        return len(table) >= length

    @doc_field(
        name="column name(s); multiple names access nested struct fields",
        cast=f"cast array to {links.type}",
        safe="check for conversion errors on cast",
    )
    def column(
        self, info: Info, name: list[str], cast: str = '', safe: bool = True
    ) -> Column | None:
        """Return column of any type by name.

        This is typically only needed for aliased or casted columns.
        If the column is in the schema, `columns` can be used instead.
        """
        if isinstance(self.source, pa.Table) and len(name) == 1:
            column = self.source.column(*name)
            return Column.cast(column.cast(cast, safe) if cast else column)
        column = Projection(alias='_', name=name, cast=cast, safe=safe)  # type: ignore
        source = self.scan(info, Expression(), [column]).source
        return Column.cast(*(source if isinstance(source, pa.Table) else source.to_table()))

    @doc_field
    def cache(self, info: Info) -> Self:
        """Evaluate and cache the table."""
        return type(self)(self.to_table(info))

    @doc_field(
        offset="number of rows to skip; negative value skips from the end",
        limit="maximum number of rows to return",
    )
    def slice(self, info: Info, offset: Long = 0, limit: Long | None = None) -> Self:
        """Return zero-copy slice of table."""
        table = self.source
        if not isinstance(self.source, (ibis.Table, pa.Table)):
            table = self.to_table(info, limit and (offset + limit if offset >= 0 else None))
        return type(self)(table[offset:][:limit])

    @doc_field(
        by="column names; empty will aggregate into a single row table",
        counts="optionally include counts in an aliased column",
        row_number="optionally include first row number in an aliased column",
        aggregate="aggregation functions applied to other columns",
    )
    def group(
        self,
        info: Info,
        by: list[str] = [],
        counts: str = '',
        row_number: str = '',
        aggregate: Aggregates = {},  # type: ignore
    ) -> Self:
        """Return table grouped by columns.

        See `column` for accessing any column which has changed type.
        """
        aggs = dict(aggregate)  # type: ignore
        if not aggs and by == Parquet.keys(self.source, *by):
            return type(self)(Parquet.group(self.source, *by, counts=counts))
        if counts:
            aggs[counts] = ibis._.count()
        table = self.to_ibis(info)
        if row_number:
            table = table.mutate({row_number: ibis.row_number()})
            aggs[row_number] = ibis._[row_number].first()
        return type(self)(table.aggregate(aggs, by=by).cache())

    @doc_field(
        by="column names; prefix with `-` for descending order",
        limit="maximum number of rows to return; optimized for partitioned dataset keys",
    )
    def order(self, info: Info, by: list[str], limit: Long | None = None) -> Self:
        """Return table sorted by specified columns."""
        source = self.source
        if isinstance(source, ds.Dataset) and limit is not None:
            expr, by = T.rank_keys(self.source, limit, *by, dense=False)
            if expr is not None:
                self = type(self)(self.source.filter(expr))
            source = self.select(info)
            if not by:
                return type(self)(source.head(limit))
        if not isinstance(source, ibis.Table):  # pragma: no branch
            source = ibis.memtable(self.to_table(info))
        return type(self)(source.order_by(*map(order_key, by))[:limit])

    @doc_field(
        by="column names; prefix with `-` for descending order",
        max="maximum dense rank to select; optimized for == 1 (min or max)",
    )
    def rank(self, info: Info, by: list[str], max: int = 1) -> Self:
        """Return table selected by maximum dense rank."""
        source = self.to_table(info) if isinstance(self.source, ibis.Table) else self.source
        expr, by = T.rank_keys(source, max, *by)
        if expr is not None:
            source = source.filter(expr)
        return type(self)(T.rank(source, max, *by) if by else source)

    @doc_field
    def unnest(
        self,
        info: Info,
        name: str,
        offset: str = '',
        keep_empty: bool = False,
        row_number: str = '',
    ) -> Self:
        """[Unnest](https://ibis-project.org/reference/expression-tables#ibis.expr.types.relations.Table.unnest) an array column from a table."""
        table = self.to_ibis(info)
        if row_number:
            table = table.mutate({row_number: ibis.row_number()})
        return type(self)(table.unnest(name, offset=offset or None, keep_empty=keep_empty))

    @doc_field(filter="selected rows", columns="projected columns")
    def scan(self, info: Info, filter: Expression = {}, columns: list[Projection] = []) -> Self:  # type: ignore
        """Select rows and project columns without memory usage."""
        expr = filter.to_arrow()
        projection = {name: pc.field(name) for name in self.references(info, level=1)}
        projection |= {col.alias or '.'.join(col.name): col.to_arrow() for col in columns}
        if '' in projection:
            raise ValueError(f"projected columns need a name or alias: {projection['']}")
        source = self.source.to_pyarrow() if isinstance(self.source, ibis.Table) else self.source
        if expr is not None:
            source = source.filter(expr)
        if columns or isinstance(source, ds.Dataset):
            source = Nodes.scan(source, projection)
        return type(self)(source)

    @doc_field(
        right="name of right table; must be on root Query type",
        keys="column names used as keys on the left side",
        rkeys="column names used as keys on the right side; defaults to left side",
        how="the kind of join: 'inner', 'left', 'right', ...",
        lname="format string to use to rename overlapping columns in the left table",
        rname="format string to use to rename overlapping columns in the right table",
    )
    def join(
        self,
        info: Info,
        right: str,
        keys: list[str],
        rkeys: list[str] = [],
        how: str = 'inner',
        lname: str = '',
        rname: str = '{name}_right',
    ) -> Self:
        """Perform a [join](https://ibis-project.org/reference/expression-tables#ibis.expr.types.relations.Table.join) between two tables."""
        left = self.to_ibis(info)
        right = getattr(info.root_value, right).to_ibis(info)
        if rkeys:
            keys = [getattr(left, key) == getattr(right, rkey) for key, rkey in zip(keys, rkeys)]
        return type(self)(left.join(right, predicates=keys, how=how, lname=lname, rname=rname))

    @doc_field
    def take(self, info: Info, indices: list[Long]) -> Self:
        """Select rows from indices."""
        source = self.select(info)
        if isinstance(source, ibis.Table):  # pragma: no branch
            source = ds.Scanner.from_batches(source.to_pyarrow_batches())
        return type(self)(source.take(indices))

    @doc_field
    def drop_null(self, info: Info) -> Self:
        """Remove missing values from referenced columns in the table."""
        table = self.to_ibis(info)
        return type(self)(table.drop_null())

    @doc_field
    def project(self, info: Info, columns: list[IProjection]) -> Self:
        """Apply functions to columns.

        Equivalent to [mutate](https://ibis-project.org/reference/expression-tables#ibis.expr.types.relations.Table.mutate);
        renamed to not be confused with a mutation.
        """
        table = self.to_ibis(info)
        projection = {column.alias or column.name: column.to_ibis() for column in columns}
        return type(self)(table.mutate(projection))
