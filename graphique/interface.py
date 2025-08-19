"""
Primary Dataset interface.

Doesn't require knowledge of the schema.
"""

import itertools
from collections.abc import Iterable, Iterator, Mapping
from typing import TypeAlias, no_type_check
import ibis
import pyarrow.dataset as ds
import strawberry.asgi
from strawberry import Info
from typing_extensions import Self
from .core import Parquet, order_key
from .inputs import Aggregates, Field, Filter, Expression, Projection
from .models import Column, doc_field, links, selections
from .core import getitems
from .scalars import BigInt

Source: TypeAlias = ds.Dataset | ibis.Table


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


@strawberry.type(description=links.schema)
class Schema:
    names: list[str] = strawberry.field(description="field names")
    types: list[str] = strawberry.field(description=f"{links.types}, corresponding to `names`")
    partitioning: list[str] = strawberry.field(description="partition keys")


def ibis_schema(root: Source) -> ibis.Schema:
    return root.schema() if isinstance(root, ibis.Table) else ibis.Schema.from_pyarrow(root.schema)


@strawberry.interface(description="arrow `Dataset` or ibis `Table`")
class Dataset:
    def __init__(self, source: Source):
        self.source = source

    @property
    def table(self) -> ibis.Table:
        """source as ibis table"""
        return self.source if isinstance(self.source, ibis.Table) else Parquet.to_table(self.source)

    def resolve(self, info: Info, source: ibis.Table) -> Self:
        """Cache the table if it will be reused."""
        count = sum(len(field.selections) for field in info.selected_fields)
        if count > 1 and isinstance(source, ibis.Table):
            if names := self.references(info, level=1):
                source = source.select(*names).cache()
        return type(self)(source)

    @classmethod
    @no_type_check
    def resolve_reference(cls, info: Info, **keys) -> Self:
        """Return table filtered by federated keys."""
        self = getattr(info.root_value, cls.field)
        queries = {name: Filter(eq=[keys[name]]) for name in keys}
        return self.filter(info, **queries)

    def references(self, info: Info, level: int = 0) -> set:
        """Return set of every possible future column reference."""
        fields = info.selected_fields
        for _ in range(level):
            fields = itertools.chain(*[field.selections for field in fields])
        return set(itertools.chain(*map(references, fields))) & set(self.schema().names)

    def columns(self, info: Info) -> dict:
        """fields for each column"""
        names = selections(*info.selected_fields)
        projection = {} if names else {'_': ibis.row_number()}
        table = self.table.select(*names, **projection)
        if len(names) > 1:
            table = table.cache()
        return {name: Column.cast(table[name]) for name in table.columns}

    def row(self, info: Info, index: int = 0) -> dict:
        """Return scalar values at index."""
        names = selections(*info.selected_fields)
        table = self.table.select(*names)[index:][:1].cache()
        row = {}
        for name in table.columns:
            if isinstance(table[name], ibis.expr.types.ArrayColumn):
                row[name] = Column.cast(table[name].first().unnest())
            else:
                (row[name],) = table[name].to_list()
        return row

    def filter(self, info: Info, where: Expression | None = None, **queries: Filter) -> Self:
        """[Filter](https://ibis-project.org/reference/expression-tables#ibis.expr.types.relations.Table.filter) rows by predicates.

        Schema derived fields provide syntax for simple queries; `where` supports complex queries.
        """
        exprs: list = [] if where is None else list(where)  # type: ignore
        source = Parquet.filter(self.source, Filter.to_arrow(**queries))
        if source is None:
            exprs += Filter.to_exprs(**queries)
            source = self.table
        elif exprs:
            source = Parquet.to_table(source)
        return self.resolve(info, source.filter(*exprs) if exprs else source)

    @strawberry.field(
        description=f"[arrow dataset](https://arrow.apache.org/docs/python/api/dataset.html) or [ibis table]({links.ref}/expression-table)"
    )
    def type(self) -> str:
        return type(self.source).__name__

    @strawberry.field(description=links.schema)
    def schema(self) -> Schema:
        schema = ibis_schema(self.source)
        partitioning = Parquet.schema(self.source).names
        return Schema(names=schema.names, types=schema.types, partitioning=partitioning)  # type: ignore

    @doc_field(
        schema="field names and types",
        try_="return null if cast fails",
    )
    def cast(self, info: Info, schema: list[Field], try_: bool = False) -> Self:
        """[Cast](https://ibis-project.org/reference/expression-tables#ibis.expr.types.relations.Table.cast) the columns of a table."""
        cast = self.table.try_cast if try_ else self.table.cast
        return self.resolve(info, cast({field.name: field.type for field in schema}))

    @doc_field
    def optional(self, info: Info) -> Self | None:
        """Nullable field to stop error propagation, enabling partial query results.

        Will be replaced by client controlled nullability.
        """
        return self.resolve(info, self.source)

    @doc_field
    def count(self) -> BigInt:
        """number of rows"""
        if isinstance(self.source, ibis.Table):
            return self.source.count().to_pyarrow().as_py()
        return self.source.count_rows()

    @doc_field
    def any(self, info: Info, limit: BigInt = 1) -> bool:
        """Return whether there are at least `limit` rows.

        May be significantly faster than `count` for out-of-core data.
        """
        return self.table[:limit].count().to_pyarrow().as_py() >= limit

    @doc_field(
        name="column name(s); multiple names access nested struct fields",
        cast=f"cast expression to indicated {links.types}",
        try_="return null if cast fails",
    )
    def column(
        self, info: Info, name: list[str], cast: str = '', try_: bool = False
    ) -> Column | None:
        """Return column of any type by name.

        This is typically only needed for aliased or casted columns.
        If the column is in the schema, `columns` can be used instead.
        """
        column = getitems(self.table, *name)
        if cast:
            column = (column.try_cast if try_ else column.cast)(cast)
        return Column.cast(column)

    @doc_field(
        offset="number of rows to skip; negative value skips from the end",
        limit="maximum number of rows to return",
    )
    def slice(self, info: Info, offset: BigInt = 0, limit: BigInt | None = None) -> Self:
        """[Limit](https://ibis-project.org/reference/expression-tables#ibis.expr.types.relations.Table.limit) row selection."""
        return self.resolve(info, self.table[offset:][:limit])

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
        """[Group](https://ibis-project.org/reference/expression-tables#ibis.expr.types.relations.Table.group_by) table by columns.

        See `column` for accessing any column which has changed type.
        """
        aggs = dict(aggregate)  # type: ignore
        if not aggs and by == Parquet.keys(self.source, *by):
            return self.resolve(info, Parquet.group(self.source, *by, counts=counts))
        if counts:
            aggs[counts] = ibis._.count()
        table = self.table
        if row_number:
            table = table.mutate({row_number: ibis.row_number()})
            aggs[row_number] = ibis._[row_number].first()
        return self.resolve(info, table.aggregate(aggs, by=by))

    @doc_field(
        by="column names; prefix with `-` for descending order",
        limit="maximum number of rows to return; optimized for partitioned dataset keys",
        dense="use dense rank with `limit`",
    )
    def order(
        self, info: Info, by: list[str], limit: BigInt | None = None, dense: bool = False
    ) -> Self:
        """[Sort](https://ibis-project.org/reference/expression-tables#ibis.expr.types.relations.Table.order_by) table by columns."""
        keys = Parquet.keys(self.source, *by)
        if keys and limit is not None:
            table = Parquet.rank(self.source, limit, *keys, dense=dense)
        else:
            table = self.table
        table = table.order_by(*map(order_key, by))
        if dense:
            groups = table.aggregate(_=ibis._.count(), by=[name.lstrip('-') for name in by])
            limit = groups.order_by(*map(order_key, by))[:limit]['_'].sum().to_pyarrow().as_py()
        return self.resolve(info, table[:limit])

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
        table = self.table
        if row_number:
            table = table.mutate({row_number: ibis.row_number()})
        return self.resolve(info, table.unnest(name, offset=offset or None, keep_empty=keep_empty))

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
        """[Join](https://ibis-project.org/reference/expression-tables#ibis.expr.types.relations.Table.join) two tables."""
        left = self.table
        right = getattr(info.root_value, right).table
        if rkeys:
            keys = [getattr(left, key) == getattr(right, rkey) for key, rkey in zip(keys, rkeys)]
        return self.resolve(
            info, left.join(right, predicates=keys, how=how, lname=lname, rname=rname)
        )

    @doc_field
    def take(self, info: Info, indices: list[BigInt]) -> Self:
        """[Take](https://arrow.apache.org/docs/python/generated/pyarrow.dataset.Dataset.html#pyarrow.dataset.Dataset.take) rows by index."""
        names = self.references(info)
        if isinstance(self.source, ds.Dataset):
            table = self.source.take(indices, columns=list(names))
        else:
            batches = self.source.select(*names).to_pyarrow_batches()
            table = ds.Scanner.from_batches(batches).take(indices)
        return type(self)(ibis.memtable(table))

    @doc_field
    def drop_null(self, info: Info, subset: list[str] | None = None, how: str = 'any') -> Self:
        """[Drop](https://ibis-project.org/reference/expression-tables#ibis.expr.types.relations.Table.drop_null) rows with null values."""
        return self.resolve(info, self.table.drop_null(subset, how=how))

    @doc_field
    def project(self, info: Info, columns: list[Projection]) -> Self:
        """[Mutate](https://ibis-project.org/reference/expression-tables#ibis.expr.types.relations.Table.mutate) columns by expressions.

        Renamed to not be confused with a mutation.
        """
        projection = {column.alias or ''.join(column.name): column.to_ibis() for column in columns}
        if '' in projection:
            raise ValueError(f"projected fields require a name or alias: {projection['']}")
        return self.resolve(info, self.table.mutate(projection))
