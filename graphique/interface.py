"""
Primary Dataset interface.

Doesn't require knowledge of the schema.
"""

import itertools
import operator
from collections.abc import Iterable, Iterator, Mapping
from typing import TypeAlias, no_type_check
import ibis
import pyarrow.dataset as ds
import strawberry.asgi
from strawberry import Info, UNSET
from strawberry.scalars import JSON
from typing_extensions import Self
from .core import Parquet, order_key
from .inputs import Aggregates, Field, Filter, Expression, Projection, Scalars
from .models import Column, doc_field, links, selections
from .core import getitems
from .scalars import BigInt

Source: TypeAlias = ibis.Table | ds.Dataset


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


@strawberry.interface(description="ibis `Table` or arrow `Dataset`")
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
            if names := self.select(info, source, level=1):
                source = source.select(*names).cache()
        return type(self)(source)

    @classmethod
    @no_type_check
    def resolve_reference(cls, info: Info, **keys) -> Self:
        """Return table filtered by federated keys."""
        self = getattr(info.root_value, cls.field)
        queries = {name: Filter(eq=[keys[name]]) for name in keys}
        return self.filter(info, **queries)

    @staticmethod
    def select(info: Info, source: Source, level: int = 0) -> list:
        """Return minimal schema needed to continue."""
        fields = info.selected_fields
        for _ in range(level):
            fields = itertools.chain(*[field.selections for field in fields])
        refs = set(itertools.chain(*map(references, fields)))
        return [name for name in ibis_schema(source).names if name in refs]

    def columns(self, info: Info) -> dict:
        """Fields for each column."""
        names = selections(*info.selected_fields)
        table = self.table.select(*names).cache()
        return {name: Column.cast(table[name]) for name in table.columns}

    def row(self, info: Info, index: int = 0) -> dict:
        """Scalar values at index."""
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
        description=f"[ibis table]({links.ref}/expression-table) or [arrow dataset](https://arrow.apache.org/docs/python/api/dataset.html)"
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

    @strawberry.field(
        description=f"[Count]({links.ref}/expression-tables#ibis.expr.types.relations.Table.count) the number of rows."
    )
    def count(self) -> BigInt:
        if isinstance(self.source, ibis.Table):
            return self.source.count().to_pyarrow().as_py()
        return self.source.count_rows()

    @doc_field
    def any(self, limit: BigInt = 1) -> bool:
        """Whether there are at least `limit` rows.

        May be significantly faster than `count` for out-of-core data.
        """
        return self.table[:limit].count().to_pyarrow().as_py() >= limit

    @doc_field(
        name="column name(s); multiple names access nested struct fields",
        cast=f"cast expression to indicated {links.types}",
        try_="return null if cast fails",
    )
    def column(self, name: list[str], cast: str = '', try_: bool = False) -> Column | None:
        """Column of any type by name.

        If the column is in the schema, `columns` can be used instead.
        """
        column = getitems(self.table, *name)
        if cast:
            column = (column.try_cast if try_ else column.cast)(cast)
        return Column.cast(column.as_table().cache()[0])

    @doc_field(
        offset="number of rows to skip; negative value skips from the end",
        limit="maximum number of rows to return",
    )
    def slice(self, info: Info, offset: BigInt = 0, limit: BigInt | None = None) -> Self:
        """[Limit](https://ibis-project.org/reference/expression-tables#ibis.expr.types.relations.Table.limit) row selection."""
        return self.resolve(info, self.table[offset:][:limit])

    @doc_field(
        on="column names to deduplicate on; defaults to all",
        keep="which duplicates to keep",
        counts=f"[value counts]({links.ref}/expression-tables#ibis.expr.types.relations.Table.value_counts); incompatible with `keep: null`",
        row_number="optionally include row number in an aliased column",
    )
    def distinct(
        self,
        info: Info,
        on: list[str] | None = None,
        keep: str | None = 'first',
        counts: str = '',
        row_number: str = '',
    ) -> Self:
        """[Remove duplicate](https://ibis-project.org/reference/expression-tables#ibis.expr.types.relations.Table.distinct) rows from table.

        Differs from `group` by keeping all columns, and defaulting to all keys.
        """
        table = self.table
        if row_number:
            table = table.mutate({row_number: ibis.row_number()})
        if not counts or keep is None:
            return self.resolve(info, table.distinct(on=on, keep=keep))
        if on is None:
            return self.resolve(info, table.value_counts(name=counts))
        keys, func = set(on), operator.methodcaller(keep)
        aggs = {name: func(table[name]) for name in table.columns if name not in keys}
        aggs[counts] = table.count()
        return self.resolve(info, table.aggregate(aggs, by=on))

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
        """[Group](https://ibis-project.org/reference/expression-tables#ibis.expr.types.relations.Table.group_by) table by columns."""
        aggs = dict(aggregate)  # type: ignore
        if not aggs and not row_number and by == Parquet.keys(self.source, *by):
            return self.resolve(info, Parquet.group(self.source, *by, counts=counts))
        table = self.table
        if counts:
            aggs[counts] = table.count()
        if row_number:
            table = table.mutate({row_number: ibis.row_number()})
            aggs[row_number] = table[row_number].first()
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
            groups = table.aggregate(_=table.count(), by=[name.lstrip('-') for name in by])
            limit = groups.order_by(*map(order_key, by))[:limit]['_'].sum().to_pyarrow().as_py()
        return self.resolve(info, table[:limit])

    @doc_field(
        name="column name",
        offset="optionally include index column",
        keep_empty="keep empty array values as null",
        row_number="optionally include first row number in an aliased column",
    )
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
        names = self.select(info, self.source)
        if isinstance(self.source, ds.Dataset):
            table = self.source.take(indices, columns=names)
        else:
            batches = self.source.select(*names).to_pyarrow_batches()
            table = ds.Scanner.from_batches(batches).take(indices)
        return type(self)(ibis.memtable(table))

    @doc_field(subset="columns names; defaults to all", how="remove if `any` or `all` are null")
    def drop_null(self, info: Info, subset: list[str] | None = None, how: str = 'any') -> Self:
        """[Drop](https://ibis-project.org/reference/expression-tables#ibis.expr.types.relations.Table.drop_null) rows with null values."""
        return self.resolve(info, self.table.drop_null(subset, how=how))

    @doc_field(name="column name(s); defaults to all", value="JSON scalar", scalar="typed scalar")
    def fill_null(
        self,
        info: Info,
        name: list[str] | None = None,
        value: JSON | None = UNSET,
        scalar: Scalars = {},  # type: ignore
    ) -> Self:
        """[Fill null](https://ibis-project.org/reference/expression-tables.html#ibis.expr.types.relations.Table.fill_null) values."""
        (value,) = itertools.chain(scalar, [] if value is UNSET else [value])  # type: ignore
        replacements = dict.fromkeys(ibis_schema(self.source) if name is None else name, value)
        return self.resolve(info, self.table.fill_null(replacements))

    @doc_field
    def project(self, info: Info, columns: list[Projection]) -> Self:
        """[Mutate](https://ibis-project.org/reference/expression-tables#ibis.expr.types.relations.Table.mutate) columns by expressions.

        Renamed to not be confused with a mutation.
        """
        projection = {column.alias or ''.join(column.name): column.to_ibis() for column in columns}
        if '' in projection:
            raise ValueError(f"projected fields require a name or alias: {projection['']}")
        return self.resolve(info, self.table.mutate(projection))
