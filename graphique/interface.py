"""
Primary Dataset interface.

Doesn't require knowledge of the schema.
"""

import collections
import itertools
import operator
from collections.abc import Iterable, Iterator, Mapping
from typing import Self

import ibis.expr.operations
import ibis.expr.types
import pyarrow as pa
import pyarrow.dataset as ds
import strawberry
from sqlglot import exp, parse_one
from strawberry import UNSET, Info
from strawberry.permission import BasePermission, PermissionExtension
from strawberry.scalars import JSON

from .inputs import Aggregates, Expression, Field, Filter, Projection, Scalars
from .models import Column, doc_field
from .scalars import BigInt, getitems, links, order_key


def selections(*fields) -> collections.Counter:
    """Return field name selections from strawberry `SelectedField`."""
    counts = collections.Counter()
    for field in fields:
        for selection in field.selections:
            if hasattr(selection, "name"):
                counts[selection.name] += 1
            else:
                counts.update(selections(selection))
    return counts


def references(field) -> Iterator:
    """Generate every possible column reference from strawberry `SelectedField`."""
    if isinstance(field, str):
        yield field.lstrip("-")
    elif isinstance(field, Iterable):
        for value in field:
            yield from references(value)
        if isinstance(field, Mapping):
            for value in field.values():
                yield from references(value)
    else:
        for name in ("name", "arguments", "selections"):
            yield from references(getattr(field, name, []))


def cached(table: ibis.Table) -> ibis.expr.types.CachedTable:
    """`Table.cache`, or raise with advice to use `materialize`."""
    try:
        return table.cache()
    except Exception as exc:
        raise TypeError("see `materialize` for client-side caching") from exc


class Deny(BasePermission):
    message = "field is denied by default"

    def has_permission(self, source, info: Info, **kwargs) -> bool:
        return False


@strawberry.type(
    description=f"[table schema]({links.ref}/schemas#ibis.expr.schema.Schema) and info"
)
class Schema:
    type: str = strawberry.field(description=f"[table]({links.ref}/expression-tables) type")
    backend: str = strawberry.field(
        description=f"[backend name]({links.ref}/expression-tables#ibis.expr.types.relations.Table.get_backend)"
    )
    names: list[str] = strawberry.field(description="field names")
    types: list[str] = strawberry.field(description=f"{links.types}, corresponding to `names`")


@strawberry.interface(description="ibis `Table`")
class Dataset:
    table: strawberry.Private[ibis.Table]

    def resolve(self, info: Info, table: ibis.Table) -> Self:
        """Cache the table if it will be reused."""
        if not isinstance(table, ibis.expr.types.CachedTable):
            counts = selections(*info.selected_fields)
            counts["schema"] = counts["toSql"] = counts["__typename"] = 0
            if counts.total() > 1:
                names = self.select(info, table) or [ibis.row_number()]
                table = cached(table.select(*names))
        return type(self)(table=table)

    @classmethod
    def resolve_reference(cls, info: Info, **keys) -> Self:
        """Return table filtered by federated keys."""
        self = getattr(info.root_value, cls.field)
        return self.filter(info, **{name: Filter(eq=[keys[name]]) for name in keys})

    @staticmethod
    def select(info: Info, table: ibis.Table) -> list:
        """Return minimal schema needed to continue."""
        refs: set = set()
        for field in info.selected_fields:
            for selection in field.selections:
                refs.update(references(selection))
        return [name for name in table.schema() if name in refs]

    @doc_field(
        dialect="output SQL dialect; defaults to the backend’s native dialect",
        pretty="whether to use pretty formatting",
    )
    def to_sql(self, dialect: str | None = None, pretty: bool = True) -> str:
        """[Compile to](https://ibis-project.org/reference/expression-tables#ibis.expr.types.relations.Table.to_sql) a formatted SQL string."""
        return self.table.to_sql(dialect=dialect, pretty=pretty)

    @strawberry.field(
        description=f"[table schema]({links.ref}/expression-tables#ibis.expr.types.relations.Table.schema)"
    )
    def schema(self) -> Schema:
        schema = self.table.schema()
        return Schema(
            type=type(self.table).__name__,
            backend=self.table.get_backend().name,
            names=schema.names,
            types=schema.types,
        )

    @doc_field(schema="field names and types", try_="return null if cast fails")
    def cast(self, info: Info, schema: list[Field], try_: bool = False) -> Self:
        """[Cast](https://ibis-project.org/reference/expression-tables#ibis.expr.types.relations.Table.cast) the columns of a table."""
        cast = self.table.try_cast if try_ else self.table.cast
        return self.resolve(info, cast({field.name: field.type for field in schema}))

    @doc_field
    def optional(self, info: Info) -> Self | None:
        """Nullable field to stop error propagation, enabling partial query results."""
        return self.resolve(info, self.table)

    @doc_field
    def materialize(self, info: Info) -> Self:
        """[Materialize](https://ibis-project.org/reference/expression-tables.html#ibis.memtable) an in-memory table.

        An alternative to implicit [table caching](https://ibis-project.org/reference/expression-tables.html#ibis.expr.types.relations.Table.cache), which may fail or only cache server side.
        """
        names = self.select(info, self.table) or [ibis.row_number()]
        return type(self)(table=ibis.memtable(self.table.select(*names).to_pyarrow()))

    @strawberry.field(
        description=f"[Count]({links.ref}/expression-tables#ibis.expr.types.relations.Table.count) the number of rows."
    )
    def count(self) -> BigInt:
        return self.table.count().to_pyarrow().as_py()

    @doc_field
    def any(self, limit: BigInt = BigInt(1)) -> bool:
        """Whether there are at least `limit` rows.

        May be significantly faster than `count` for out-of-core data.
        """
        return self.table[:limit].count().to_pyarrow().as_py() >= limit

    @doc_field(
        name="column name(s); multiple names access nested struct fields",
        cast=f"cast expression to indicated {links.types}",
        try_="return null if cast fails",
        index="column index(es); may access array offsets",
    )
    def column(
        self,
        info: Info,
        name: list[str],
        cast: str = "",
        try_: bool = False,
        index: list[BigInt] = [],
    ) -> Column | None:
        """Column of any type by name.

        If the column is in the schema, `columns` can be used instead.
        """
        column = getitems(self.table, *(name + index))
        if cast:
            column = (column.try_cast if try_ else column.cast)(cast)
        counts = selections(*info.selected_fields)
        if counts.total() > 1 and not isinstance(self.table, ibis.expr.types.CachedTable):
            column = cached(column.as_table())[0]
        return Column.cast(column)

    @doc_field(
        offset="number of rows to skip; negative value skips from the end",
        limit="maximum number of rows to return",
    )
    def slice(self, info: Info, offset: BigInt = BigInt(0), limit: BigInt | None = None) -> Self:
        """[Limit](https://ibis-project.org/reference/expression-tables#ibis.expr.types.relations.Table.limit) row selection."""
        return self.resolve(info, self.table[offset:][:limit])

    @doc_field(
        on="column names to deduplicate on; defaults to all remaining columns",
        keep="which duplicates to keep",
        counts=f"[value counts]({links.ref}/expression-tables#ibis.expr.types.relations.Table.value_counts); incompatible with `keep: null`",
        order="optionally include and order by first row number; incompatible with `on: null`",
    )
    def distinct(
        self,
        info: Info,
        on: list[str] | None = None,
        keep: str | None = "first",
        counts: str = "",
        order: str = "",
    ) -> Self:
        """[Remove duplicate](https://ibis-project.org/reference/expression-tables#ibis.expr.types.relations.Table.distinct) rows from table.

        Differs from `group` by keeping all columns, and defaulting to all remaining keys. May use ibis `value_counts` or `aggregate`.
        Note this can have varying behavior given automatic column selection; specify `on` as needed.
        """
        table = self.table
        if on:
            table = table.select(*dict.fromkeys(on + self.select(info, table)))
        if order:
            table = table.mutate({order: ibis.row_number()})
        if not counts or keep is None:
            table = table.distinct(on=on, keep=keep)
        elif on is None:
            table = table.value_counts(name=counts)
        else:
            keys, func = set(on), operator.methodcaller(keep)
            aggs = {name: func(table[name]) for name in table.columns if name not in keys}
            aggs[counts] = ibis._.count()
            table = table.aggregate(aggs, by=on)
        return self.resolve(info, table.order_by(order) if order else table)

    @doc_field(
        by="column names; empty will aggregate into a single row table",
        counts="optionally include counts in an aliased column",
        order="optionally include and order by first row number",
        aggregate="aggregation functions applied to other columns",
    )
    def group(
        self,
        info: Info,
        by: list[str] = [],
        counts: str = "",
        order: str = "",
        aggregate: Aggregates = Aggregates(),
    ) -> Self:
        """[Group](https://ibis-project.org/reference/expression-tables#ibis.expr.types.relations.Table.group_by) table by columns."""
        aggs = dict(aggregate)
        table = self.table
        if counts:
            aggs[counts] = table[counts].sum() if counts in table else ibis._.count()
        if order:
            table = table.mutate({order: ibis.row_number()})
            aggs[order] = table[order].min()
        table = table.aggregate(aggs, by=by)
        return self.resolve(info, table.order_by(order) if order else table)

    @doc_field(
        by="column names; prefix with `-` for descending order",
        limit="maximum number of rows to return",
        over="column names; `limit` applies over separate grouping windows, which may be significantly slower than `group`",
    )
    def order(
        self,
        info: Info,
        by: list[str],
        limit: BigInt | None = None,
        over: list[str] = [],
    ) -> Self:
        """[Sort](https://ibis-project.org/reference/expression-tables#ibis.expr.types.relations.Table.order_by) table by columns."""
        if over and limit is not None:
            window = ibis.row_number().over(group_by=over, order_by=map(order_key, by))
            return self.resolve(info, self.table.filter(window < limit))
        return self.resolve(info, self.table.order_by(*map(order_key, by))[:limit])

    @doc_field(
        by="column names; prefix with `-` for descending order",
        rank="maximum 1-based rank of rows to return",
        dense="use dense rank (all ties) instead of sparse rank (only ties at the boundary)",
        over="column names; `rank` applies over separate grouping windows, which may be significantly slower than `group`",
    )
    def first(
        self,
        info: Info,
        by: list[str],
        rank: int = 1,
        dense: bool = False,
        over: list[str] = [],
    ) -> Self:
        """Sort and filter by maximum 1-based rank.

        Differs from `order` with `limit` by including ties at the boundary. Not directly supported in ibis; requires a semi-join which may be slower.
        """
        if over:
            index = ibis.dense_rank() if dense else ibis.rank()
            window = index.over(group_by=over, order_by=map(order_key, by))
            return self.resolve(info, self.table.filter(window < rank))
        order_by = {name.lstrip("-"): order_key(name) for name in by}
        mask = self.table.select(*order_by)
        if dense and rank > 1:
            mask = mask.distinct()
        table = self.table.semi_join(mask.order_by(*order_by.values())[:rank], list(order_by))
        return self.resolve(info, table.order_by(*order_by.values()))

    @doc_field(
        name="column name",
        offset="optionally include index column",
        keep_empty="keep empty array values as null",
        order="optionally include and order by row number; optionally by `offset` too",
    )
    def unnest(
        self,
        info: Info,
        name: str,
        offset: str = "",
        keep_empty: bool = False,
        order: str = "",
    ) -> Self:
        """[Unnest](https://ibis-project.org/reference/expression-tables#ibis.expr.types.relations.Table.unnest) an array column from a table."""
        table = self.table.mutate({order: ibis.row_number()}) if order else self.table
        table = table.unnest(name, offset=offset or None, keep_empty=keep_empty)
        if order:
            table = table.order_by(order, offset) if offset else table.order_by(order)
        return self.resolve(info, table)

    @doc_field(names="column names")
    def unpack(self, info: Info, names: list[str]) -> Self:
        """[Unpack](https://ibis-project.org/reference/expression-tables#ibis.expr.types.relations.Table.unpack) the struct fields of each column."""
        return self.resolve(info, self.table.unpack(*names))

    @doc_field(
        right="name of right table; must be on root Query type",
        keys="column names used as keys on the left side",
        rkeys="column names used as keys on the right side; defaults to left side",
        how="the kind of join: 'inner', 'left', 'outer', 'semi', ...",
        lname="format string to use to rename overlapping columns in the left table",
        rname="format string to use to rename overlapping columns in the right table",
    )
    def join(
        self,
        info: Info,
        right: str,
        keys: list[str],
        rkeys: list[str] = [],
        how: str = "inner",
        lname: str = "",
        rname: str = "{name}_right",
    ) -> Self:
        """[Join](https://ibis-project.org/reference/expression-tables#ibis.expr.types.relations.Table.join) two tables."""
        left = self.table
        right = getattr(info.root_value, right).table
        if rkeys:
            keys = [getattr(left, key) == getattr(right, rkey) for key, rkey in zip(keys, rkeys)]
        return self.resolve(info, left.join(right, keys, how=how, lname=lname, rname=rname))

    @doc_field(
        right="name of right table; must be on root Query type",
        on="column name for closest-match inequality (e.g. timestamp)",
        keys="column names used as keys on the left side",
        rkeys="column names used as keys on the right side; defaults to left side",
        tolerance="optional look-back window for the as-of match",
        scalar="typed scalar for `tolerance",
        lname="format string to rename overlapping columns in the left table",
        rname="format string to rename overlapping columns in the right table",
    )
    def asof_join(
        self,
        info: Info,
        right: str,
        on: str,
        keys: list[str] = [],
        rkeys: list[str] = [],
        tolerance: JSON | None = None,
        scalar: Scalars = Scalars(),
        lname: str = "",
        rname: str = "{name}_right",
    ) -> Self:
        """[As-of join](https://ibis-project.org/reference/expression-tables#ibis.expr.types.relations.Table.asof_join) on nearest key rather than equal keys."""
        left = self.table
        right = getattr(info.root_value, right).table.alias("right")
        if rkeys:
            keys = [getattr(left, key) == getattr(right, rkey) for key, rkey in zip(keys, rkeys)]
        (tolerance,) = list(scalar) or [tolerance]
        table = left.asof_join(right, on, keys, tolerance=tolerance, lname=lname, rname=rname)
        return self.resolve(info, table)

    @doc_field(
        right="name(s) of right table; must be on root Query type",
        lname="format string to rename overlapping columns in the left table",
        rname="format string to rename overlapping columns in the right table",
    )
    def cross_join(
        self, info: Info, right: list[str], lname: str = "", rname: str = "{name}_right"
    ) -> Self:
        """[Cross join](https://ibis-project.org/reference/expression-tables#ibis.expr.types.relations.Table.cross_join) with one or more tables."""
        tables = [getattr(info.root_value, name).table for name in right]
        return self.resolve(info, self.table.cross_join(*tables, lname=lname, rname=rname))

    @doc_field(
        table="name(s) of other table; must be on root Query type",
        distinct="use set difference (true) or multiset difference (false)",
    )
    def difference(self, info: Info, table: list[str], distinct: bool = True) -> Self:
        """[Set difference](https://ibis-project.org/reference/expression-tables#ibis.expr.types.relations.Table.difference) of tables."""
        tables = [getattr(info.root_value, name).table for name in table]
        return self.resolve(info, self.table.difference(*tables, distinct=distinct))

    @doc_field(
        table="name(s) of other table; must be on root Query type",
        distinct="use set difference (true) or multiset difference (false)",
    )
    def intersect(self, info: Info, table: list[str], distinct: bool = True) -> Self:
        """[Set intersection](https://ibis-project.org/reference/expression-tables#ibis.expr.types.relations.Table.intersect) of tables."""
        tables = [getattr(info.root_value, name).table for name in table]
        return self.resolve(info, self.table.intersect(*tables, distinct=distinct))

    @doc_field(
        table="name(s) of other table; must be on root Query type",
        distinct="use set difference (true) or multiset difference (false)",
    )
    def union(self, info: Info, table: list[str], distinct: bool = False) -> Self:
        """[Set union](https://ibis-project.org/reference/expression-tables#ibis.expr.types.relations.Table.union) of tables."""
        tables = [getattr(info.root_value, name).table for name in table]
        return self.resolve(info, self.table.union(*tables, distinct=distinct))

    @doc_field
    def take(self, info: Info, indices: list[BigInt]) -> Self:
        """[Take](https://arrow.apache.org/docs/python/generated/pyarrow.dataset.Scanner.html#pyarrow.dataset.Scanner.take) rows by index.

        Not supported in ibis; uses arrow `take`.
        """
        names = self.select(info, self.table)
        if not names:
            table = pa.table({"_": indices})
        else:
            batches = self.table.select(*names).to_pyarrow_batches()
            table = ds.Scanner.from_batches(batches).take(indices)
        return type(self)(table=ibis.memtable(table))

    @doc_field(subset="columns names; defaults to all", how="remove if `any` or `all` are null")
    def drop_null(self, info: Info, subset: list[str] | None = None, how: str = "any") -> Self:
        """[Drop](https://ibis-project.org/reference/expression-tables#ibis.expr.types.relations.Table.drop_null) rows with null values.

        Note this can have varying behavior given automatic column selection.
        """
        return self.resolve(info, self.table.drop_null(subset, how=how))

    @doc_field(name="column name(s); defaults to all", value="JSON scalar", scalar="typed scalar")
    def fill_null(
        self,
        info: Info,
        name: list[str] | None = None,
        value: JSON | None = UNSET,
        scalar: Scalars = Scalars(),
    ) -> Self:
        """[Fill null](https://ibis-project.org/reference/expression-tables.html#ibis.expr.types.relations.Table.fill_null) values."""
        (value,) = itertools.chain(scalar, [] if value is UNSET else [value])
        replacements = dict.fromkeys(self.table.schema() if name is None else name, value)
        return self.resolve(info, self.table.fill_null(replacements))

    @doc_field
    def project(self, info: Info, columns: list[Projection]) -> Self:
        """[Mutate](https://ibis-project.org/reference/expression-tables#ibis.expr.types.relations.Table.mutate) columns by expressions.

        Renamed to not be confused with a mutation.
        """
        projection = dict(map(Projection.to_ibis, columns))
        return self.resolve(info, self.table.mutate(projection))

    @doc_field(
        by="column names to compare by equality",
        split="boolean column expressions to split on true values",
        counts="optionally include counts in an aliased column",
        order="alias of row number to maintain order",
        aggregate="aggregation functions applied to other columns",
    )
    def runs(
        self,
        info: Info,
        by: list[str] = [],
        split: list[Expression] = [],
        counts: str = "",
        order: str = "_",
        aggregate: Aggregates = Aggregates(),
    ) -> Self:
        """Group table by adjacent values in columns.

        Not directly supported in ibis; requires window functions, which may be significantly slower than `group`.
        """
        aggs = {name: ibis._[name].min() for name in by}
        aggs.update(aggregate)
        if counts:
            aggs[counts] = ibis._.count()
        columns = [~ibis._[name].identical_to(ibis._[name].lag()) for name in by]
        columns += map(Expression.to_ibis, split)
        table = self.table.mutate({order: ibis.or_(*columns)})  # window functions can't be nested
        table = table.mutate({order: table[order].cumsum()})
        return self.resolve(info, table.aggregate(aggs, by=[order]).order_by(order))

    # dynamic fields - these methods are wrapped with type annotations

    def columns(self, info: Info) -> dict:
        """Fields for each column."""
        names = selections(*info.selected_fields)
        table = self.table
        counts = collections.Counter()
        for field in info.selected_fields:
            counts.update(selections(*field.selections))
        if counts.total() > 1 and not isinstance(table, ibis.expr.types.CachedTable):
            table = cached(table.select(*names))
        return {name: Column.cast(table[name]) for name in names}

    def row(self, info: Info, index: int = 0) -> dict:
        """Scalar values at index."""
        names = selections(*info.selected_fields)
        table = ibis.memtable(self.table.select(*names)[index:][:1].to_pyarrow())
        row = {}
        for name in table.columns:
            (row[name],) = table[name].to_list()
            if isinstance(row[name], list):
                row[name] = Column.cast(table[name].unnest())
        return row

    def filter(self, info: Info, where: list[Expression] = [], **predicates: Filter) -> Self:
        """[Filter](https://ibis-project.org/reference/expression-tables#ibis.expr.types.relations.Table.filter) rows by predicates.

        Schema derived fields provide syntax for simple queries; `where` supports complex queries, combined with `and`.
        """
        exprs = list(map(Expression.to_ibis, where)) + list(Filter.to_exprs(**predicates))
        return self.resolve(info, self.table.filter(*exprs) if exprs else self.table)

    # protected fields = these methods can optionally be registered with `strawberry.field`

    @doc_field(
        query="SQL query string",
        alias="[alias](https://ibis-project.org/reference/expression-tables#ibis.expr.types.relations.Table.alias) of the table expression referenced in the query",
        dialect="input SQL dialect; defaults to the backend's native dialect",
        params="JSON object or array for `:name` or `?` placeholders; replaced by `sqlglot` before execution",
        schema="output names and types to skip schema inference, which may require round trips to remote backends",
    )
    def sql(
        self,
        info: Info,
        query: str,
        alias: str = "",
        dialect: str | None = None,
        params: JSON | None = UNSET,
        schema: list[Field] | None = None,
    ) -> Self:
        """[Run a SQL](https://ibis-project.org/reference/expression-tables#ibis.expr.types.relations.Table.sql) query against a table expression."""
        if params is not UNSET:
            expr = parse_one(query, dialect=dialect)
            match params:
                case dict():
                    params: dict = {key: exp.convert(params[key]) for key in params}
                    expr = exp.replace_placeholders(expr, **params)
                case list():
                    expr = exp.replace_placeholders(expr, *params)
                case _:
                    expr = exp.replace_placeholders(expr, params)
            query = expr.sql(dialect=dialect)
        table = self.table.alias(alias) if alias else self.table
        if schema is None:  # ibis-project/ibis#12216
            return self.resolve(info, table.sql(query, dialect=dialect))
        if dialect is not None:
            query = table.get_backend()._transpile_sql(query, dialect=dialect)
        view = ibis.expr.operations.SQLStringView(  # type: ignore
            parent=table.op(), query=query, schema={field.name: field.type for field in schema}
        )
        return self.resolve(info, view.to_expr())

    sql.extensions.append(PermissionExtension([Deny()], use_directives=False))
