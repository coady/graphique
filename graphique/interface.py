"""
Primary Dataset interface.

Doesn't require knowledge of the schema.
"""

# mypy: disable-error-code=valid-type
import collections
import inspect
import itertools
from collections.abc import Callable, Iterable, Iterator, Mapping
from datetime import timedelta
from typing import Annotated, Optional, Union, no_type_check
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import strawberry.asgi
from strawberry import Info
from strawberry.extensions.utils import get_path_from_info
from typing_extensions import Self
from .core import Batch, Column as C, ListChunk, Table as T
from .inputs import CountAggregate, Cumulative, Diff, Expression, Field, Filter
from .inputs import HashAggregates, ListFunction, Pairwise, Projection, Rank
from .inputs import ScalarAggregate, TDigestAggregate, VarianceAggregate, links, provisional
from .models import Column, doc_field, selections
from .scalars import Long

Root = Union[ds.Dataset, ds.Scanner, pa.Table]


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


def doc_argument(annotation, func: Callable, **kwargs):
    """Use function doc for argument description."""
    kwargs['description'] = inspect.getdoc(func).splitlines()[0]  # type: ignore
    return Annotated[annotation, strawberry.argument(**kwargs)]


@strawberry.type(description="dataset schema")
class Schema:
    names: list[str] = strawberry.field(description="field names")
    types: list[str] = strawberry.field(
        description="[arrow types](https://arrow.apache.org/docs/python/api/datatypes.html), corresponding to `names`"
    )
    partitioning: list[str] = strawberry.field(description="partition keys")
    index: list[str] = strawberry.field(description="sorted index columns")


@strawberry.interface(description="an arrow dataset, scanner, or table")
class Dataset:
    def __init__(self, table: Root):
        self.table = table

    def references(self, info: Info, level: int = 0) -> set:
        """Return set of every possible future column reference."""
        fields = info.selected_fields
        for _ in range(level):
            fields = itertools.chain(*[field.selections for field in fields])
        return set(itertools.chain(*map(references, fields))) & set(self.schema().names)

    def scanner(self, info: Info, **options) -> ds.Scanner:
        """Return scanner with only the columns necessary to proceed."""
        options.setdefault('columns', list(self.references(info)))
        dataset = ds.dataset(self.table) if isinstance(self.table, pa.Table) else self.table
        if isinstance(dataset, ds.Dataset):
            return dataset.scanner(**options)
        options['schema'] = dataset.projected_schema
        return ds.Scanner.from_batches(dataset.to_batches(), **options)

    def select(self, info: Info, length: Optional[int] = None) -> pa.Table:
        """Return table with only the rows and columns necessary to proceed."""
        if isinstance(self.table, pa.Table):
            return self.table.select(self.references(info))
        scanner = self.scanner(info)
        if length is None:
            return self.add_metric(info, scanner.to_table(), mode='read')
        return self.add_metric(info, scanner.head(length), mode='head')

    @classmethod
    @no_type_check
    def resolve_reference(cls, info: Info, **keys) -> Self:
        """Return table from federated keys."""
        self = getattr(info.root_value, cls.field)
        queries = {name: Filter(eq=[keys[name]]) for name in keys}
        return self.filter(info, **queries)

    def columns(self, info: Info) -> dict:
        """fields for each column"""
        table = self.select(info)
        return {name: Column.cast(table[name]) for name in table.schema.names}

    def row(self, info: Info, index: int = 0) -> dict:
        """Return scalar values at index."""
        table = self.select(info, index + 1 if index >= 0 else None)
        row = {}
        for name in table.schema.names:
            scalar = table[name][index]
            columnar = isinstance(scalar, pa.ListScalar)
            row[name] = Column.fromscalar(scalar) if columnar else scalar.as_py()
        return row

    def filter(self, info: Info, **queries: Filter) -> Self:
        """Return table with rows which match all queries.

        See `scan(filter: ...)` for more advanced queries. Additional feature: sorted tables
        support binary search
        """
        table = self.table
        prev = info.path.prev
        search = isinstance(table, pa.Table) and (prev is None or prev.typename == 'Query')
        for name in self.schema().index if search else []:
            assert not table[name].null_count, f"search requires non-null column: {name}"
            query = dict(queries.pop(name))
            if 'eq' in query:
                table = T.is_in(table, name, *query['eq'])
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
            if len(query.pop('eq', [])) != 1 or query:
                break
        self = type(self)(table)
        expr = Expression.from_query(**queries)
        return self if expr.to_arrow() is None else self.scan(info, filter=expr)

    @doc_field
    def type(self) -> str:
        """[arrow type](https://arrow.apache.org/docs/python/api/dataset.html#classes)"""
        return type(self.table).__name__

    @doc_field
    def schema(self) -> Schema:
        """dataset schema"""
        table = self.table
        schema = table.projected_schema if isinstance(table, ds.Scanner) else table.schema
        partitioning = getattr(table, 'partitioning', None)
        index = (schema.pandas_metadata or {}).get('index_columns', [])
        return Schema(
            names=schema.names,
            types=schema.types,
            partitioning=partitioning.schema.names if partitioning else [],
            index=[name for name in index if isinstance(name, str)],
        )  # type: ignore

    @doc_field
    def optional(self) -> Optional[Self]:
        """Nullable field to stop error propagation, enabling partial query results.

        Will be replaced by client controlled nullability.
        """
        return self

    @staticmethod
    def add_context(info: Info, key: str, **data):  # pragma: no cover
        """Add data to context with path info."""
        info.context.setdefault(key, []).append(dict(data, path=get_path_from_info(info)))

    @staticmethod
    def add_metric(info: Info, table: pa.Table, **data):
        """Add memory usage and other metrics to context with path info."""
        path = tuple(get_path_from_info(info))
        info.context.setdefault('metrics', {})[path] = dict(data, memory=T.size(table))
        return table

    @doc_field
    def length(self) -> Long:
        """number of rows"""
        return len(self.table) if hasattr(self.table, '__len__') else self.table.count_rows()

    @doc_field
    def any(self, info: Info, length: Long = 1) -> bool:
        """Return whether there are at least `length` rows.

        May be significantly faster than `length` for out-of-core data.
        """
        table = self.select(info, length)
        return len(table) >= length

    @doc_field
    def size(self) -> Optional[Long]:
        """buffer size in bytes; null if table is not loaded"""
        return getattr(self.table, 'nbytes', None)

    @doc_field(
        name="column name(s); multiple names access nested struct fields",
        cast=f"cast array to {links.type}",
        safe="check for conversion errors on cast",
    )
    def column(
        self, info: Info, name: list[str], cast: str = '', safe: bool = True
    ) -> Optional[Column]:
        """Return column of any type by name.

        This is typically only needed for aliased or casted columns.
        If the column is in the schema, `columns` can be used instead.
        """
        expr = Expression(name=name, cast=cast, safe=safe).to_arrow()  # type: ignore
        return Column.cast(*self.scanner(info, columns={'': expr}).to_table())

    @doc_field(
        offset="number of rows to skip; negative value skips from the end",
        length="maximum number of rows to return",
        reverse="reverse order after slicing; forces a copy",
    )
    def slice(
        self, info: Info, offset: Long = 0, length: Optional[Long] = None, reverse: bool = False
    ) -> Self:
        """Return zero-copy slice of table."""
        table = self.select(info, length and (offset + length if offset >= 0 else None))
        table = table[offset:][:length]  # `slice` bug: ARROW-15412
        return type(self)(table[::-1] if reverse else table)

    @doc_field(
        by="column names; empty will aggregate into a single row table",
        counts="optionally include counts in an aliased column",
        aggregate="aggregation functions applied to other columns",
    )
    def group(
        self,
        info: Info,
        by: list[str] = [],
        counts: str = '',
        aggregate: HashAggregates = {},  # type: ignore
    ) -> Self:
        """Return table grouped by columns.

        See `column` for accessing any column which has changed type. See `tables` to split on any
        aggregated list columns.
        """
        table, aggs = self.table, dict(aggregate)
        refs = {agg.name for values in aggs.values() for agg in values}
        fragments = set(T.fragment_keys(self.table))
        dicts = (pa.types.is_dictionary(table.schema.field(name).type) for name in set(by) | refs)
        if isinstance(table, ds.Scanner) or any(dicts):
            table = self.select(info)
        if fragments and set(by) <= fragments:
            if set(by) == fragments:
                return type(self)(self.fragments(info, counts, aggregate))
            if fragments.isdisjoint(refs) and set(aggs) <= Field.associatives:
                table = self.fragments(info, counts, aggregate)
                aggs.setdefault('sum', []).extend(Field(agg.alias) for agg in aggs.pop('count', []))
                if counts:
                    aggs['sum'].append(Field(counts))
                    counts = ''
                for agg in itertools.chain(*aggs.values()):
                    agg.name = agg.alias
        loaded = isinstance(table, pa.Table)
        table = T.group(table, *by, counts=counts, **aggs)
        return type(self)(table if loaded else self.add_metric(info, table, mode='group'))

    def fragments(self, info: Info, counts: str = '', aggregate: HashAggregates = {}) -> pa.Table:  # type: ignore
        """Return table from scanning fragments and grouping by partitions.

        Requires a partitioned dataset. Faster and less memory intensive than `group`.
        """
        schema = self.table.partitioning.schema  # requires a Dataset
        aggs = dict(aggregate)
        names = self.references(info, level=1)
        names.update(agg.name for value in aggs.values() for agg in value)
        projection = {name: pc.field(name) for name in names - set(schema.names)}
        columns = collections.defaultdict(list)
        for fragment in T.get_fragments(self.table):
            row = ds.get_partition_keys(fragment.partition_expression)
            if projection:
                table = fragment.to_table(columns=projection)
                row |= T.aggregate(table, counts=counts, **aggs)
            elif counts:
                row[counts] = fragment.count_rows()
            arrays = {name: value for name, value in row.items() if isinstance(value, pa.Array)}
            row |= T.columns(pa.RecordBatch.from_pydict(arrays))
            for name in row:
                columns[name].append(row[name])
        for name, values in columns.items():
            if isinstance(values[0], pa.Scalar):
                columns[name] = C.from_scalars(values)
            elif isinstance(values[0], pa.Array):
                columns[name] = ListChunk.from_scalars(values)
        columns |= {field.name: pa.array(columns[field.name], field.type) for field in schema}
        return self.add_metric(info, pa.table(columns), mode='fragment')

    @doc_field(
        by="column names",
        split="optional predicates to split on; scalars are compared to pairwise difference",
        counts="optionally include counts in an aliased column",
    )
    @no_type_check
    def runs(
        self, info: Info, by: list[str] = [], split: list[Diff] = [], counts: str = ''
    ) -> Self:
        """Return table grouped by pairwise differences.

        Differs from `group` by relying on adjacency, and is typically faster. Other columns are
        transformed into list columns. See `column` and `tables` to further access lists.
        """
        table = self.select(info)
        predicates = {}
        for diff in map(dict, split):
            name = diff.pop('name')
            ((func, value),) = diff.items()
            if pa.types.is_timestamp(table.field(name).type):
                value = timedelta(seconds=value)
            predicates[name] = (getattr(pc, func), value)[: 1 if value is None else 2]
        table, counts_ = T.runs(table, *by, **predicates)
        return type(self)(table.append_column(counts, counts_) if counts else table)

    @doc_field(
        by="column names; prefix with `-` for descending order",
        length="maximum number of rows to return; may be significantly faster but is unstable",
        null_placement="where nulls in input should be sorted; incompatible with `length`",
    )
    def sort(
        self,
        info: Info,
        by: list[str],
        length: Optional[Long] = None,
        null_placement: str = 'at_end',
    ) -> Self:
        """Return table slice sorted by specified columns.

        Optimized for length == 1; matches min or max values.
        """
        kwargs = dict(length=length, null_placement=null_placement)
        if isinstance(self.table, pa.Table) or length is None:
            table = self.select(info)
        else:
            expr, by = T.rank_keys(self.table, length, *by, dense=False)
            scanner = self.scanner(info, filter=expr)
            if not by:
                return type(self)(self.add_metric(info, scanner.head(length), mode='head'))
            table = T.map_batch(scanner, T.sort, *by, **kwargs)
            self.add_metric(info, table, mode='batch')
        return type(self)(T.sort(table, *by, **kwargs))  # type: ignore

    @doc_field(
        by="column names; prefix with `-` for descending order",
        max="maximum dense rank to select; optimized for == 1 (min or max)",
        null_placement="where nulls in input should be ranked",
    )
    def rank(self, info: Info, by: list[str], max: int = 1, null_placement: str = 'at_end') -> Self:
        """Return table selected by maximum dense rank."""
        kwargs = dict(null_placement=null_placement)
        if isinstance(self.table, pa.Table):
            table = self.select(info)
        else:
            expr, by = T.rank_keys(self.table, max, *by)
            if not by:
                return type(self)(self.table.filter(expr))
            table = T.map_batch(self.scanner(info, filter=expr), T.ranked, max, *by, **kwargs)
            self.add_metric(info, table, mode='batch')
        return type(self)(T.ranked(table, max, *by, **kwargs))

    @staticmethod
    def apply_list(table: Batch, list_: ListFunction) -> Batch:
        expr = list_.filter.to_arrow() if list_.filter else None
        if expr is not None:
            table = T.filter_list(table, expr)
        if list_.rank:
            table = T.map_list(table, T.ranked, list_.rank.max, *list_.rank.by)
        if list_.sort:
            table = T.map_list(table, T.sort, *list_.sort.by, length=list_.sort.length)
        columns = {}
        for func, field in dict(list_).items():
            columns[field.alias] = getattr(ListChunk, func)(table[field.name], **field.options)
        return T.union(table, pa.RecordBatch.from_pydict(columns))

    @doc_field
    @no_type_check
    def apply(
        self,
        info: Info,
        cumulative_max: doc_argument(list[Cumulative], func=pc.cumulative_max) = [],
        cumulative_mean: doc_argument(list[Cumulative], func=pc.cumulative_mean) = [],
        cumulative_min: doc_argument(list[Cumulative], func=pc.cumulative_min) = [],
        cumulative_prod: doc_argument(list[Cumulative], func=pc.cumulative_prod) = [],
        cumulative_sum: doc_argument(list[Cumulative], func=pc.cumulative_sum) = [],
        fill_null_backward: doc_argument(list[Field], func=pc.fill_null_backward) = [],
        fill_null_forward: doc_argument(list[Field], func=pc.fill_null_forward) = [],
        pairwise_diff: doc_argument(list[Pairwise], func=pc.pairwise_diff) = [],
        rank: doc_argument(list[Rank], func=pc.rank) = [],
        list_: Annotated[
            ListFunction,
            strawberry.argument(name='list', description="functions for list arrays."),
        ] = {},
    ) -> Self:
        """Return view of table with vector functions applied across columns.

        Applied functions load arrays into memory as needed. See `scan` for scalar functions,
        which do not require loading.
        """
        table = T.map_batch(self.scanner(info), self.apply_list, list_)
        self.add_metric(info, table, mode='batch')
        columns = {}
        funcs = pc.cumulative_max, pc.cumulative_mean, pc.cumulative_min, pc.cumulative_prod
        funcs += pc.cumulative_sum, C.fill_null_backward, C.fill_null_forward, C.pairwise_diff
        funcs += (pc.rank,)
        for func in funcs:
            for field in locals()[func.__name__]:
                callable = func
                if field.options.pop('checked', False):
                    callable = getattr(pc, func.__name__ + '_checked')
                columns[field.alias] = callable(table[field.name], **field.options)
        return type(self)(T.union(table, pa.table(columns)))

    @doc_field
    def flatten(self, info: Info, indices: str = '') -> Self:
        """Return table with list arrays flattened.

        At least one list column must be referenced, and all list columns must have the same lengths.
        """
        batches = T.flatten(self.scanner(info), indices)
        batch = next(batches)
        scanner = ds.Scanner.from_batches(itertools.chain([batch], batches), schema=batch.schema)
        return type(self)(self.oneshot(info, scanner))

    @doc_field
    def tables(self, info: Info) -> list[Optional[Self]]:  # type: ignore
        """Return a list of tables by splitting list columns.

        At least one list column must be referenced, and all list columns must have the same lengths.
        """
        for batch in self.scanner(info).to_batches():
            for row in T.split(batch):
                yield None if row is None else type(self)(pa.Table.from_batches([row]))

    @doc_field
    def aggregate(
        self,
        info: Info,
        approximate_median: doc_argument(list[ScalarAggregate], func=pc.approximate_median) = [],
        count: doc_argument(list[CountAggregate], func=pc.count) = [],
        count_distinct: doc_argument(list[CountAggregate], func=pc.count_distinct) = [],
        distinct: Annotated[
            list[CountAggregate],
            strawberry.argument(description="distinct values within each scalar"),
        ] = [],
        first: doc_argument(list[Field], func=ListChunk.first) = [],
        last: doc_argument(list[Field], func=ListChunk.last) = [],
        max: doc_argument(list[ScalarAggregate], func=pc.max) = [],
        mean: doc_argument(list[ScalarAggregate], func=pc.mean) = [],
        min: doc_argument(list[ScalarAggregate], func=pc.min) = [],
        product: doc_argument(list[ScalarAggregate], func=pc.product) = [],
        stddev: doc_argument(list[VarianceAggregate], func=pc.stddev) = [],
        sum: doc_argument(list[ScalarAggregate], func=pc.sum) = [],
        tdigest: doc_argument(list[TDigestAggregate], func=pc.tdigest) = [],
        variance: doc_argument(list[VarianceAggregate], func=pc.variance) = [],
    ) -> Self:
        """Return table with scalar aggregate functions applied to list columns."""
        table = self.select(info)
        columns = T.columns(table)
        agg_fields: dict = collections.defaultdict(dict)
        keys: tuple = 'approximate_median', 'count', 'count_distinct', 'distinct', 'first', 'last'
        keys += 'max', 'mean', 'min', 'product', 'stddev', 'sum', 'tdigest', 'variance'
        for key in keys:
            func = getattr(ListChunk, key, None)
            for agg in locals()[key]:
                if func is None or key == 'sum':  # `sum` is a method on `Array``
                    agg_fields[agg.name][key] = agg
                else:
                    columns[agg.alias] = func(table[agg.name], **agg.options)
        for name, aggs in agg_fields.items():
            funcs = {key: agg.astuple(key)[2] for key, agg in aggs.items()}
            batch = ListChunk.aggregate(table[name], **funcs)
            columns.update(zip([agg.alias for agg in aggs.values()], batch))
        return type(self)(pa.table(columns))

    aggregate.deprecation_reason = ListFunction.deprecation

    def project(self, info: Info, columns: list[Projection]) -> dict:
        """Return projected columns, including all references from below fields."""
        projection = {name: pc.field(name) for name in self.references(info, level=1)}
        projection |= {col.alias or '.'.join(col.name): col.to_arrow() for col in columns}
        if '' in projection:
            raise ValueError(f"projected columns need a name or alias: {projection['']}")
        return projection

    @classmethod
    def oneshot(cls, info: Info, scanner: ds.Scanner) -> Union[ds.Scanner, pa.Table]:
        """Load oneshot scanner if needed."""
        selected = selections(*info.selected_fields)
        selected['type'] = selected['schema'] = 0
        if sum(selected.values()) > 1:
            return cls.add_metric(info, scanner.to_table(), mode='oneshot')
        return scanner

    @doc_field(filter="selected rows", columns="projected columns")
    def scan(self, info: Info, filter: Expression = {}, columns: list[Projection] = []) -> Self:  # type: ignore
        """Select rows and project columns without memory usage."""
        expr = filter.to_arrow()
        if expr is not None and not columns and isinstance(self.table, ds.Dataset):
            return type(self)(self.table.filter(expr))
        scanner = self.scanner(info, filter=expr, columns=self.project(info, columns))
        if isinstance(self.table, ds.Scanner):
            scanner = self.oneshot(info, scanner)
        return type(self)(scanner)

    @doc_field(
        right="name of right table; must be on root Query type",
        keys="column names used as keys on the left side",
        right_keys="column names used as keys on the right side; defaults to left side.",
        join_type="the kind of join: 'left semi', 'right semi', 'left anti', 'right anti', 'inner', 'left outer', 'right outer', 'full outer'",
        left_suffix="add suffix to left column names; for preventing collisions",
        right_suffix="add suffix to right column names; for preventing collisions.",
        coalesce_keys="omit duplicate keys",
    )
    def join(
        self,
        info: Info,
        right: str,
        keys: list[str],
        right_keys: Optional[list[str]] = None,
        join_type: str = 'left outer',
        left_suffix: str = '',
        right_suffix: str = '',
        coalesce_keys: bool = True,
    ) -> Self:
        """Provisional: [join](https://arrow.apache.org/docs/python/generated/pyarrow.dataset.Dataset.html#pyarrow.dataset.Dataset.join) this table with another table on the root Query type."""
        left, right = (
            root.table if isinstance(root.table, ds.Dataset) else root.select(info)
            for root in (self, getattr(info.root_value, right))
        )
        table = left.join(
            right,
            keys=keys,
            right_keys=right_keys,
            join_type=join_type,
            left_suffix=left_suffix,
            right_suffix=right_suffix,
            coalesce_keys=coalesce_keys,
        )
        return type(self)(table)

    join.directives = [provisional()]

    @doc_field
    def take(self, info: Info, indices: list[Long]) -> Self:
        """Select rows from indices."""
        table = self.scanner(info).take(indices)
        return type(self)(self.add_metric(info, table, mode='take'))

    @doc_field
    def drop_null(self, info: Info) -> Self:
        """Remove missing values from referenced columns in the table."""
        if isinstance(self.table, pa.Table):
            return type(self)(pc.drop_null(self.select(info)))
        scanner = self.scanner(info)
        batches = map(pc.drop_null, scanner.to_batches())
        scanner = ds.Scanner.from_batches(batches, schema=scanner.projected_schema)
        return type(self)(self.oneshot(info, scanner))
