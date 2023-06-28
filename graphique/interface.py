"""
Primary Dataset interface.

Doesn't require knowledge of the schema.
"""
# mypy: disable-error-code=valid-type
import collections
import inspect
import itertools
from datetime import timedelta
from typing import Callable, Iterable, Iterator, List, Mapping, Optional, Union, no_type_check
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import strawberry.asgi
from strawberry import UNSET
from strawberry.extensions.utils import get_path_from_info
from strawberry.types import Info
from typing_extensions import Annotated, Self
from .core import Batch, Column as C, ListChunk, Table as T, sort_key
from .inputs import CountAggregate, Cumulative, Diff, Expression, Field, Filter
from .inputs import HashAggregates, ListFunction, Projection, Rank, Ranked, ScalarAggregate, Sort
from .inputs import TDigestAggregate, VarianceAggregate, links, provisional
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
    names: List[str] = strawberry.field(description="field names")
    types: List[str] = strawberry.field(
        description="[arrow types](https://arrow.apache.org/docs/python/api/datatypes.html), corresponding to `names`"
    )
    partitioning: List[str] = strawberry.field(description="partition keys")
    index: List[str] = strawberry.field(description="sorted index columns")


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
        return scanner.to_table() if length is None else scanner.head(length)

    @classmethod
    @no_type_check
    def resolve_reference(cls, info, **keys) -> Self:
        """Return table from federated keys."""
        self = getattr(info.root_value, cls.field)
        queries = {name: Filter(eq=[keys[name]]) for name in keys}
        return self.filter(Info(info, None), **queries)

    def columns(self, info: Info) -> dict:
        """fields for each column"""
        table = self.select(info)
        return {name: Column.cast(table[name]) for name in table.column_names}

    def row(self, info: Info, index: int = 0) -> dict:
        """Return scalar values at index."""
        table = self.select(info, index + 1 if index >= 0 else None)
        row = {}
        for name in table.column_names:
            scalar = table[name][index]
            columnar = isinstance(scalar, pa.ListScalar)
            row[name] = Column.fromscalar(scalar) if columnar else scalar.as_py()
        return row

    def filter(self, info: Info, **queries: Filter) -> Self:
        """Return table with rows which match all queries.

        See `scan(filter: ...)` for more advanced queries. Additional features
        * sorted tables support binary search
        * partitioned datasets retain fragment information when filtered on keys
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

    @staticmethod
    def add_context(info: Info, key: str, **data):
        """Add data to context with path info."""
        info.context.setdefault(key, []).append(dict(data, path=get_path_from_info(info)))

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

    @doc_field(
        name="column name(s); multiple names access nested struct fields",
        cast=f"cast array to {links.type}",
        safe="check for conversion errors on cast",
    )
    def column(self, info: Info, name: List[str], cast: str = '', safe: bool = True) -> Column:
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
        by="column names",
        counts="optionally include counts in an aliased column",
        aggregate="grouped aggregation functions",
        filter="filter within list scalars",
        sort="sort within list scalars",
        rank="filter by dense rank within list scalars",
    )
    @no_type_check
    def group(
        self,
        info: Info,
        by: List[str] = [],
        counts: str = '',
        aggregate: HashAggregates = {},
        filter: Expression = {},
        sort: Optional[Sort] = None,
        rank: Optional[Ranked] = None,
    ) -> Self:
        """Return table grouped by columns, with stable ordering.

        See `column`, `aggregate`, and `tables` for further usage of list columns.
        `filter`, `sort`, and `rank` are equivalent to the functions in `apply(list: ...)`,
        but memory optimized.

        Deprecated: columns which are not aggregated are transformed into list columns.
        Use the explicit `list` input instead, which also supports aliasing.
        """
        scalars, refs, aggs = set(by), set(), dict(aggregate)
        for values in aggs.values():
            scalars.update(agg.alias for agg in values)
            refs.update(agg.name for agg in values)
        flat = isinstance(self.table, pa.Table) or not set(aggs) <= Field.associatives
        if not aggs.setdefault('list', []):
            aggs['list'] += map(Field, self.references(info, level=1) - scalars - {counts})
            aggregate.list += aggs['list']  # only needed for fragments
            if aggs['list']:
                reason = "specify list aggregations explicitly"
                self.add_context(info, 'deprecations', reason=reason)
        distincts = []
        list_opts = dict(filter=filter, sort=sort or UNSET, rank=rank or UNSET)
        list_func = ListFunction(**list_opts)
        fragments = set(T.fragment_keys(self.table))
        if flat:
            table = self.select(info)
        else:  # scan fragments or batches when possible
            if fragments and set(by) <= fragments and fragments.isdisjoint(refs):
                table = self.fragments(info, counts, aggregate, **list_opts).table
            else:
                table = T.map_batch(
                    self.scanner(info),
                    lambda b: self.apply_list(T.group(b, *by, counts=counts, **aggs), list_func),
                )
            list_func.filter = Expression()
            aggs.setdefault('sum', []).extend(Field(agg.alias) for agg in aggs.pop('count', []))
            if counts:
                aggs['sum'].append(Field(counts))
                counts = ''
            distincts = aggs.pop('distinct', [])
            aggs['list'] += (Field(agg.alias) for agg in distincts)
            for agg in itertools.chain(*aggs.values()):
                agg.name = agg.alias
        table = T.group(table, *by, counts=counts, **aggs)
        columns = dict(zip(table.column_names, table))
        if not flat:
            for agg in aggs['list']:
                columns[agg.name] = ListChunk.inner_flatten(*columns[agg.name].chunks)
            for agg in distincts:
                column = columns[agg.name]
                columns[agg.name] = ListChunk.map_list(column, agg.distinct, **agg.options)
        return type(self)(self.apply_list(pa.table(columns), list_func))

    @strawberry.field(deprecation_reason="use `group(by: [<fragment key>, ...])`")
    @no_type_check
    def fragments(
        self,
        info: Info,
        counts: str = '',
        aggregate: HashAggregates = {},
        filter: Expression = {},
        sort: Optional[Sort] = None,
        rank: Optional[Ranked] = None,
    ) -> Self:
        """Return table from scanning fragments and grouping by partitions.

        Requires a partitioned dataset. Faster and less memory intensive than `group`.
        """
        schema = self.table.partitioning.schema  # requires a Dataset
        filter, aggs = (filter.to_arrow() if filter else None), dict(aggregate)
        names = self.references(info, level=1)
        names.update(agg.name for value in aggs.values() for agg in value)
        if rank:
            names.update(dict(map(sort_key, rank.by)))
        if sort:
            names.update(dict(map(sort_key, sort.by)))
        projection = {name: pc.field(name) for name in names - set(schema.names)}
        columns = collections.defaultdict(list)
        for fragment in T.get_fragments(self.table):
            row = ds.get_partition_keys(fragment.partition_expression)
            if projection:
                table = fragment.to_table(filter=filter, columns=projection)
                row.update(T.aggregate(table, counts=counts, **aggs))
            elif counts:
                row[counts] = fragment.count_rows(filter=filter)
            arrays = {name: value for name, value in row.items() if isinstance(value, pa.Array)}
            batch = pa.RecordBatch.from_pydict(arrays)
            if rank:
                batch = T.ranked(batch, rank.max, *rank.by)
            if sort:
                batch = T.sort(batch, *sort.by, length=sort.length)
            row.update({name: batch[name] for name in batch.schema.names})
            for name in row:
                columns[name].append(row[name])
        for name, values in columns.items():
            if isinstance(values[0], pa.Scalar):
                columns[name] = C.from_scalars(values)
            elif isinstance(values[0], pa.Array):
                columns[name] = ListChunk.from_scalars(values)
        columns.update({field.name: pa.array(columns[field.name], field.type) for field in schema})
        return type(self)(pa.table(columns))

    @doc_field(
        by="column names",
        diffs="optional inequality predicates; scalars are compared to the adjacent difference",
        counts="optionally include counts in an aliased column",
    )
    @no_type_check
    def partition(
        self, info: Info, by: List[str], diffs: List[Diff] = [], counts: str = ''
    ) -> Self:
        """Return table partitioned by discrete differences of the values.

        Differs from `group` by relying on adjacency, and is typically faster.
        Other columns can be accessed by the `column` field as a `ListColumn`.
        Typically used in conjunction with `aggregate` or `tables`.
        """
        table = self.select(info)
        funcs = {diff.pop('name'): diff for diff in map(dict, diffs)}
        names = list(itertools.takewhile(lambda name: name not in funcs, by))
        predicates = {}
        for name in by[len(names) :]:
            ((func, value),) = funcs.pop(name, {'not_equal': None}).items()
            predicates[name] = (getattr(pc, func),)
            if value is not None:
                if pa.types.is_timestamp(C.scalar_type(table[name])):
                    value = timedelta(seconds=value)
                predicates[name] += (value,)
        table, counts_ = T.partition(table, *names, **predicates)
        return type(self)(table.append_column(counts, counts_) if counts else table)

    @doc_field(
        by="column names; prefix with `-` for descending order",
        length="maximum number of rows to return; may be significantly faster but is unstable",
        null_placement="where nulls in input should be sorted; incompatible with `length`",
    )
    def sort(
        self,
        info: Info,
        by: List[str],
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
            expr, names = T.rank_keys(self.table, length, *by)
            if length == 1:
                by = names
            scanner = self.scanner(info, filter=expr)
            if not by:
                return type(self)(scanner.head(length))
            table = T.map_batch(scanner, T.sort, *by, **kwargs)
        return type(self)(T.sort(table, *by, **kwargs))  # type: ignore

    @doc_field(
        by="column names; prefix with `-` for descending order",
        max="maximum dense rank to select; optimized for == 1 (min or max)",
        null_placement="where nulls in input should be ranked",
    )
    def rank(self, info: Info, by: List[str], max: int = 1, null_placement: str = 'at_end') -> Self:
        """Return table selected by maximum dense rank."""
        kwargs = dict(null_placement=null_placement)
        if isinstance(self.table, pa.Table):
            table = self.select(info)
        else:
            expr, names = T.rank_keys(self.table, max, *by)
            if not names:
                return type(self)(self.table.filter(expr))
            if max == 1:
                by = names
            table = T.map_batch(self.scanner(info, filter=expr), T.ranked, max, *by, **kwargs)
        return type(self)(T.ranked(table, max, *by, **kwargs))

    @strawberry.field(deprecation_reason="use `rank(by: [...])`")
    def min(self, info: Info, by: List[str]) -> Self:
        return self.rank(info, by)

    @strawberry.field(deprecation_reason="use `rank(by: [-...])`")
    def max(self, info: Info, by: List[str]) -> Self:
        return self.rank(info, ['-' + name for name in by])

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
        cumulative_sum: doc_argument(List[Cumulative], func=pc.cumulative_sum) = [],
        fill_null_backward: doc_argument(List[Field], func=pc.fill_null_backward) = [],
        fill_null_forward: doc_argument(List[Field], func=pc.fill_null_forward) = [],
        rank: doc_argument(List[Rank], func=pc.rank) = [],
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
        columns = {}
        args = cumulative_sum, fill_null_backward, fill_null_forward, rank
        funcs = pc.cumulative_sum, C.fill_null_backward, C.fill_null_forward, pc.rank
        for fields, func in zip(args, funcs):
            for field in fields:
                columns[field.alias] = func(table[field.name], **field.options)
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
    def tables(self, info: Info) -> List[Optional[Self]]:  # type: ignore
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
        approximate_median: doc_argument(List[ScalarAggregate], func=pc.approximate_median) = [],
        count: doc_argument(List[CountAggregate], func=pc.count) = [],
        count_distinct: doc_argument(List[CountAggregate], func=pc.count_distinct) = [],
        distinct: Annotated[
            List[CountAggregate],
            strawberry.argument(description="distinct values within each scalar"),
        ] = [],
        first: doc_argument(List[Field], func=ListChunk.first) = [],
        last: doc_argument(List[Field], func=ListChunk.last) = [],
        max: doc_argument(List[ScalarAggregate], func=pc.max) = [],
        mean: doc_argument(List[ScalarAggregate], func=pc.mean) = [],
        min: doc_argument(List[ScalarAggregate], func=pc.min) = [],
        product: doc_argument(List[ScalarAggregate], func=pc.product) = [],
        stddev: doc_argument(List[VarianceAggregate], func=pc.stddev) = [],
        sum: doc_argument(List[ScalarAggregate], func=pc.sum) = [],
        tdigest: doc_argument(List[TDigestAggregate], func=pc.tdigest) = [],
        variance: doc_argument(List[VarianceAggregate], func=pc.variance) = [],
    ) -> Self:
        """Return table with scalar aggregate functions applied to list columns."""
        table = self.select(info)
        columns = {name: table[name] for name in table.column_names}
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
            funcs = {key: agg.astuple(key)[-1] for key, agg in aggs.items()}
            batch = ListChunk.aggregate(table[name], **funcs)
            columns.update(zip([agg.alias for agg in aggs.values()], batch))
        return type(self)(pa.table(columns))

    aggregate.deprecation_reason = ListFunction.deprecation

    def project(self, info: Info, columns: List[Projection]) -> dict:
        """Return projected columns, including all references from below fields."""
        projection = {name: pc.field(name) for name in self.references(info, level=1)}
        projection.update({col.alias or '.'.join(col.name): col.to_arrow() for col in columns})
        if '' in projection:
            raise ValueError(f"projected columns need a name or alias: {projection['']}")
        return projection

    @staticmethod
    def oneshot(info: Info, scanner: ds.Scanner) -> Union[ds.Scanner, pa.Table]:
        """Load oneshot scanner if needed."""
        selected = selections(*info.selected_fields)
        selected['type'] = selected['schema'] = 0
        return scanner.to_table() if sum(selected.values()) > 1 else scanner

    @doc_field(filter="selected rows", columns="projected columns")
    def scan(
        self, info: Info, filter: Expression = {}, columns: List[Projection] = []  # type: ignore
    ) -> Self:
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
        keys: List[str],
        right_keys: Optional[List[str]] = None,
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
    def take(self, info: Info, indices: List[Long]) -> Self:
        """Select rows from indices."""
        return type(self)(self.scanner(info).take(indices))
