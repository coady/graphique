"""
Primary Dataset interface.

Doesn't require knowledge of the schema.
"""
import collections
import inspect
import itertools
import types
from datetime import timedelta
from typing import Callable, Iterable, Iterator, List, Mapping, Optional, Union, no_type_check
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import strawberry.asgi
from strawberry.types import Info
from typing_extensions import Annotated
from .core import Agg, Column as C, ListChunk, Table as T
from .inputs import Aggregations, Cumulative, Diff, Digitize, Expression, Field, Filter, Projection
from .inputs import links, ListFunction
from .models import Column, annotate, doc_field, selections
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

    def __init_subclass__(cls):
        """Downcast fields which return an `Dataset` to its implemented type."""
        for name, func in inspect.getmembers(cls, inspect.isfunction):
            if func.__annotations__.get('return') in ('Dataset', List['Dataset']):
                clone = types.FunctionType(func.__code__, func.__globals__)
                clone.__annotations__.update({'return': cls, 'info': Info})
                if name == 'aggregate':
                    field = Aggregations.resolver(clone)
                else:
                    field = annotate(func, List[cls] if name == 'tables' else cls)
                setattr(cls, name, field)

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

    def select(self, info: Info, length: int = None) -> pa.Table:
        """Return table with only the rows and columns necessary to proceed."""
        if isinstance(self.table, pa.Table):
            return self.table.select(self.references(info))
        scanner = self.scanner(info)
        return scanner.to_table() if length is None else scanner.head(length)

    @classmethod
    @no_type_check
    def resolve_reference(cls, info, **keys) -> 'Dataset':
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

    def filter(self, info: Info, **queries: Filter) -> 'Dataset':
        """Return table with rows which match all queries."""
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
    def length(self) -> Long:
        """number of rows"""
        return len(self.table) if hasattr(self.table, '__len__') else self.table.count_rows()

    @doc_field(
        name="column name(s); multiple names access nested struct fields",
        cast=f"cast array to {links.type}",
        safe="check for conversion errors on cast",
    )
    def column(self, info: Info, name: List[str], cast: str = '', safe: bool = True) -> Column:
        """Return column of any type by name.

        This is typically only needed for aliased columns added by `apply` or `aggregate`.
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
    ) -> 'Dataset':
        """Return zero-copy slice of table."""
        table = self.select(info, length and (offset + length if offset >= 0 else None))
        table = table[offset:][:length]  # `slice` bug: ARROW-15412
        return type(self)(table[::-1] if reverse else table)

    @doc_field(
        by="column names",
        counts="optionally include counts in an aliased column",
        aggregate="grouped aggregation functions",
    )
    def group(
        self, info: Info, by: List[str], counts: str = '', aggregate: Aggregations = {}  # type: ignore
    ) -> 'Dataset':
        """Return table grouped by columns, with stable ordering.

        Columns which are not aggregated are transformed into list columns.
        See `column`, `aggregate`, and `tables` for further usage of list columns.
        """
        scalars, aggs = set(by), {}
        for func, values in dict(aggregate).items():
            if values:
                aggs[func] = [Agg(**dict(value)) for value in values]
                scalars.update(agg.alias for agg in aggs[func])
        lists = self.references(info, level=1) - scalars - {counts}
        if isinstance(self.table, pa.Table) or lists or not set(aggs) <= Agg.associatives:
            table = self.select(info)
        else:  # scan in parallel when possible
            table = T.map_batch(self.scanner(info), T.group, *by, counts=counts, **aggs)
            if counts:
                aggs.setdefault('sum', []).append(Agg(counts))
                counts = ''
        return type(self)(T.group(table, *by, counts=counts, list=list(map(Agg, lists)), **aggs))

    @doc_field(
        by="column names",
        diffs="optional inequality predicates; scalars are compared to the adjacent difference",
        counts="optionally include counts in an aliased column",
    )
    @no_type_check
    def partition(
        self, info: Info, by: List[str], diffs: List[Diff] = [], counts: str = ''
    ) -> 'Dataset':
        """Return table partitioned by discrete differences of the values.

        Differs from `group` by relying on adjacency, and is typically faster.
        Other columns can be accessed by the `column` field as a `ListColumn`.
        Typically used in conjunction with `aggregate` or `tables`.
        """
        table = self.select(info)
        funcs = {diff.pop('name'): diff for diff in map(dict, diffs)}
        names = list(itertools.takewhile(lambda name: name not in funcs, by))
        predicates = {}
        for name in by[len(names) :]:  # noqa: E203
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
    )
    def sort(self, info: Info, by: List[str], length: Optional[Long] = None) -> 'Dataset':
        """Return table slice sorted by specified columns.

        Sorting on list columns will sort within scalars, all of which must have the same lengths.
        """
        table = self.select(info)
        scalars, lists = [], []  # type: ignore
        for key in by:
            (lists if C.is_list_type(table[key.lstrip('-')]) else scalars).append(key)
        if scalars:
            table = T.sort(table, *scalars, length=length)
        if lists:
            table = T.sort_list(table, *lists, length=length)
        return type(self)(table)

    @doc_field(by="column names")
    def min(self, info: Info, by: List[str]) -> 'Dataset':
        """Return table with minimum values per column."""
        table = self.select(info)
        return type(self)(T.matched(table, C.min, *by))

    @doc_field(by="column names")
    def max(self, info: Info, by: List[str]) -> 'Dataset':
        """Return table with maximum values per column."""
        table = self.select(info)
        return type(self)(T.matched(table, C.max, *by))

    @doc_field
    @no_type_check
    def apply(
        self,
        info: Info,
        digitize: doc_argument(List[Digitize], func=C.digitize) = [],
        cumulative_sum: doc_argument(List[Cumulative], func=pc.cumulative_sum) = [],
        fill_null_backward: doc_argument(List[Field], func=pc.fill_null_backward) = [],
        fill_null_forward: doc_argument(List[Field], func=pc.fill_null_forward) = [],
        list: List[ListFunction] = [],
    ) -> 'Dataset':
        """Return view of table with functions applied across columns.

        If no alias is provided, the column is replaced and should be of the same type.
        If an alias is provided, a column is appended under that name.
        Applied function load arrays into memory as needed; see `scan` for more functions, which
        do not require loading.
        """
        table = self.select(info)
        columns = {}
        for value in map(dict, list):
            expr = value.pop('filter').to_arrow()
            if expr is not None:
                table = T.filter_list(table, expr)
            for func, field in value.items():
                name, args, kwargs = field.serialize(table)
                columns[name] = getattr(ListChunk, func)(*args, **kwargs)
        args = digitize, cumulative_sum, fill_null_backward, fill_null_forward
        funcs = C.digitize, pc.cumulative_sum, C.fill_null_backward, C.fill_null_forward
        for fields, func in zip(args, funcs):
            for field in fields:
                name, args, kwargs = field.serialize(table)
                columns[name] = func(*args, **kwargs)
        return type(self)(T.union(table, pa.table(columns)))

    @doc_field
    def tables(self, info: Info) -> List['Dataset']:  # type: ignore
        """Return a list of tables by splitting list columns, typically used after grouping.

        At least one list column must be referenced, and all list columns must have the same lengths.
        """
        table = self.select(info)
        lists = {name for name in table.column_names if C.is_list_type(table[name])}
        scalars = set(table.column_names) - lists
        for index, count in enumerate(T.list_value_length(table).to_pylist()):
            row = {name: pa.repeat(table[name][index], count) for name in scalars}
            row.update({name: table[name][index].values for name in lists})
            yield type(self)(pa.table(row))

    @Aggregations.resolver
    @no_type_check
    def aggregate(self, info: Info, **fields) -> 'Dataset':
        """Return table with aggregate functions applied to list columns, typically used after grouping.

        Columns which are aliased or change type can be accessed by the `column` field.
        """
        table = self.select(info)
        columns = {name: table[name] for name in table.column_names}
        agg_fields = collections.defaultdict(dict)
        for key in fields:
            func = getattr(ListChunk, key, None)
            for field in fields[key]:
                agg = Agg(**dict(field))
                if func is None or key == 'sum':  # `sum` is a method on `Array``
                    agg_fields[agg.name][key] = agg
                else:
                    columns[agg.alias] = func(table[agg.name], **agg.options)
        for name, aggs in agg_fields.items():
            funcs = {key: agg.astuple(key)[1] for key, agg in aggs.items()}
            arrays = ListChunk.aggregate(table[name], **funcs).flatten()
            columns.update(zip([agg.alias for agg in aggs.values()], arrays))
        return type(self)(pa.table(columns))

    @doc_field(filter="selected rows", columns="projected columns")
    def scan(
        self, info: Info, filter: Expression = {}, columns: List[Projection] = []  # type: ignore
    ) -> 'Dataset':
        """Select rows and project columns without memory usage."""
        projection = {name: pc.field(name) for name in self.references(info, level=1)}
        projection.update({col.alias or '.'.join(col.name): col.to_arrow() for col in columns})
        if '' in projection:
            raise ValueError("projected columns need a name or alias")
        scanner = self.scanner(info, filter=filter.to_arrow(), columns=projection)
        oneshot = len(selections(*info.selected_fields)) > 1 and isinstance(self.table, ds.Scanner)
        return type(self)(scanner.to_table() if oneshot else scanner)

    @doc_field(
        right="name of right table; must be on root Query type",
        keys="column names used as keys on the left side",
        right_keys="column names used as keys on the right side; defaults to left side.",
        join_type="""the kind of join: “left semi”, “right semi”, “left anti”, “right anti”, “inner”, “left outer”, “right outer”, “full outer”""",
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
    ) -> 'Dataset':
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

    @doc_field
    def take(self, info: Info, indices: List[Long]) -> 'Dataset':
        """Provisional: select rows from indices."""
        return type(self)(self.scanner(info).take(indices))
