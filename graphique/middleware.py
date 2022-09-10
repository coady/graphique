"""
Service related utilities which don't require knowledge of the schema.
"""
import collections
import inspect
import itertools
import types
from datetime import datetime, timedelta
from typing import Iterable, Iterator, List, Mapping, Optional, Union, no_type_check
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import strawberry.asgi
from strawberry.types import Info
from .core import Agg, Column as C, ListChunk, Table as T
from .inputs import Aggregations, Diff, Expression
from .inputs import Base64Function, BooleanFunction, DateFunction, DateTimeFunction, DecimalFunction
from .inputs import DurationFunction, FloatFunction, IntFunction, LongFunction, ListFunction
from .inputs import StringFunction, StructFunction, TimeFunction, links
from .models import Column, annotate, doc_field, selections
from .scalars import Long, scalar_map


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


class TimingExtension(strawberry.extensions.Extension):
    def on_request_start(self):
        self.start = datetime.now()

    def on_request_end(self):
        end = datetime.now()
        print(f"[{end.replace(microsecond=0)}]: {end - self.start}")


class GraphQL(strawberry.asgi.GraphQL):
    def __init__(self, root_value, debug=False, federated='', **kwargs):
        Schema = strawberry.Schema
        if federated:
            Schema = strawberry.federation.Schema
            Query = type('Query', (), {'__annotations__': {federated: type(root_value)}})
            root_value = strawberry.type(Query)(**{federated: root_value})
        schema = Schema(
            type(root_value),
            types=Column.type_map.values(),
            extensions=[TimingExtension] * bool(debug),
            scalar_overrides=scalar_map,
        )
        super().__init__(schema, debug=debug, **kwargs)
        self.root_value = root_value

    async def get_root_value(self, request):
        return self.root_value


@strawberry.type(description="dataset schema")
class Schema:
    names: List[str] = strawberry.field(description="field names")
    types: List[str] = strawberry.field(
        description="[arrow types](https://arrow.apache.org/docs/python/api/datatypes.html), corresponding to `names`"
    )
    partitioning: Optional[List[str]] = strawberry.field(description="partition keys")


@strawberry.interface(description="an arrow dataset or table")
class Dataset:
    def __init__(self, table: Union[ds.Dataset, ds.Scanner, pa.Table]):
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

    @staticmethod
    def references(info: Info, level: int = 0) -> set:
        """Return set of every possible future column reference."""
        fields = info.selected_fields
        for _ in range(level):
            fields = itertools.chain(*[field.selections for field in fields])
        return set(itertools.chain(*map(references, fields)))

    def scanner(self, info: Info) -> ds.Scanner:
        """Return scanner with only the columns necessary to proceed."""
        dataset = self.table
        schema = dataset.projected_schema if isinstance(dataset, ds.Scanner) else dataset.schema
        if isinstance(dataset, ds.Scanner):
            return dataset
        names = self.references(info) & set(schema.names)
        return dataset.scanner(columns=list(names))

    def select(self, info: Info, length: int = None) -> pa.Table:
        """Return table with only the rows and columns necessary to proceed."""
        table = self.table
        if not isinstance(table, pa.Table):
            scanner = self.scanner(info)
            table = scanner.to_table() if length is None else scanner.head(length)
        return table.select(self.references(info) & set(table.column_names))

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
        return Schema(
            names=schema.names,
            types=schema.types,
            partitioning=partitioning and partitioning.schema.names,
        )  # type: ignore

    @doc_field
    def length(self) -> Long:
        """number of rows"""
        return len(self.table) if hasattr(self.table, '__len__') else self.table.count_rows()

    @doc_field(name="column name", cast=f"cast array to {links.type}")
    def column(self, info: Info, name: str, cast: str = '') -> Column:
        """Return column of any type by name.

        This is typically only needed for aliased columns added by `apply` or `aggregate`.
        If the column is in the schema, `columns` can be used instead.
        """
        table = self.select(info)
        return Column.cast(table[name].cast(cast) if cast else table[name])

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
        table = None
        if not isinstance(self.table, pa.Table) and set(aggs) <= Agg.associatives:
            scanner = self.scanner(info)
            if lists.isdisjoint(scanner.projected_schema.names):
                table = T.map_batch(scanner, T.group, *by, counts=counts, **aggs)
                if counts:
                    aggs.setdefault('sum', []).append(Agg(counts))
                    counts = ''
        table = self.select(info) if table is None else table
        aggs['list'] = list(map(Agg, lists & set(table.column_names)))
        return type(self)(T.group(table, *by, counts=counts, **aggs))

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
            ((func, value),) = funcs.pop(name, {'ne': None}).items()
            predicates[name] = (Diff.predicates[func],)
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
        base64: List[Base64Function] = [],
        boolean: List[BooleanFunction] = [],
        date: List[DateFunction] = [],
        datetime: List[DateTimeFunction] = [],
        decimal: List[DecimalFunction] = [],
        duration: List[DurationFunction] = [],
        float: List[FloatFunction] = [],
        int: List[IntFunction] = [],
        long: List[LongFunction] = [],
        list: List[ListFunction] = [],
        string: List[StringFunction] = [],
        struct: List[StructFunction] = [],
        time: List[TimeFunction] = [],
    ) -> 'Dataset':
        """Return view of table with functions applied across columns.

        If no alias is provided, the column is replaced and should be of the same type.
        If an alias is provided, a column is added and may be referenced in the `column` field,
        in filter `predicates`, and in the `by` arguments of grouping and sorting.
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
        args = boolean, date, datetime, decimal, duration, float, long, int, string, struct, time
        for value in map(dict, itertools.chain(base64, *args)):
            for func, field in value.items():
                name, args, kwargs = field.serialize(table)
                columns[name] = C.call(getattr(pc, func, C.digitize), *args, **kwargs)
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
        self, info: Info, filter: Expression = {}, columns: List[Expression] = []  # type: ignore
    ) -> 'Dataset':
        """Select rows and project columns without memory usage."""
        dataset = ds.dataset(self.table) if isinstance(self.table, pa.Table) else self.table
        schema = dataset.projected_schema if isinstance(dataset, ds.Scanner) else dataset.schema
        names = self.references(info, level=1) & set(schema.names)
        projection = {name: ds.field(name) for name in names}
        projection.update({col.alias or '.'.join(col.name): col.to_arrow() for col in columns})
        if '' in projection:
            raise ValueError("projected columns need a name or alias")
        if isinstance(dataset, ds.Dataset):
            return type(self)(dataset.scanner(filter=filter.to_arrow(), columns=projection))
        scanner = ds.Scanner.from_batches(
            dataset.to_batches(), schema, filter=filter.to_arrow(), columns=projection
        )  # one-shot scanner can't be reused
        fields = selections(*info.selected_fields)
        return type(self)(scanner.to_table() if len(fields) > 1 else scanner)
