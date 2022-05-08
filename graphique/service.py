"""
GraphQL service and top-level resolvers.
"""
from typing import List, Optional, no_type_check
import pyarrow as pa
import pyarrow.dataset as ds
import strawberry
from strawberry import UNSET
from strawberry.utils.str_converters import to_camel_case
from .core import Column as C, Table as T
from .inputs import Filters, Input, Query as QueryInput
from .middleware import Dataset, GraphQL, filter_expression
from .models import Column, doc_field, selections
from .scalars import Long, Operator, type_map
from .settings import COLUMNS, DEBUG, DICTIONARIES, FEDERATED, FILTERS, INDEX, PARQUET_PATH

format = ds.ParquetFileFormat(read_options={'dictionary_columns': DICTIONARIES})
table = dataset = ds.dataset(PARQUET_PATH, format=format, partitioning='hive')
indexed = list(map(to_camel_case, INDEX))
types = {to_camel_case(field.name): type_map[C.scalar_type(field).id] for field in dataset.schema}


@strawberry.type(description="fields for each column")
class Columns:
    __annotations__ = {name: Column.type_map[types[name]] for name in types}  # type: ignore
    locals().update(dict.fromkeys(__annotations__))


@strawberry.type(description="scalar fields")
class Row:
    __annotations__ = {
        name: Optional[Column if types[name] is list else types[name]]
        for name in types
        if types[name] is not dict
    }
    locals().update(dict.fromkeys(__annotations__))


@strawberry.input(description="predicates for each column")
class Queries(Input):
    __annotations__ = QueryInput.annotations(types)
    locals().update(dict.fromkeys(__annotations__, UNSET))


@strawberry.type(description="a column-oriented table")
class Table(Dataset):
    @doc_field
    def columns(self, info) -> Columns:
        """fields for each column"""
        table = self.select(info)
        columns = {name: Columns.__annotations__[name](table[name]) for name in table.column_names}
        return Columns(**columns)

    @doc_field
    def row(self, info, index: Long = 0) -> Row:
        """Return scalar values at index."""
        table = self.select(info, index + 1 if index >= 0 else None)
        row = {}
        for name in table.column_names:
            scalar = table[name][index]
            row[name] = (
                Column.fromscalar(scalar) if isinstance(scalar, pa.ListScalar) else scalar.as_py()
            )
        return Row(**row)

    @doc_field(
        query="simple queries by column",
        on="extended filters on columns organized by type",
        invert="optionally exclude matching rows",
        reduce="binary operator to combine filters; within a filter all predicates must match",
    )
    @no_type_check
    def filter(
        self,
        info,
        query: Queries = {},
        on: Filters = {},
        invert: bool = False,
        reduce: Operator = Operator.AND,
    ) -> 'Table':
        """Return table with rows which match all (by default) queries.

        List columns apply their respective filters to the scalar values within lists.
        All referenced list columns must have the same lengths.
        """
        fields = selections(*info.selected_fields)
        if reduce.value in ('and', 'or') and dict(query):
            scanner = self.scanner(info, dict(query), invert=invert, reduce=reduce.value)
            oneshot = isinstance(self.table, ds.Scanner) and len(fields) > 1
            query, self = {}, type(self)(scanner.to_table() if oneshot else scanner)
        filters = dict(on)
        for name, value in dict(query).items():
            value['name'] = name
            filters.setdefault('', []).append(value)
        if not any(filters.values()):
            return self
        return Dataset.filter(self, info, filters, invert, reduce)


@strawberry.type(description="a table sorted by a composite index")
class IndexedTable(Table):
    index: List[str] = strawberry.field(default=tuple(indexed), description="the composite index")

    @QueryInput.resolve_types({name: types[name] for name in indexed})
    def search(self, info, **queries) -> Table:
        """Return table with matching values for composite `index`.

        Queries must be a prefix of the `index`.
        Only one inequality query is allowed, and must be last.
        """
        for name in queries:
            if queries[name] is None:
                raise TypeError(f"`{name}` is optional, not nullable")
        queries = {name: dict(queries[name]) for name in queries}
        if isinstance(self.table, ds.Dataset):
            return Table(self.scanner(info, queries))
        table = self.select(info)
        for name in self.index:
            if name not in queries:
                break
            query = queries.pop(name)
            if 'equal' in query:
                table = T.is_in(table, name, query.pop('equal'))
            if query and queries:
                raise ValueError(f"inequality query for {name} not last; have {queries} remaining")
            if 'not_equal' in query:
                table = T.not_equal(table, name, query['not_equal'])
            if 'is_in' in query:
                table = T.is_in(table, name, *query['is_in'])
            lower, upper = query.get('greater'), query.get('less')
            includes = {'include_lower': False, 'include_upper': False}
            if 'greater_equal' in query and (lower is None or query['greater_equal'] > lower):
                lower, includes['include_lower'] = query['greater_equal'], True
            if 'less_equal' in query and (upper is None or query['less_equal'] > upper):
                upper, includes['include_upper'] = query['less_equal'], True
            if {lower, upper} != {None}:
                table = T.range(table, name, lower, upper, **includes)
        if queries:
            raise ValueError(f"expected query for {name}; have {queries} remaining")
        return Table(table)


if COLUMNS or FILTERS:
    names = dataset.schema.names if ''.join(COLUMNS) in '*' else COLUMNS
    columns = {to_camel_case(name): ds.field(name) for name in names}
    table = dataset.to_table(columns=columns, filter=filter_expression(FILTERS))
    for name in indexed:
        assert not table[name].null_count, f"binary search requires non-null columns: {name}"
Query = IndexedTable if indexed else Table
app = GraphQL(Query(table), debug=DEBUG, federated=FEDERATED)
