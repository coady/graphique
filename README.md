[![image](https://img.shields.io/pypi/v/graphique.svg)](https://pypi.org/project/graphique/)
![image](https://img.shields.io/pypi/pyversions/graphique.svg)
[![image](https://pepy.tech/badge/graphique)](https://pepy.tech/project/graphique)
![image](https://img.shields.io/pypi/status/graphique.svg)
[![image](https://github.com/coady/graphique/workflows/build/badge.svg)](https://github.com/coady/graphique/actions)
[![image](https://codecov.io/gh/coady/graphique/branch/main/graph/badge.svg)](https://codecov.io/gh/coady/graphique/)
[![image](https://github.com/coady/graphique/workflows/codeql/badge.svg)](https://github.com/coady/graphique/security/code-scanning)
[![image](https://img.shields.io/badge/code%20style-black-000000.svg)](https://pypi.org/project/black/)
[![image](http://mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)

[GraphQL](https://graphql.org) service for [arrow](https://arrow.apache.org) tables and [parquet](https://parquet.apache.org) data sets. The schema is derived automatically.

## Usage
```console
% env PARQUET_PATH=... uvicorn graphique.service:app
```

Open http://localhost:8000/graphql to try out the API in [GraphiQL](https://github.com/graphql/graphiql/tree/main/packages/graphiql#readme). There is a test fixture at `./tests/fixtures/zipcodes.parquet`.

### Configuration
Graphique uses [Starlette's config](https://www.starlette.io/config/): in environment variables or a `.env` file. Config variables are used as input to [ParquetDataset](https://arrow.apache.org/docs/python/parquet.html#reading-from-partitioned-datasets).

* COLUMNS = None
* DEBUG = False
* DICTIONARIES = None
* INDEX = None
* MMAP = True
* PARQUET_PATH

### Queries
A `Table` is the primary interface.  It has fields for filtering, sorting, and grouping.

```graphql
"""a column-oriented table"""
type Table {
  """number of rows"""
  length: Long!

  """fields for each column"""
  columns: Columns!

  """
  Return column of any type by name.
          This is typically only needed for aliased columns added by `apply` or `aggregate`.
          If the column is in the schema, `columns` can be used instead.
  """
  column(name: String!): Column!

  """Return scalar values at index."""
  row(index: Long! = 0): Row!

  """Return table slice."""
  slice(offset: Long! = 0, length: Long = null): Table!

  """
  Return table grouped by columns, with stable ordering.
          Optionally include counts in an aliased column.
          `length` is the maximum number of groups to return.
  """
  group(by: [String!]!, reverse: Boolean! = false, length: Long = null, count: String! = ""): Table!

  """
  Return table partitioned by discrete differences of the values.
          Optionally include counts in an aliased column.
          Differs from `group` by relying on adjacency, and is typically faster.
  """
  partition(by: [String!]!, diffs: Diffs = null, count: String! = ""): Table!

  """
  Return table of first or last occurrences grouped by columns, with stable ordering.
          Optionally include counts in an aliased column.
          Faster than `group` when only scalars are needed.
  """
  unique(by: [String!]!, reverse: Boolean! = false, length: Long = null, count: String! = ""): Table!

  """Return table slice sorted by specified columns."""
  sort(by: [String!]!, reverse: Boolean! = false, length: Long = null): Table!

  """Return table with minimum values per column."""
  min(by: [String!]!): Table!

  """Return table with maximum values per column."""
  max(by: [String!]!): Table!

  """
  Return table with rows which match all (by default) queries.
          `invert` optionally excludes matching rows.
          `reduce` is the binary operator to combine filters; within a column all predicates must match.
          `predicates` are additional filters for column of unknown types, as the result of `apply`.
          List columns apply their respective filters to their own scalar values.
  """
  filter(query: Filters = null, invert: Boolean! = false, reduce: Operator! = AND, predicates: [Filter!]! = []): Table!

  """
  Return view of table with functions applied across columns.
          If no alias is provided, the column is replaced and should be of the same type.
          If an alias is provided, a column is added and may be referenced in the `column` interface,
          in filter `predicates`, and in the `by` arguments of grouping and sorting.
  """
  apply(...): Table!

  """
  Return a list of tables by splitting list columns, typically used after grouping.
          At least one list column must be referenced, and all list columns must have the same shape.
  """
  tables: [Table!]!

  """
  Return single table with aggregate functions applied to columns.
          The grouping keys are automatically included.
          Any remaining columns referenced in fields are kept as list columns.
          Columns which are aliased or change type can be accessed by the `column` field.
  """
  aggregate(count: [Field!]! = [], first: [Field!]! = [], last: [Field!]! = [], min: [Field!]! = [], max: [Field!]! = [], sum: [Field!]! = [], mean: [Field!]! = [], any: [Field!]! = [], all: [Field!]! = [], unique: [Field!]! = []): Table!
}
```

### Performance
Graphique relies on native [pyarrow](https://arrow.apache.org/docs/python/index.html) routines wherever possible. Otherwise it falls back to using [NumPy](https://numpy.org/doc/stable/), with zero-copy views. Graphique also has custom optimizations for grouping, dictionary-encoded arrays, and chunked arrays.

Specifying an `INDEX` of columns indicates the table is sorted, and enables a binary search interface.
```graphql
  """
  Return table with matching values for compound `index`.
          Queries must be a prefix of the `index`.
          Only one non-equal query is allowed, and applied last.
  """
  search(...): Table!
```

## Installation
```console
% pip install graphique
```

## Dependencies
* pyarrow >=3
* strawberry-graphql >=0.54
* uvicorn (or other [ASGI server](https://asgi.readthedocs.io/en/latest/implementations.html))
* pytz (optional timestamp support)

## Tests
100% branch coverage.

```console
% pytest [--cov]
```

## Changes
dev

* `sort` updated to use new native routines
* `partition` tables by adjacent values and differences
* `filter` supports unknown column types using tagged union pattern
* `Groups` replaced with `Table.tables` and `Table.aggregate` fields

0.3

* Pyarrow >=3 required
* `any` and `all` fields
* String column `split` field

0.2

* `ListColumn` and `StructColumn` types
* `Groups` type with `aggregate` field
* `group` and `unique` optimized
* pyarrow >= 2 required
* Statistical fields: `mode`, `stddev`, `variance`
* `is_in`, `min`, and `max` optimized
