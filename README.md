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

  """Return table grouped by columns, with stable ordering."""
  group(
    by: [String!]!

    """return groups in reversed stable order"""
    reverse: Boolean! = false

    """maximum number of groups to return"""
    length: Long = null

    """optionally include counts in an aliased column"""
    count: String! = ""
  ): Table!

  """
  Return table partitioned by discrete differences of the values.
          Differs from `group` by relying on adjacency, and is typically faster.
  """
  partition(
    by: [String!]!

    """
    predicates defaulting to `not_equal`; scalars are compared to the adjacent difference
    """
    diffs: Diffs = null

    """optionally include counts in an aliased column"""
    count: String! = ""
  ): Table!

  """
  Return table of first or last occurrences grouped by columns, with stable ordering.
          Faster than `group` when only scalars are needed.
  """
  unique(
    by: [String!]!

    """return last occurrences in reversed order"""
    reverse: Boolean! = false

    """maximum number of rows to return"""
    length: Long = null

    """optionally include counts in an aliased column"""
    count: String! = ""
  ): Table!

  """Return table slice sorted by specified columns."""
  sort(
    by: [String!]!

    """descending stable order"""
    reverse: Boolean! = false

    """
    maximum number of rows to return; may be significantly faster on a single column
    """
    length: Long = null
  ): Table!

  """Return table with minimum values per column."""
  min(by: [String!]!): Table!

  """Return table with maximum values per column."""
  max(by: [String!]!): Table!

  """
  Return table with rows which match all (by default) queries.
          List columns apply their respective filters to their own scalar values.
  """
  filter(
    """filters organized by column"""
    query: Filters = null

    """optionally exclude matching rows"""
    invert: Boolean! = false

    """
    binary operator to combine filters; within a column all predicates must match
    """
    reduce: Operator! = AND

    """
    additional filters for columns of unknown types, as the result of `apply`
    """
    predicates: [Filter!]! = []
  ): Table!
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
