[![image](https://img.shields.io/pypi/v/graphique.svg)](https://pypi.org/project/graphique/)
![image](https://img.shields.io/pypi/pyversions/graphique.svg)
[![image](https://pepy.tech/badge/graphique)](https://pepy.tech/project/graphique)
![image](https://img.shields.io/pypi/status/graphique.svg)
[![image](https://api.travis-ci.com/coady/graphique.svg)](https://travis-ci.com/coady/graphique)
[![image](https://img.shields.io/codecov/c/github/coady/graphique.svg)](https://codecov.io/github/coady/graphique)
[![image](https://readthedocs.org/projects/graphique/badge)](https://graphique.readthedocs.io)
[![image](https://requires.io/github/coady/graphique/requirements.svg)](https://requires.io/github/coady/graphique/requirements/)
[![image](https://img.shields.io/badge/code%20style-black-000000.svg)](https://pypi.org/project/black/)
[![image](http://mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)

[GraphQL](https://graphql.org) service for [arrow](https://arrow.apache.org) tables
and [parquet](https://parquet.apache.org) files.
The schema is derived automatically.

# Usage
```console
% env PARQUET_PATH=... uvicorn graphique.service:app [--reload]
```

Open http://localhost:8000/graphql to try out the API in [GraphiQL](https://github.com/graphql/graphiql/tree/main/packages/graphiql#readme).
There is a test fixture at `./tests/fixtures/zipcodes.parquet`.

## Configuration
Graphique uses [Starlette's config](https://www.starlette.io/config/): in environment variables or a `.env` file.
Config variables are used as input to [ParquetDataset](https://arrow.apache.org/docs/python/parquet.html#reading-from-partitioned-datasets).
* COLUMNS = None
* DEBUG = False
* DICTIONARIES = None
* INDEX = None
* MMAP = True
* PARQUET_PATH

## Queries
A `Table` is the primary interface.  It has fields for filtering, sorting, and grouping.

```typescript
"""a column-oriented table"""
type Table {
  """number of rows"""
  length: Long!

  """fields for each column"""
  columns: Columns!

  """Return scalar values at index."""
  row(index: Long! = 0): Row!

  """Return table slice."""
  slice(offset: Long! = 0, length: Long): Table!

  """
  Return tables grouped by columns, with stable ordering.
          `length` is the maximum number of tables to return.
          `count` filters and sorts tables based on the number of rows within each table.
  """
  group(by: [String!]!, reverse: Boolean! = false, length: Long, count: LongReduce): [Table!]!

  """
  Return table of first or last occurrences grouped by columns, with stable ordering.
  """
  unique(by: [String!]!, reverse: Boolean! = false): Table!

  """Return table slice sorted by specified columns."""
  sort(by: [String!]!, reverse: Boolean! = false, length: Long): Table!

  """Return table with minimum values per column."""
  min(by: [String!]!): Table!

  """Return table with maximum values per column."""
  max(by: [String!]!): Table!

  """
  Return table with rows which match all (by default) queries.
          `invert` optionally excludes matching rows.
          `reduce` is the binary operator to combine filters; within a column all predicates must match.
  """
  filter(query: Filters!, invert: Boolean! = false, reduce: Operator! = AND): Table!
```

## Performance
Graphique relies on native [pyarrow](https://arrow.apache.org/docs/python/index.html) routines wherever possible.
Otherwise it falls back to using [NumPy](https://numpy.org/doc/stable/), with zero-copy views.
Graphique also has custom optimizations for grouping, dictionary-encoded arrays, and chunked arrays.

Specifying an `INDEX` of columns indicates the table is sorted, and enables a binary search interface.
```typescript
  """
  Return table with matching values for compound `index`.
          Queries must be a prefix of the `index`.
          Only one non-equal query is allowed, and applied last.
  """
  search(...): Table!
```

# Installation
```console
% pip install graphique
```

# Dependencies
* pyarrow >=1
* strawberry-graphql >=0.30
* pytz (optional timestamp support)

# Tests
100% branch coverage.

```console
% pytest [--cov]
```
