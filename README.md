[![image](https://img.shields.io/pypi/v/graphique.svg)](https://pypi.org/project/graphique/)
![image](https://img.shields.io/pypi/pyversions/graphique.svg)
[![image](https://pepy.tech/badge/graphique)](https://pepy.tech/project/graphique)
![image](https://img.shields.io/pypi/status/graphique.svg)
[![build](https://github.com/coady/graphique/actions/workflows/build.yml/badge.svg)](https://github.com/coady/graphique/actions/workflows/build.yml)
[![image](https://codecov.io/gh/coady/graphique/branch/main/graph/badge.svg)](https://codecov.io/gh/coady/graphique/)
[![CodeQL](https://github.com/coady/graphique/actions/workflows/github-code-scanning/codeql/badge.svg)](https://github.com/coady/graphique/actions/workflows/github-code-scanning/codeql)
[![CodSpeed Badge](https://img.shields.io/endpoint?url=https://codspeed.io/badge.json)](https://codspeed.io/coady/graphique)
[![image](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![image](https://mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)

[GraphQL](https://graphql.org) service for [arrow](https://arrow.apache.org) tables and [parquet](https://parquet.apache.org) data sets. The schema for a query API is derived automatically.

## Usage
```console
% env PARQUET_PATH=... uvicorn graphique.service:app
```

Open http://localhost:8000/ to try out the API in [GraphiQL](https://github.com/graphql/graphiql/tree/main/packages/graphiql#readme). There is a test fixture at `./tests/fixtures/zipcodes.parquet`.

```console
% env PARQUET_PATH=... strawberry export-schema graphique.service:app.schema
```
outputs the graphql schema for a parquet data set.

### Configuration
Graphique uses [Starlette's config](https://www.starlette.io/config/): in environment variables or a `.env` file. Config variables are used as input to a [parquet dataset](https://arrow.apache.org/docs/python/dataset.html).

* PARQUET_PATH: path to the parquet directory or file
* FEDERATED = '': field name to extend type `Query` with a federated `Table` 
* DEBUG = False: run service in debug mode, which includes metrics
* COLUMNS = None: list of names, or mapping of aliases, of columns to select
* FILTERS = None: json `filter` query for which rows to read at startup

For more options create a custom [ASGI](https://asgi.readthedocs.io/en/latest/index.html) app. Call graphique's `GraphQL` on an arrow [Dataset](https://arrow.apache.org/docs/python/api/dataset.html), [Scanner](https://arrow.apache.org/docs/python/generated/pyarrow.dataset.Scanner.html), or [Table](https://arrow.apache.org/docs/python/generated/pyarrow.Table.html). The GraphQL `Table` type will be the root Query type.

Supply a mapping of names to datasets for multiple roots, and to enable federation.

```python
import pyarrow.dataset as ds
from graphique import GraphQL

app = GraphQL(ds.dataset(...))  # Table is root query type
app = GraphQL.federated({<name>: ds.dataset(...), ...}, keys={...})  # Tables on federated fields
```

Start like any ASGI app.

```console
uvicorn <module>:app
```

Configuration options exist to provide a convenient no-code solution, but are subject to change in the future. Using a custom app is recommended for production usage.

### API
#### types
* `Dataset`: interface for an arrow dataset, scanner, or table.
* `Table`: implements the `Dataset` interface. Adds typed `row`, `columns`, and `filter` fields from introspecting the schema.
* `Column`: interface for an arrow column (a.k.a. ChunkedArray). Each arrow data type has a corresponding column implementation: Boolean, Int, Long, Float, Decimal, Date, Datetime, Time, Duration, Base64, String, List, Struct. All columns have a `values` field for their list of scalars. Additional fields vary by type.
* `Row`: scalar fields. Arrow tables are column-oriented, and graphique encourages that usage for performance. A single `row` field is provided for convenience, but a field for a list of rows is not. Requesting parallel columns is far more efficient.

#### selection
* `slice`: contiguous selection of rows
* `filter`: select rows with simple predicates
* `scan`: select rows and project columns with expressions

#### projection
* `columns`: provides a field for every `Column` in the schema
* `column`: access a column of any type by name
* `row`: provides a field for each scalar of a single row
* `apply`: transform columns by applying a function
* `join`: join tables by key columns

#### aggregation
* `group`: group by given columns, and aggregate the others
* `runs`: partition on adjacent values in given columns, transforming the others into list columns
* `tables`: return a list of tables by splitting on the scalars in list columns
* `flatten`: flatten list columns with repeated scalars

#### ordering
* `sort`: sort table by given columns
* `rank`: select rows with smallest or largest values

### Performance
Graphique relies on native [PyArrow](https://arrow.apache.org/docs/python/index.html) routines wherever possible. Otherwise it falls back to using [NumPy](https://numpy.org/doc/stable/) or custom optimizations.

By default, datasets are read on-demand, with only the necessary rows and columns scanned. Although graphique is a running service, [parquet is performant](https://arrow.apache.org/docs/python/generated/pyarrow.dataset.Dataset.html) at reading a subset of data. Optionally specify `FILTERS` in the json `filter` format to read a subset of rows at startup, trading-off memory for latency. An empty filter (`{}`) will read the whole table.

Specifying `COLUMNS` will limit memory usage when reading at startup (`FILTERS`). There is little speed difference as unused columns are inherently ignored. Optional aliasing can also be used for camel casing.

If index columns are detected in the schema metadata, then an initial `filter` will also attempt a binary search on tables.

## Installation
```console
% pip install graphique[server]
```

## Dependencies
* pyarrow
* strawberry-graphql[asgi,cli]
* uvicorn (or other [ASGI server](https://asgi.readthedocs.io/en/latest/implementations.html))

## Tests
100% branch coverage.

```console
% pytest [--cov]
```
