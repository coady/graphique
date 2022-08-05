[![image](https://img.shields.io/pypi/v/graphique.svg)](https://pypi.org/project/graphique/)
![image](https://img.shields.io/pypi/pyversions/graphique.svg)
[![image](https://pepy.tech/badge/graphique)](https://pepy.tech/project/graphique)
![image](https://img.shields.io/pypi/status/graphique.svg)
[![image](https://github.com/coady/graphique/workflows/build/badge.svg)](https://github.com/coady/graphique/actions)
[![image](https://codecov.io/gh/coady/graphique/branch/main/graph/badge.svg)](https://codecov.io/gh/coady/graphique/)
[![image](https://github.com/coady/graphique/workflows/codeql/badge.svg)](https://github.com/coady/graphique/security/code-scanning)
[![image](https://img.shields.io/badge/code%20style-black-000000.svg)](https://pypi.org/project/black/)
[![image](http://mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)

[GraphQL](https://graphql.org) service for [arrow](https://arrow.apache.org) tables and [parquet](https://parquet.apache.org) data sets. The schema for a query API is derived automatically.

## Usage
```console
% env PARQUET_PATH=... uvicorn graphique.service:app
```

Open http://localhost:8000/graphql to try out the API in [GraphiQL](https://github.com/graphql/graphiql/tree/main/packages/graphiql#readme). There is a test fixture at `./tests/fixtures/zipcodes.parquet`.

```console
% python3 -m graphique.schema ...
```
outputs the graphql schema for a parquet data set.

### Configuration
Graphique uses [Starlette's config](https://www.starlette.io/config/): in environment variables or a `.env` file. Config variables are used as input to [parquet dataset](https://arrow.apache.org/docs/python/dataset.html).

* PARQUET_PATH: path to the parquet directory or file
* INDEX = []: partition keys or names of columns which represent a sorted composite index 
* FEDERATED = '': field name to extend type `Query` with a federated `Table` 
* DEBUG = False: run service in debug mode, which includes timing
* DICTIONARIES = []: names of columns to read as dictionaries
* COLUMNS = []: names of columns to read at startup; `*` indicates all
* FILTERS = {}: json `Queries` for which rows to read at startup

### API
#### types
* `Table`: an arrow Table; the primary interface.
* `Column`: an arrow Column (a.k.a. ChunkedArray). Each arrow data type has a corresponding column implementation: Boolean, Int, Long, Float, Decimal, Date, DateTime, Time, Duration, Base64, String, List, Struct. All columns have a `values` field for their list of scalars. Additional fields vary by type.
* `Row`: scalar fields. Arrow tables are column-oriented, and graphique encourages that usage for performance. A single `row` field is provided for convenience, but a field for a list of rows is not. Requesting parallel columns is far more efficient.

#### selection
* `slice`: contiguous selection of rows
* `search`: binary search if the table is sorted, i.e., provides an index
* `filter`: select rows from predicate functions

#### projection
* `columns`: provides a field for every `Column` in the schema
* `column`: access a column of any type by name
* `row`: provides a field for each scalar of a single row
* `apply`: transform columns by applying a function

#### aggregation
* `group`: group by given columns, transforming the others into list columns
* `partition`: partition on adjacent values in given columns, transforming the others into list columns
* `aggregate`: apply reduce functions to list columns
* `tables`: return a list of tables by splitting on the scalars in list columns

#### ordering
* `sort`: sort table by given columns
* `min`: select rows with smallest values
* `max`: select rows with largest values

### Performance
Graphique relies on native [PyArrow](https://arrow.apache.org/docs/python/index.html) routines wherever possible. Otherwise it falls back to using [NumPy](https://numpy.org/doc/stable/) or custom optimizations.

By default, datasets are read on-demand, with only the necessary columns selected. Additionally `filter(query: ...)` is optimized to filter rows while reading the dataset. Although graphique is a running service, [parquet is performant](https://arrow.apache.org/docs/python/generated/pyarrow.dataset.Dataset.html) at reading a subset of data. Optionally specify `COLUMNS` to read a subset of columns (or `*`) at startup, trading-off memory for latency. Similarly specify `FILTERS` in the json format of the `Query` input type to read a subset of rows at startup.

Specifying an `INDEX` indicates the table is sorted, and enables the binary `search` field. Specifying just `INDEX` without reading (`FILTERS` or `COLUMNS`) is allowed but only recommended if it corresponds to the partition keys. In that case, `search(...)` is functionally equivalent to `filter(query: ...)`.

## Installation
```console
% pip install graphique[server]
```

## Dependencies
* pyarrow >=9
* strawberry-graphql[asgi] >=0.116.4
* uvicorn (or other [ASGI server](https://asgi.readthedocs.io/en/latest/implementations.html))

## Tests
100% branch coverage.

```console
% pytest [--cov]
```

## Changes
0.9

* Pyarrow >=9 required
* Multi-directional sorting
* Removed unnecessary interfaces
* Filtering has stricter typing

0.8

* Pyarrow >=8 required
* Grouping and aggregation integrated
* `AbstractTable` interface renamed to `Dataset`
* `Binary` scalar renamed to `Base64`

0.7

* Pyarrow >=7 required
* `FILTERS` use query syntax and trigger reading the dataset
* `FEDERATED` field configuration
* List columns support sorting and filtering
* Group by and aggregate optimizations
* Dataset scanning

0.6

* Pyarrow >=6 required
* Group by optimized and replaced `unique` field
* Dictionary related optimizations
* Null consistency with arrow `count` functions

0.5

* Pyarrow >=5 required
* Stricter validation of inputs
* Columns can be cast to another arrow data type
* Grouping uses large list arrays with 64-bit counts
* Datasets are read on-demand or optionally at startup

0.4

* Pyarrow >=4 required
* `sort` updated to use new native routines
* `partition` tables by adjacent values and differences
* `filter` supports unknown column types using tagged union pattern
* `Groups` replaced with `Table.tables` and `Table.aggregate` fields
* Tagged unions used for `filter`, `apply`, and `partition` functions

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
