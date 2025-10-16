[![image](https://img.shields.io/pypi/v/graphique.svg)](https://pypi.org/project/graphique/)
![image](https://img.shields.io/pypi/pyversions/graphique.svg)
[![image](https://pepy.tech/badge/graphique)](https://pepy.tech/project/graphique)
![image](https://img.shields.io/pypi/status/graphique.svg)
[![build](https://github.com/coady/graphique/actions/workflows/build.yml/badge.svg)](https://github.com/coady/graphique/actions/workflows/build.yml)
[![image](https://codecov.io/gh/coady/graphique/branch/main/graph/badge.svg)](https://codecov.io/gh/coady/graphique/)
[![CodeQL](https://github.com/coady/graphique/actions/workflows/github-code-scanning/codeql/badge.svg)](https://github.com/coady/graphique/actions/workflows/github-code-scanning/codeql)
[![image](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![image](https://mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)

[GraphQL](https://graphql.org) service for [ibis](https://ibis-project.org) dataframes, [arrow](https://arrow.apache.org) tables, and [parquet](https://parquet.apache.org) datasets. The schema for a query API is derived automatically.

## Version 2
When this project started, there was no out-of-core execution engine with performance comparable to [PyArrow](https://arrow.apache.org/docs/python/index.html). So it effectively included one, based on datasets and [Acero](https://arrow.apache.org/docs/python/api/acero.html).

Since then the ecosystem has grown considerably: [DuckDB](https://duckdb.org), [DataFusion](https://datafusion.apache.org), and [Ibis](https://ibis-project.org). As of version 2, graphique is based on `ibis`. It provides a common dataframe API for multiple backends, enabling graphique to also have a default but configurable backend.

Being a major version upgrade, there are incompatible changes from version 1. However the overall API remains largely the same.

## Usage
There is an example app which reads a parquet dataset.
```console
env PARQUET_PATH=... uvicorn graphique.service:app
```

Open http://localhost:8000/ to try out the API in [GraphiQL](https://github.com/graphql/graphiql/tree/main/packages/graphiql#readme). There is a test fixture at `./tests/fixtures/zipcodes.parquet`.

```console
env PARQUET_PATH=... strawberry export-schema graphique.service:app.schema
```
outputs the graphql schema.

### Configuration
The example app uses [Starlette's config](https://www.starlette.io/config/): in environment variables or a `.env` file.

* PARQUET_PATH: path to the parquet directory or file
* FEDERATED = '': field name to extend type `Query` with a federated `Table`
* METRICS = False: include timings from apollo tracing extension
* COLUMNS = None: list of names, or mapping of aliases, of columns to select
* FILTERS = None: json `filter` query for which rows to read at startup

Configuration options exist to provide a convenient no-code solution, but are subject to change in the future. Using a custom app is recommended for production usage.

### App
For more options create a custom [ASGI](https://asgi.readthedocs.io/en/latest/index.html) app. Call graphique's `GraphQL` on an ibis [Table](https://ibis-project.org/reference/expression-tables) or arrow [Dataset](https://arrow.apache.org/docs/python/api/dataset.html).
Supply a mapping of names to datasets for multiple roots, and to enable federation.

```python
import ibis
from graphique import GraphQL

source = ibis.read_*(...)  # or ibis.connect(...).table(...) or pyarrow.dataset.dataset(...)
# apply initial projections or filters to `source`
app = GraphQL(source)  # Table is root query type
app = GraphQL.federated({<name>: source, ...}, keys={<name>: [], ...})  # Tables on federated fields
```

Start like any ASGI app.

```console
uvicorn <module>:app
```

### API
#### types
* `Dataset`: interface for an ibis table or arrow dataset.
* `Table`: implements the `Dataset` interface. Adds typed `row`, `columns`, and `filter` fields from introspecting the schema.
* `Column`: interface for an ibis column. Each data type has a corresponding column implementation: Boolean, Int, BigInt, Float, Decimal, Date, Datetime, Time, Duration, Base64, String, Array, Struct. All columns have a `values` field for their list of scalars. Additional fields vary by type.
* `Row`: scalar fields. Tables are column-oriented, and graphique encourages that usage for performance. A single `row` field is provided for convenience, but a field for a list of rows is not. Requesting parallel columns is far more efficient.

#### selection
* `slice`: contiguous selection of rows
* `filter`: select rows by predicates
* `join`: join tables by key columns
* `take`: rows by index
* `dropNull`: remove rows with nulls

#### projection
* `project`: project columns with expressions
* `columns`: provides a field for every `Column` in the schema
* `column`: access a column of any type by name
* `row`: provides a field for each scalar of a single row
* `cast`: cast column types
* `fillNull`: fill null values

#### aggregation
* `group`: group by given columns, and aggregate the others
* `distinct`: group with all columns
* `runs`: provisionally group by adjacency
* `unnest`: unnest an array column
* `count`: number of rows

#### ordering
* `order`: sort table by given columns
* options `limit` and `dense`: select rows with smallest or largest values

### Performance
Performance is dependent on the [ibis backend](https://ibis-project.org/backends/duckdb), which defaults to [duckdb](https://duckdb.org/). There are no internal Python loops. Scalars do not become Python types until serialized.

[PyArrow](https://arrow.apache.org/docs/python/) is also used for partitioned dataset optimizations, and for any feature which ibis does not support. Table fields are lazily evaluated up until scalars are reached, and automatically cached as needed for multiple fields.

## Installation
```console
pip install graphique[server]
```

## Dependencies
* ibis-framework (with duckdb or other backend)
* strawberry-graphql[asgi,cli]
* pyarrow
* isodate
* uvicorn (or other [ASGI server](https://asgi.readthedocs.io/en/latest/implementations.html))

## Tests
100% branch coverage.

```console
pytest [--cov]
```
