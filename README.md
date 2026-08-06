[![image](https://img.shields.io/pypi/v/graphique.svg)](https://pypi.org/project/graphique/)
![image](https://img.shields.io/pypi/pyversions/graphique.svg)
[![image](https://pepy.tech/badge/graphique)](https://pepy.tech/project/graphique)
![image](https://img.shields.io/pypi/status/graphique.svg)
[![build](https://github.com/coady/graphique/actions/workflows/build.yml/badge.svg)](https://github.com/coady/graphique/actions/workflows/build.yml)
[![image](https://codecov.io/gh/coady/graphique/branch/main/graph/badge.svg)](https://codecov.io/gh/coady/graphique/)
[![CodeQL](https://github.com/coady/graphique/actions/workflows/github-code-scanning/codeql/badge.svg)](https://github.com/coady/graphique/actions/workflows/github-code-scanning/codeql)
[![image](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![ty](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json)](https://github.com/astral-sh/ty)

[GraphQL](https://graphql.org) service for [ibis](https://ibis-project.org) tables. Ibis supports [20+ backends](https://ibis-project.org/why#how-does-ibis-work) — DuckDB, PostgreSQL, Polars, BigQuery, etc. — so the same query API works across local files and remote databases. The schema is derived automatically.

[Parquet](https://parquet.apache.org) datasets are also supported as a root source, with custom optimizations for partitions. As of version 2, execution is based on ibis (default backend: [DuckDB](https://duckdb.org)).

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
* NAME = '': GraphQL field on `Query`; defaults to root type
* COLUMNS = None: list of names, or mapping of aliases, of columns to select

Configuration options exist to provide a convenient no-code solution, but are subject to change in the future. Using a custom app is recommended for production usage.

### App
For more options create a custom [ASGI](https://asgi.readthedocs.io/en/latest/index.html) app. Call graphique's `GraphQL` on an ibis [Table](https://ibis-project.org/reference/expression-tables) or parquet [Dataset](https://arrow.apache.org/docs/python/dataset.html).
Use a `Query` type with dataset attributes for multiple roots, and to enable federation.

```python
import ibis
from graphique import GraphQL, typed

# any ibis backend: DuckDB, PostgreSQL, Polars, BigQuery, ...
source = ibis.read_(...)  # or `ibis.connect(...).table(...)` or `pyarrow.dataset.dataset(...)`
# apply initial projections or filters to `source`
app = GraphQL(source)  # Table is root query type


# multiple named fields, with optional federation keys
class Query:
    name = source  # or `typed(source, name, keys=...)`


app = GraphQL(Query)
```

Start like any ASGI app.

```console
uvicorn <module>:app
```

### API
#### types
* `Dataset`: interface for an ibis table or parquet dataset.
* `Table`: implements the `Dataset` interface. Adds typed `row`, `columns`, and `filter` fields from introspecting the schema.
* `Column`: interface for an ibis column. Each data type has a corresponding column implementation: Boolean, Int, BigInt, Float, Decimal, Date, Datetime, Time, Duration, Base64, String, Array, Struct. All columns have a `values` field for their list of scalars. Additional fields vary by type.
* `Row`: scalar fields. Tables are column-oriented, and graphique encourages that usage for performance. A single `row` field is provided for convenience, but a field for a list of rows is not. Requesting parallel columns is far more efficient.

#### selection
* `slice`: contiguous selection of rows
* `filter`: select rows by predicates
* `join`, `asofJoin`, `crossJoin`: join tables by key columns
* `difference`, `intersect`, `union`: set operations on tables
* `take`: rows by index
* `dropNull`: remove rows with nulls

#### projection
* `project`: project columns with expressions
* `columns`: provides a field for every `Column` in the schema
* `column`: access a column of any type by name
* `row`: provides a field for each scalar of a single row
* `cast`: cast column types
* `unpack`: project struct fields
* `fillNull`: fill null values

#### aggregation
* `group`: group by given columns, and aggregate the others
* `distinct`: group with all columns
* `runs`: group by adjacency
* `unnest`: unnest an array column
* `count`, `any`: number of rows

#### ordering
* `order`: sort table by given columns
* `first`: sort and filter by rank

#### reflection
* `type`: type of data source
* `schema`: field names and types
* `optional`: nullable for errors
* `toSql`: compiles SQL query

### Performance
Performance is dependent on the Ibis backend, which defaults to DuckDB. There are no internal Python loops. Scalars do not become Python types until serialized. Table fields are lazily evaluated up until scalars are reached, and automatically cached as needed for multiple fields.

[PyArrow](https://arrow.apache.org/docs/python/) is also used for partitioned dataset optimizations. `python -m graphique.partition` is a command-line script provided in `graphique[cli]`, for out-of-core partitioning.

## Installation
```console
pip install graphique[server,cli]
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
