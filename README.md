[![image](https://img.shields.io/travis/coady/graphique.svg)](https://travis-ci.com/coady/graphique)
[![image](https://readthedocs.org/projects/graphique/badge)](https://graphique.readthedocs.io)
[![image](https://img.shields.io/badge/code%20style-black-000000.svg)](https://pypi.org/project/black/)
[![image](http://mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)

[GraphQL](https://graphql.org) service for [arrow](https://arrow.apache.org) tables
and [parquet](https://parquet.apache.org) files.
The schema is derived automatically.

# Usage
```console
% PARQUET_PATH=... uvicorn graphique:app [--reload]
```

Open http://localhost:8000/graphql.

# Installation
```console
% pip install graphique
```

# Tests
100% branch coverage.

```console
% pytest [--cov]
```
