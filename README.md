[![image](https://img.shields.io/pypi/v/graphique.svg)](https://pypi.org/project/graphique/)
![image](https://img.shields.io/pypi/pyversions/graphique.svg)
[![image](https://pepy.tech/badge/graphique)](https://pepy.tech/project/graphique)
![image](https://img.shields.io/pypi/status/graphique.svg)
[![image](https://api.travis-ci.com/coady/graphique.svg)](https://travis-ci.com/coady/graphique)
[![image](https://img.shields.io/codecov/c/github/coady/graphique.svg)](https://codecov.io/github/coady/graphique)
[![image](https://readthedocs.org/projects/graphique/badge)](https://graphique.readthedocs.io)
[![image](https://requires.io/github/coady/graphique/requirements.svg)](https://requires.io/github/coady/graphique/requirements/)
[![image](https://api.codeclimate.com/v1/badges/c6d16624c2c444f531c4/maintainability)](https://codeclimate.com/github/coady/graphique/maintainability)
[![image](https://img.shields.io/badge/code%20style-black-000000.svg)](https://pypi.org/project/black/)
[![image](http://mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)

[GraphQL](https://graphql.org) service for [arrow](https://arrow.apache.org) tables
and [parquet](https://parquet.apache.org) files.
The schema is derived automatically.

# Usage
```console
% PARQUET_PATH=... uvicorn graphique.service:app [--reload]
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
