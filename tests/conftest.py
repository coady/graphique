import collections
import sys
from importlib import metadata
from pathlib import Path

import ibis
import pyarrow.dataset as ds
import pytest

from graphique import GraphQL

fixtures = Path(__file__).parent / "fixtures"


def pytest_report_header(config):
    names = "ibis-framework", "strawberry-graphql", "duckdb", "pyarrow"
    return [f"{name}: {metadata.version(name)}" for name in names]


class unordered(collections.Counter):
    def __eq__(self, other):
        return dict(self) == collections.Counter(other)


class TestClient(GraphQL):
    def execute(self, query):
        result = self.schema.execute_sync(query, root_value=self.root_value)
        for error in result.errors or []:
            raise ValueError(error)
        return result.data


def load(path, **vars):
    vars["PARQUET_PATH"] = str(fixtures / path)
    with pytest.MonkeyPatch.context() as mp:
        for key in vars:
            mp.setenv(key, vars[key])
        mp.delitem(sys.modules, "graphique.service", raising=False)
        from graphique.service import app
    return app


@pytest.fixture(scope="module")
def dataset():
    return ds.dataset(fixtures / "zipcodes.parquet")


@pytest.fixture(scope="module")
def partitioned():
    return ds.dataset(fixtures / "partitioned", partitioning="hive")


@pytest.fixture(scope="module")
def client():
    return TestClient(ibis.read_parquet(fixtures / "zipcodes.parquet", table_name="zipcodes"))


@pytest.fixture(scope="module")
def dsclient(partitioned):
    return TestClient(partitioned)


@pytest.fixture(scope="module")
def fedclient():
    from .federated import app

    return app


@pytest.fixture(scope="module")
def executor():
    return TestClient(ds.dataset(fixtures / "alltypes.parquet")).execute
