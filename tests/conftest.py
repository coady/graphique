import os
import sys
from importlib import metadata
from pathlib import Path

import pyarrow.dataset as ds
import pytest

fixtures = Path(__file__).parent / 'fixtures'


def pytest_report_header(config):
    names = 'ibis-framework', 'strawberry-graphql', 'duckdb', 'pyarrow'
    return [f'{name}: {metadata.version(name)}' for name in names]


class TestClient:
    def __init__(self, app):
        self.app = app

    def execute(self, query):
        result = self.app.schema.execute_sync(query, root_value=self.app.root_value)
        for error in result.errors or []:
            raise ValueError(error)
        return result.data


def load(path, **vars):
    os.environ.update(vars, PARQUET_PATH=str(fixtures / path))
    sys.modules.pop('graphique.service', None)
    from graphique.service import app

    for var in vars:
        del os.environ[var]
    return app


@pytest.fixture(scope='module')
def dataset():
    return ds.dataset(fixtures / 'zipcodes.parquet')


@pytest.fixture(scope='module')
def client():
    return TestClient(load('zipcodes.parquet'))


@pytest.fixture(scope='module')
def dsclient(request):
    return TestClient(load('partitioned'))


@pytest.fixture(scope='module')
def fedclient():
    from .federated import app

    return TestClient(app)


@pytest.fixture(scope='module')
def executor():
    return TestClient(load('alltypes.parquet')).execute
