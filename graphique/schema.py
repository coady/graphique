"""
Output graphql schema from a parquet data set.
"""
import argparse
import strawberry
from starlette.config import environ

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('parquet_path', help="path to parquet data set")
parser.add_argument('--index', help="partition keys or sorted composite index", default='')
parser.add_argument('--federated', help="field name for federated Table", default='')

if __name__ == '__main__':
    args = parser.parse_args()
    environ.update(PARQUET_PATH=args.parquet_path, INDEX=args.index, FEDERATED=args.federated)
    from graphique import service

    print(strawberry.printer.print_schema(service.app.schema))
