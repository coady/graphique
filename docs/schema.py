"""
Output graphql schema from a parquet data set.
"""
import argparse
import os
import strawberry

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('path', help="path to parquet data set")

if __name__ == '__main__':
    os.environ['PARQUET_PATH'] = parser.parse_args().path
    from graphique import service

    schema = strawberry.Schema(query=service.Query)
    print(strawberry.printer.print_schema(schema))
