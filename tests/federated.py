from pathlib import Path

import ibis
import pyarrow.dataset as ds

from graphique import GraphQL

fixtures = Path(__file__).parent / 'fixtures'
dataset = ds.dataset(fixtures / 'zipcodes.parquet')
table = ibis.read_parquet(fixtures / 'zipcodes.parquet')
roots = {
    'zipcodes': dataset,
    'states': table.mutate({'indices': ibis.row_number()}).order_by('state', 'county'),
    'zip_db': ds.dataset(fixtures / 'zip_db.parquet'),
}
app = GraphQL.federated(roots, keys={'zipcodes': ['zipcode']})
