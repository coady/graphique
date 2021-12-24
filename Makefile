check:
	python3 -m pytest -s --cov

lint:
	black --check .
	flake8
	mypy -p graphique

html: docs/schema.md
	python3 -m mkdocs build

docs/schema.md: docs/schema.graphql
	./node_modules/.bin/graphql-markdown \
		--title "Example Schema" \
		--no-toc \
		--prologue "Generated from a test fixture of zipcodes." \
		$? > $@

docs/schema.graphql: graphique/*.py
	python3 -m graphique.schema tests/fixtures/zipcodes.parquet > $@
