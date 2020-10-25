all:
	cythonize -aX linetrace=True graphique/*.pyx
	python3 setup.py build_ext -i --define CYTHON_TRACE_NOGIL

check: all
	python3 setup.py $@ -ms
	black -q --$@ .
	flake8
	flake8 graphique/*.pyx --ignore E999,E211
	mypy -p graphique
	pytest --cov --cov-fail-under=100

docs: all
	PYTHONPATH=$(PWD) mkdocs build

dist:
	python3 setup.py sdist bdist_wheel
	docker run --rm -v $(PWD):/usr/src -w /usr/src quay.io/pypa/manylinux2014_x86_64 make cp37 cp38

cp37:
	/opt/python/$@-$@m/bin/pip wheel . -w dist
	auditwheel repair dist/*$@m-linux_x86_64.whl

cp38:
	/opt/python/$@-$@/bin/pip wheel . -w dist
	auditwheel repair dist/*$@-linux_x86_64.whl
