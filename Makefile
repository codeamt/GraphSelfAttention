install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

test:
	python -m pytest -vv --cov=main --cov=src src/tests/test_*.py

lint:
	pylint --disable=R,C --ignore-patterns=src/tests/test_.*?py *.py src/*.py

#container-lint:
#	docker run --rm -i hadolint/hadolint < docker/Dockerfile

all: install lint test #container-lint