test:
	PYTHONPATH=./ python -m pytest tests --cov-config .coveragerc --cov-report html --cov-report term --cov=. -v

test_slow:
	PYTHONPATH=./ python -m pytest tests --cov-config .coveragerc --cov-report html --cov-report term --cov=. -v --runslow