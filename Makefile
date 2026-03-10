PYTHON ?= python
PIP ?= pip

install:
	$(PIP) install -r requirements/dev.txt

test:
	PYTHONPATH=src $(PYTHON) -m pytest -q

smoke:
	PYTHONPATH=src $(PYTHON) scripts/smoke_mujoco.py

smoke-exp001:
	PYTHONPATH=src $(PYTHON) scripts/smoke_exp001.py

train-exp001:
	PYTHONPATH=src $(PYTHON) scripts/train.py --config configs/experiment/exp001_mock.yaml

eval-latest:
	PYTHONPATH=src $(PYTHON) scripts/eval.py --run-dir runs/latest

report-latest:
	PYTHONPATH=src $(PYTHON) scripts/make_report.py --run-dir runs/latest

docker-build:
	docker compose build

docker-smoke:
	docker compose run --rm app python scripts/smoke_mujoco.py

docker-train-exp001:
	docker compose run --rm app python scripts/train.py --config configs/experiment/exp001_mock.yaml
