.PHONY: help install fetch splits edges clean-edges test all clean

PYTHON ?= python3
SEASONS ?= 2023-24 2024-25 2025-26

help:
	@echo "Targets:"
	@echo "  make install      Install Python dependencies"
	@echo "  make fetch        Cache player-game logs to data/*.parquet"
	@echo "  make splits       Compute W/O splits (incl. true USG%) → out/splits.parquet"
	@echo "  make usage        Show biggest USG% jumps when a teammate sits"
	@echo "  make validate     Out-of-sample check on a held-out season"
	@echo "  make project      Price a prop line from a projected distribution"
	@echo "  make edges        Run edge finder (default filters) → out/edges.csv"
	@echo "  make clean-edges  Same, with --clean-only (no minutes confound)"
	@echo "  make test         Run pytest suite"
	@echo "  make all          install → fetch → splits → edges → test"
	@echo "  make clean        Remove caches and outputs"
	@echo ""
	@echo "Variables:"
	@echo "  PYTHON   Python interpreter (default: python3)"
	@echo "  SEASONS  Seasons to fetch (default: $(SEASONS))"

install:
	$(PYTHON) -m pip install -r requirements.txt

fetch:
	$(PYTHON) -m src.fetch --seasons $(SEASONS) --types "Regular Season" "Playoffs"

splits:
	$(PYTHON) -m src.splits

usage:
	$(PYTHON) -m src.edges --stat USG --min-z 2.5 --top 25

validate:
	$(PYTHON) -m src.validate

project:
	$(PYTHON) -m src.project --player "Jaylen Brown" --teammate "Tatum" --stat RA --line 9.5 --over -110 --under -110

edges:
	$(PYTHON) -m src.edges --top 25

clean-edges:
	$(PYTHON) -m src.edges --clean-only --min-z 2.5 --min-pct 0.15 --top 25

test:
	$(PYTHON) -m pytest tests/ -v

all: install fetch splits edges test

clean:
	rm -rf data/*.parquet out/
