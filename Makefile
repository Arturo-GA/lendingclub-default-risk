.PHONY: data train tune explain leakage deck test clean-nb all

data:
	python scripts/make_dataset.py

train:
	python scripts/train.py

tune:
	python scripts/tune.py --model lightgbm --trials 40

explain:
	python scripts/explain.py --model lightgbm

leakage:
	python scripts/leakage_ablation.py

deck:
	python scripts/make_deck.py

test:
	pytest -q

# Quita las salidas de los notebooks antes de hacer commit (evita archivos de cientos de MB)
clean-nb:
	jupyter nbconvert --clear-output --inplace notebooks/*.ipynb

all: data train explain leakage deck
