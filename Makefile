dvc.yaml: gen-dvc-yaml.py
	python gen-dvc-yaml.py > dvc.yaml

repro:
	dvc repro
