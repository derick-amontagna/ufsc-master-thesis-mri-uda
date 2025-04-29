black:
	git ls-files '*.py' | poetry run black --check .

poetry-create-env:
	poetry config virtualenvs.in-project true; poetry install

update-env-poetry:
    export PATH="$HOME/.local/bin:$PATH"

update-kernal:
    python -m ipykernel install --user --name=py3115

run-model:
	python main.py --exp_name ge_siemens_mcc_auc
