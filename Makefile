format-code:
	autopep8 -i arc_prize_2024_sebhtml.py

train-model:
	python arc_prize_2024_sebhtml.py | tee log

pip-freeze:
	pip freeze > requirements.txt

venv-activate:
	source ./my-venv/bin/activate

kaggle-push:
	kaggle kernels push


kaggle-status:
	kaggle kernels status -k sebastien/arc-prize-2024-sebhtml-py


kaggle-output:
	kaggle kernels output -k sebastien/arc-prize-2024-sebhtml-py

