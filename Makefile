format-code:
	autopep8 -i *.py

train-model:
	python main.py | tee log

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

kaggle-files:
	kaggle kernels files -k sebastien/arc-prize-2024-sebhtml-py

h5dump:
	h5dump /workspace/train_datasets/3aa6fb7a.hdf5 |less

learn:
	./learn.sh

lesslog:
	./lesslog.sh

