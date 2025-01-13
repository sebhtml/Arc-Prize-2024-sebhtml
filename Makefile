format-code:
	autopep8 -i src/*.py tests/*.py

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

learn:
	nohup ./learn.sh

less-log:
	./less-log.sh

check-result:
	./check-result.sh

watch-log:
	./watch-log.sh

setup:
	bash install-programs.sh
	bash link-git-config.sh
	bash link-ssh-key.sh
