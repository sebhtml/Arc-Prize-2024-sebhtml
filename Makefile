autopep8:
	autopep8 -i arc_prize_2024_sebhtml.py

train:
	python arc_prize_2024_sebhtml.py | tee log

freeze:
	pip freeze > requirements.txt

activate:
	source ./my-venv/bin/activate
