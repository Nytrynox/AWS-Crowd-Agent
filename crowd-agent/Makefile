.PHONY: venv install lambda-zip

venv:
	python3 -m venv .venv
	. .venv/bin/activate && pip install -U pip

install:
	. .venv/bin/activate && pip install -r requirements.txt

lambda-zip:
	cd lambda && pip install -r requirements.txt -t . && zip -r handler.zip .
