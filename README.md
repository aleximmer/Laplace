# Laplace

Main: [![Main](https://travis-ci.com/AlexImmer/Laplace.svg?token=rpuRxEjQS6cCZi7ptL9y&branch=main)](https://travis-ci.com/AlexImmer/Laplace)
Dev: [![Dev](https://travis-ci.com/AlexImmer/Laplace.svg?token=rpuRxEjQS6cCZi7ptL9y&branch=dev)](https://travis-ci.com/AlexImmer/Laplace)

The documentation is available at [https://d2ca8dae-c541-11eb-9203-acde48001122.github.io/lapras/index.html](https://d2ca8dae-c541-11eb-9203-acde48001122.github.io/lapras/index.html).

## Setup

We assume `python3.8` since the package was developed with that version.

```bash
pip install -r requirements.txt
# for development
pip install -e .
# for "production"
pip install .

# for the ASDL backend
git clone https://github.com/kazukiosawa/asdfghjkl.git
cd asdfghjkl
pip install -r requirements.txt
pip install .


# run tests
pip install pytest
pytest tests/
```

## [Documentation](https://d2ca8dae-c541-11eb-9203-acde48001122.github.io/lapras/index.html)

```bash
pip install pdoc3 matplotlib
# create docs and write to html
bash update_docs.sh
# .. or serve the docs directly
pdoc --http 0.0.0.0:8080 laplace --template-dir template
```
