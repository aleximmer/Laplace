# Laplace

Main: [![Main](https://travis-ci.com/AlexImmer/Laplace.svg?token=rpuRxEjQS6cCZi7ptL9y&branch=main)](https://travis-ci.com/AlexImmer/Laplace)
Dev: [![Dev](https://travis-ci.com/AlexImmer/Laplace.svg?token=rpuRxEjQS6cCZi7ptL9y&branch=dev)](https://travis-ci.com/AlexImmer/Laplace)

The documentation is available at [https://d2ca8dae-c541-11eb-9203-acde48001122.github.io/lapras/index.html](https://d2ca8dae-c541-11eb-9203-acde48001122.github.io/lapras/index.html).

## Setup

```bash
pip install -r requirements.txt
# for development
pip install -e .
# for "production"
pip install .

# for asdfghjkl backend
pip install dependencies/asdfghjkl-0.0.1-py3-none-any.whl

# run tests
pip install pytest
pytest tests/
```

## [Documentation](https://d2ca8dae-c541-11eb-9203-acde48001122.github.io/lapras/index.html)

```bash
pip install pdoc
# create docs and write to html
bash update_docs.sh
# .. or serve the docs directly
pdoc --http 0.0.0.0:8080 laplace --template-dir template
```
