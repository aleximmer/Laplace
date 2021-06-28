# Laplace

Main: [![Main](https://travis-ci.com/AlexImmer/Laplace.svg?token=rpuRxEjQS6cCZi7ptL9y&branch=main)](https://travis-ci.com/AlexImmer/Laplace)
Dev: [![Dev](https://travis-ci.com/AlexImmer/Laplace.svg?token=rpuRxEjQS6cCZi7ptL9y&branch=dev)](https://travis-ci.com/AlexImmer/Laplace)

The documentation is available at [https://aleximmer.github.io/Laplace/index.html](https://aleximmer.github.io/Laplace/index.html).

## Setup

We assume `python3.8` since the package was developed with that version.

```bash
pip install -r requirements.txt
# for development
pip install -e .
# for "production"
pip install .


# run tests
pip install -r tests/requirements.txt
pytest tests/
```

## [Documentation](https://aleximmer.github.io/Laplace/index.html)

```bash
pip install pdoc3 matplotlib
# create docs and write to html
bash update_docs.sh
# .. or serve the docs directly
pdoc --http 0.0.0.0:8080 laplace --template-dir template
```
