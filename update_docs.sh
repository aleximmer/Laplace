rm -rf docs/
pdoc --html --output-dir docs --template-dir template --force laplace
python examples/regression_example.py
mv docs/laplace/* docs/
rm -rf docs/laplace/
