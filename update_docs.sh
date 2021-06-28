rm -rf docs/
pdoc --html -o docs --template-dir template laplace --force
python regression_example.py
mv docs/laplace/* docs/
rm -rf docs/laplace/
