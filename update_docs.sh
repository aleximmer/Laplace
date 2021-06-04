pdoc --html -o html --template-dir template laplace --force
python regression_example.py
mv docs/laplace/* docs/
rm -r docs/laplace/