set -e
rm -rf ./docs/laplace
pdoc --html --output-dir ./docs --template-dir ./docs/template --force laplace
cp ./docs/*.png ./docs/laplace
cp -r ./docs/template ./docs/laplace

