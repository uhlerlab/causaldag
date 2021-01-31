#!/usr/bin/env

python3 -m venv venv
source venv/bin/activate
pip3 install ipython numpy numexpr ipdb tqdm pygam scikit-learn networkx conditional_independence twine wheel
pip3 install jedi==0.17.2