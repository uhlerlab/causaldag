#!/usr/bin/env

python3 -m venv venv
source venv/bin/activate
pip3 install numpy numexpr tqdm pygam scikit-learn networkx
pip3 install conditional_independence graphical_models graphical_model_learning
pip3 install twine wheel ipdb ipython
pip3 install jedi==0.17.2

# REPLACE WITH PATH TO OTHER PACKAGES
pip3 install -e ~/Documents/packages/conditional_independence/
pip3 install -e ~/Documents/packages/graphical_models/
pip3 install -e ~/Documents/packages/graphical_model_learning/
