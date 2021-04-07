#!/bin/bash

curl https://pyenv.run | bash

## echo these into ~/.bashrc too
export PATH="/home/$USER/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

pyenv install 3.7.9
pyenv global 3.7.9
python -m venv /opt/venv/sb2
source /opt/venv/sb2/bin/activate

pip install tensorflow-gpu==1.15.0
pip install stable-baselines
pip install gym_fishing

