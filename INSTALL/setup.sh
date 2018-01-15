#!/usr/bin/env bash
rm -rf ../venv
virtualenv --system-site-packages -p python3 ../venv
source ../venv/bin/activate

pip3 install --upgrade matplotlib
pip3 install --upgrade tensorflow-gpu
pip3 install --upgrade tqdm
pip3 install --upgrade pydot
pip3 install --upgrade pandas
pip3 install --upgrade git+https://github.com/UIA-CAIR/py_image_stitcher.git
pip3 install --upgrade git+https://github.com/UIA-CAIR/gym-deeplinewars.git
pip3 install --upgrade git+https://github.com/UIA-CAIR/gym-deeprts.git
pip3 install --upgrade git+https://github.com/CAIR-UIA/gym-maze.git
pip3 install --upgrade git+https://github.com/UIA-CAIR/gym-flashrl.git
pip3 install --upgrade git+https://github.com/ntasfi/PyGame-Learning-Environment.git


echo "Download cuDNN 6.0 from following url:"
echo "URL #1: https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v6/prod/8.0_20170307/Ubuntu16_04_x64/libcudnn6_6.0.20-1+cuda8.0_amd64-deb"
echo "URL #2: https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v6/prod/8.0_20170307/Ubuntu16_04_x64/libcudnn6-dev_6.0.20-1+cuda8.0_amd64-deb"
echo "-------------------------"